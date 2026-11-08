"""Run the agent over a questions.jsonl and write submission.jsonl (+ per-question logs).

Resumable: ids already in --out are skipped, so an interrupted run just continues.
At the end the output is rewritten with one row per question, abstain for any
question that has no answer yet, so the file is always upload-ready.

Example (Colab, GPU):
    python -m code_member_d.run_all --questions public_test/questions.jsonl \
        --cache-root cache --split public_test --out outputs/submission.jsonl \
        --adapter /content/drive/MyDrive/.../final_adapter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from code_member_b.executor import DuckDBExecutor
from code_member_d.inference import ContextCache, Generator, answer_batch, answer_question
from code_member_d.make_baseline_submission import read_question_ids, validate_submission
from code_member_d.run_manifest import append_run, build_run_record, finish_run


def read_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_done(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # half-written last line after a crash
                    continue
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    rows[row["id"]] = row
    return rows


def _terminate_partial_line(path: Path) -> None:
    """If a crash left the last line without a newline, end it so the next append
    starts on a fresh line instead of gluing two records together."""
    if path.is_file() and path.stat().st_size:
        with path.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")


def _durable_flush(handle) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())  # best effort; Drive mounts may ignore it
    except OSError:
        pass


def shard_businesses(shards_file: Path, shard: int) -> set[str]:
    data = json.loads(shards_file.read_text(encoding="utf-8"))
    for item in data["shards"]:
        if int(item["shard"]) == shard:
            return {b["business_id"] for b in item["businesses"]}
    raise SystemExit(f"shard {shard} not found in {shards_file}")


def write_final(path: Path, ids: list[str], rows: dict[str, dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in ids:
            row = rows.get(qid, {"id": qid, "type": "abstain"})
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run(
    questions: list[dict[str, Any]],
    generator: Generator,
    contexts: ContextCache,
    out: Path,
    log_path: Path,
    *,
    max_attempts: int = 2,
    rules_chars: int = 1600,
    unanswerable_policy: str = "keep",
    progress_every: int = 1,
    redo_abstain: bool = False,
    batch_size: int = 1,
    reject_suspicious: bool = True,
) -> dict[str, dict[str, Any]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    _terminate_partial_line(out)
    _terminate_partial_line(log_path)
    # Normally a question with any row is finished. --redo-abstain also re-runs the
    # abstained ones (e.g. after fixing GPU out-of-memory); the new row, appended
    # later in the file, replaces the old one when the file is read back.
    todo = [q for q in questions
            if str(q["id"]) not in done
            or (redo_abstain and done[str(q["id"])].get("type") == "abstain")]
    print(f"{len(questions) - len(todo)} already done, {len(todo)} to run", flush=True)
    executor = DuckDBExecutor(max_rows=1000, timeout_seconds=30.0)
    started = time.perf_counter()
    batching = batch_size > 1 and hasattr(generator, "generate_batch")
    options = dict(max_attempts=max_attempts, rules_chars=rules_chars,
                   unanswerable_policy=unanswerable_policy, reject_suspicious=reject_suspicious)
    with out.open("a", encoding="utf-8", newline="\n") as sub, \
            log_path.open("a", encoding="utf-8", newline="\n") as logf:
        finished = 0

        def emit(row: dict[str, Any], log: dict[str, Any]) -> None:
            nonlocal finished
            sub.write(json.dumps(row, ensure_ascii=False) + "\n")
            _durable_flush(sub)
            logf.write(json.dumps(log, ensure_ascii=False, default=str) + "\n")
            _durable_flush(logf)
            done[row["id"]] = row
            finished += 1

        def progress(row: dict[str, Any], log: dict[str, Any]) -> None:
            spent = time.perf_counter() - started
            eta = spent / finished * (len(todo) - finished)
            kinds: dict[str, int] = {}
            for r in done.values():
                kinds[r["type"]] = kinds.get(r["type"], 0) + 1
            print(f"[{finished}/{len(todo)}] {row['type']:<12} {log['elapsed_seconds']:>5.1f}s "
                  f"| total {spent/60:.1f} min, ETA {eta/60:.1f} min | {kinds}", flush=True)

        if batching:
            for begin in range(0, len(todo), batch_size):
                chunk = todo[begin: begin + batch_size]
                for row, log in answer_batch(chunk, contexts, generator, executor, **options):
                    emit(row, log)
                progress(row, log)  # one line per chunk: the engine works on all of them at once
        else:
            for question in todo:
                row, log = answer_question(question, contexts, generator, executor, **options)
                emit(row, log)
                if finished % progress_every == 0 or finished == len(todo):
                    progress(row, log)
    return done


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--cache-root", default="cache", type=Path)
    p.add_argument("--split", default="public_test")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--logs", type=Path, default=None, help="default: <out dir>/logs/<out name>.jsonl")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (final_adapter); omit for base model")
    p.add_argument("--shard", type=int, default=None, help="only run businesses of this shard")
    p.add_argument("--shards-file", type=Path, default=None, help="default: <cache-root>/<split>/shards.json")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--rules-chars", type=int, default=1600,
                   help="rulebook text budget in the prompt (B trained with 1600; each extra "
                        "1000 chars is ~350 tokens and forces more shrinking elsewhere)")
    p.add_argument("--unanswerable", choices=["abstain", "keep"], default="abstain",
                   help="abstain (default): score 0 when the model says unanswerable; "
                        "keep: submit its verdict (wrong verdicts cost -1)")
    p.add_argument("--max-input-tokens", type=int, default=4096,
                   help="prompt cut (middle dropped) above this; lower it on small GPUs")
    p.add_argument("--fallback-input-tokens", type=int, default=2560,
                   help="shorter cut used once if the first attempt runs out of GPU memory")
    p.add_argument("--redo-abstain", action="store_true",
                   help="also re-run questions whose saved row is abstain "
                        "(use after fixing a crash/OOM; same --out)")
    p.add_argument("--allow-suspicious", action="store_true",
                   help="keep answers that are 0 / an empty list / an empty date instead of "
                        "retrying then abstaining (they are almost always wrong)")
    p.add_argument("--progress-every", type=int, default=1,
                   help="print a progress line every N questions")
    p.add_argument("--engine", choices=["hf", "vllm"], default="hf",
                   help="hf: transformers, one question at a time (Windows ok); "
                        "vllm: continuous batching (Linux only: Colab/Kaggle/Docker)")
    p.add_argument("--batch-size", type=int, default=0,
                   help="questions per batch (vllm only); 0 = 32 for vllm")
    p.add_argument("--vllm-dtype", default="half", help="half works on every GPU (T4 has no bf16)")
    p.add_argument("--vllm-quantization", default=None, choices=[None, "bitsandbytes", "awq", "gptq"],
                   help="bitsandbytes = 4-bit like B's training; needed for 7B on a 16 GB T4")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="2 to use both RTX 5090s")
    a = p.parse_args(argv)

    all_questions = read_questions(a.questions)
    ids_all = [str(q["id"]) for q in all_questions]
    questions = all_questions
    if a.shard is not None:
        shards_file = a.shards_file or a.cache_root / a.split / "shards.json"
        keep = shard_businesses(shards_file, a.shard)
        questions = [q for q in questions if q["business_id"] in keep]
    if a.limit:
        questions = questions[: a.limit]
    log_path = a.logs or a.out.parent / "logs" / f"{a.out.stem}.jsonl"

    if a.engine == "vllm":
        from code_member_d.vllm_generator import VllmGenerator  # Linux + CUDA only

        generator = VllmGenerator(
            model_id=a.model_id, adapter_path=a.adapter, max_input_tokens=a.max_input_tokens,
            dtype=a.vllm_dtype, quantization=a.vllm_quantization,
            gpu_memory_utilization=a.gpu_memory_utilization,
            tensor_parallel_size=a.tensor_parallel_size,
        )
        batch_size = a.batch_size or 32
    else:
        from code_member_d.planner import QwenAdapterGenerator  # heavy import, GPU only

        generator = QwenAdapterGenerator(
            model_id=a.model_id, adapter_path=a.adapter,
            max_input_tokens=a.max_input_tokens, fallback_input_tokens=a.fallback_input_tokens,
        )
        batch_size = 1
    contexts = ContextCache(a.cache_root, a.split)
    manifest_path = a.out.with_suffix(".manifest.json")
    record = build_run_record(dict(vars(a), batch_size=batch_size), Path(__file__).resolve().parents[1],
                              a.questions, a.cache_root, a.split)
    for warning in append_run(manifest_path, record):
        print("WARNING:", warning, flush=True)
    done = run(questions, generator, contexts, a.out, log_path,
               max_attempts=a.max_attempts, rules_chars=a.rules_chars,
               unanswerable_policy=a.unanswerable, redo_abstain=a.redo_abstain,
               progress_every=max(1, a.progress_every), batch_size=batch_size,
               reject_suspicious=not a.allow_suspicious)

    partial = a.shard is not None or a.limit is not None
    write_final(a.out, ids_all if not partial else [i for i in ids_all if i in done], done)
    problems = validate_submission(a.out, read_question_ids(a.questions)) if not partial else []
    kinds: dict[str, int] = {}
    for row in done.values():
        kinds[row["type"]] = kinds.get(row["type"], 0) + 1
    finish_run(manifest_path, {"counts": kinds, "questions_in_this_run": len(questions)})
    print("DONE", kinds, flush=True)
    for problem in problems[:20]:
        print("PROBLEM:", problem, flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
