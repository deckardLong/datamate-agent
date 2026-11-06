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
from code_member_d.inference import ContextCache, Generator, answer_question
from code_member_d.make_baseline_submission import read_question_ids, validate_submission


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
    rules_chars: int = 3500,
    progress_every: int = 10,
) -> dict[str, dict[str, Any]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out)
    todo = [q for q in questions if str(q["id"]) not in done]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)
    executor = DuckDBExecutor(max_rows=1000, timeout_seconds=30.0)
    started = time.perf_counter()
    with out.open("a", encoding="utf-8", newline="\n") as sub, \
            log_path.open("a", encoding="utf-8", newline="\n") as logf:
        for index, question in enumerate(todo, start=1):
            row, log = answer_question(
                question, contexts, generator, executor,
                max_attempts=max_attempts, rules_chars=rules_chars,
            )
            sub.write(json.dumps(row, ensure_ascii=False) + "\n")
            sub.flush()
            logf.write(json.dumps(log, ensure_ascii=False, default=str) + "\n")
            logf.flush()
            done[row["id"]] = row
            if index % progress_every == 0 or index == len(todo):
                spent = time.perf_counter() - started
                eta = spent / index * (len(todo) - index)
                kinds: dict[str, int] = {}
                for r in done.values():
                    kinds[r["type"]] = kinds.get(r["type"], 0) + 1
                print(f"[{index}/{len(todo)}] {spent/60:.1f} min, ETA {eta/60:.1f} min | {kinds}",
                      flush=True)
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
    p.add_argument("--rules-chars", type=int, default=3500,
                   help="rulebook text budget in the prompt (B trained with 1600)")
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

    from code_member_d.planner import QwenAdapterGenerator  # heavy import, GPU only

    generator = QwenAdapterGenerator(model_id=a.model_id, adapter_path=a.adapter)
    contexts = ContextCache(a.cache_root, a.split)
    done = run(questions, generator, contexts, a.out, log_path,
               max_attempts=a.max_attempts, rules_chars=a.rules_chars)

    partial = a.shard is not None or a.limit is not None
    write_final(a.out, ids_all if not partial else [i for i in ids_all if i in done], done)
    problems = validate_submission(a.out, read_question_ids(a.questions)) if not partial else []
    kinds: dict[str, int] = {}
    for row in done.values():
        kinds[row["type"]] = kinds.get(row["type"], 0) + 1
    print("DONE", kinds, flush=True)
    for problem in problems[:20]:
        print("PROBLEM:", problem, flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
