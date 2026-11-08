from __future__ import annotations

import json

import duckdb
import pytest

from code_member_b.executor import DuckDBExecutor
from code_member_d.inference import ContextCache, answer_question, build_prompt, render_rules
from code_member_d.make_baseline_submission import main as submission_main
from code_member_d.run_all import read_done, read_questions, run, write_final


@pytest.fixture()
def cache(tmp_path):
    biz = tmp_path / "cache" / "public_test" / "dev_x"
    biz.mkdir(parents=True)
    con = duckdb.connect(str(biz / "db.duckdb"))
    con.execute("CREATE TABLE ban_hang (ngay VARCHAR, thanh_tien DOUBLE)")
    con.execute("INSERT INTO ban_hang VALUES ('2026-01-01', 10), ('2026-01-02', 32)")
    con.close()
    profile = {
        "business_id": "dev_x",
        "tables": {"ban_hang": {
            "row_count": 2,
            "columns": {"ngay": {"type": "VARCHAR"}, "thanh_tien": {"type": "DOUBLE"}},
            "samples": [{"ngay": "2026-01-01", "thanh_tien": 10.0}],
        }},
        "readme_notes": "thanh_tien la thanh tien dong hang",
    }
    (biz / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (biz / "schema_card.md").write_text("# dev_x", encoding="utf-8")
    rules = {"topics": {"revenue": {"rules": [
        {"text": "Luật Mặc định", "section": "s"},
        {"text": "Đơn tính vào doanh thu là đơn **không huỷ** và không trả hàng.", "section": "s"},
    ], "text": ""}}}
    (biz.parent / "rules.json").write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    return ContextCache(tmp_path / "cache", "public_test")


class Scripted:
    model_id = "scripted"

    def __init__(self, *plans):
        self.plans, self.prompts = list(plans), []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.plans.pop(0)


def _q(qid="q1"):
    return {"id": qid, "business_id": "dev_x", "question": "tổng doanh thu?", "as_of_date": "2026-01-31"}


def _answer(cache, generator, **kw):
    return answer_question(_q(), cache, generator, DuckDBExecutor(), **kw)


def test_render_rules_skips_header_rows_and_duplicates():
    rules = {"topics": {
        "revenue": {"rules": [{"text": "Luật Mặc định"},
                              {"text": "Đơn tính vào doanh thu là đơn không huỷ và không trả."}]},
        "cancel_return": {"rules": [{"text": "Luật Mặc định"},
                                    {"text": "Đơn tính vào doanh thu là đơn không huỷ và không trả."}]},
    }}
    text = render_rules(rules, 1000)
    assert "Luật Mặc định" not in text
    assert text.count("Đơn tính vào doanh thu") == 1
    assert len(render_rules(rules, 20)) <= 20


def test_prompt_carries_real_rules_and_business_notes(cache):
    from code_member_b.pseudo_label import TrainingRecord

    record = TrainingRecord("q1", "dev_x", "tổng doanh thu?", "2026-01-31", "unknown", None, [])
    prompt = build_prompt(record, cache.get("dev_x"), 3500)
    assert "Đơn tính vào doanh thu" in prompt and "không huỷ" in prompt
    assert "thanh tien dong hang" in prompt  # README notes
    assert "Required answer_type" not in prompt  # same shape B trained on


def test_reason_is_always_a_clean_string(cache, tmp_path):
    from code_member_d.inference import DEFAULT_REASON, clean_reason

    assert clean_reason("Không có cột giá vốn nên không tính được lợi nhuận") == \
        "Không có cột giá vốn nên không tính được lợi nhuận"
    for bad in (None, "", 123, ["a"], {"x": 1}, '{"query": "SELECT 1"}',
                "SELECT SUM(x) FROM t", "invalid_json: {...", "x" * 400):
        assert clean_reason(bad) == DEFAULT_REASON
    row, _ = _answer(cache, Scripted({"answer_type": "unanswerable", "reason": {"k": 1}}))
    assert row["type"] == "unanswerable" and row["reason"] == DEFAULT_REASON

    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"id": "a", "business_id": "b"}) + "\n", encoding="utf-8")
    bad_sub = tmp_path / "bad.jsonl"
    bad_sub.write_text(json.dumps({"id": "a", "type": "unanswerable", "reason": ["x"]}) + "\n",
                       encoding="utf-8")
    assert submission_main(["--questions", str(q), "--validate", str(bad_sub)]) == 1


def test_nan_or_infinity_never_reaches_the_submission(cache, tmp_path):
    nan_sql = "SELECT CAST('NaN' AS DOUBLE) FROM ban_hang LIMIT 1"
    row, log = _answer(cache, Scripted({"answer_type": "number", "sql": nan_sql},
                                       {"answer_type": "number", "sql": nan_sql}))
    assert row == {"id": "q1", "type": "abstain"} and "non-finite" in log["attempts"][-1]["error"]
    # ...and the validator rejects such a file if one is ever produced
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"id": "a", "business_id": "b"}) + "\n", encoding="utf-8")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "type": "number", "answer": NaN}\n', encoding="utf-8")
    assert submission_main(["--questions", str(q), "--validate", str(bad)]) == 1


def test_success_number(cache):
    gen = Scripted({"answer_type": "number", "sql": "SELECT SUM(thanh_tien) FROM ban_hang"})
    row, log = _answer(cache, gen)
    assert row == {"id": "q1", "type": "number", "answer": 42}
    assert log["attempts"][0]["sql"].startswith("SELECT SUM")


def test_type_alias_and_percent_rounding(cache):
    gen = Scripted({"answer_type": "percentage", "sql": "SELECT 100.0 * 1 / 3 FROM ban_hang LIMIT 1"})
    assert _answer(cache, gen)[0] == {"id": "q1", "type": "percent", "answer": 33.33}


def test_unanswerable_passes_reason(cache):
    row, _ = _answer(cache, Scripted({"answer_type": "unanswerable", "reason": "không có giá vốn"}))
    assert row == {"id": "q1", "type": "unanswerable", "reason": "không có giá vốn"}


def test_unanswerable_policy_abstain_scores_zero_instead_of_risking(cache):
    gen = Scripted({"answer_type": "unanswerable", "reason": "x"})
    row, log = _answer(cache, gen, unanswerable_policy="abstain")
    assert row == {"id": "q1", "type": "abstain"}
    assert "policy=abstain" in log["attempts"][0]["note"]


def test_policy_can_be_flipped_from_the_log_without_rerunning(cache, tmp_path):
    from code_member_d.apply_unanswerable_policy import apply_policy

    out, log = tmp_path / "sub.jsonl", tmp_path / "log.jsonl"
    qs = [_q("q1"), _q("q2"), _q("q3")]
    plans = [
        {"answer_type": "unanswerable", "reason": "thiếu giá vốn"},        # verdict -> abstain
        {"answer_type": "number", "sql": "SELECT COUNT(*) FROM ban_hang"},  # real answer
        {"answer_type": "number", "sql": "DROP TABLE ban_hang"},            # SQL failure -> abstain
        {"answer_type": "number", "sql": "DROP TABLE ban_hang"},
    ]
    run(qs, Scripted(*plans), cache, out, log, unanswerable_policy="abstain")
    kept = tmp_path / "keep.jsonl"
    changed, total = apply_policy(out, log, kept, "keep")
    rows = {r["id"]: r for r in map(json.loads, kept.read_text(encoding="utf-8").splitlines())}
    assert (changed, total) == (1, 3)
    assert rows["q1"] == {"id": "q1", "type": "unanswerable", "reason": "thiếu giá vốn"}
    assert rows["q2"]["answer"] == 2
    assert rows["q3"] == {"id": "q3", "type": "abstain"}  # a failed SQL is not a verdict
    back = tmp_path / "back.jsonl"
    assert apply_policy(kept, log, back, "abstain")[0] == 1
    assert json.loads(back.read_text(encoding="utf-8").splitlines()[0]) == {"id": "q1", "type": "abstain"}


def test_retry_after_error_then_success(cache):
    gen = Scripted(
        {"answer_type": "number", "sql": "SELECT SUM(khong_co) FROM ban_hang"},
        {"answer_type": "number", "sql": "SELECT COUNT(*) FROM ban_hang"},
    )
    row, log = _answer(cache, gen, max_attempts=2)
    assert row["answer"] == 2 and len(log["attempts"]) == 2
    assert "CORRECTION REQUIRED" in gen.prompts[1]


def test_all_attempts_fail_becomes_abstain(cache):
    bad = {"answer_type": "number", "sql": "DROP TABLE ban_hang"}
    row, log = _answer(cache, Scripted(bad, bad), max_attempts=2)
    assert row == {"id": "q1", "type": "abstain"}
    assert "error" in log["attempts"][-1]


def test_unknown_business_abstains_without_raising(cache):
    q = {**_q(), "business_id": "nope"}
    row, log = answer_question(q, cache, Scripted(), DuckDBExecutor())
    assert row["type"] == "abstain" and "setup" in log["error"]


def test_run_resumes_and_final_file_fills_abstain(cache, tmp_path):
    out, log = tmp_path / "sub.jsonl", tmp_path / "log.jsonl"
    questions = [_q("q1"), _q("q2"), _q("q3")]
    ok = {"answer_type": "number", "sql": "SELECT COUNT(*) FROM ban_hang"}
    run(questions[:1], Scripted(ok), cache, out, log)
    done = run(questions, Scripted(ok, ok), cache, out, log)  # q1 must be skipped
    assert sorted(done) == ["q1", "q2", "q3"]
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3

    write_final(out, ["q1", "q2", "q3", "q4"], done)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["q1", "q2", "q3", "q4"]
    assert rows[-1] == {"id": "q4", "type": "abstain"}


def _sectioned_prompt():
    from code_member_d import prompt_budget as pb

    schema = "\n".join(f'TABLE "t{i}" (a INT, b INT)' for i in range(20))
    return "\n\n".join([
        "You generate one safe read-only DuckDB SELECT query.",
        pb.H_RULES + "\n".join(f"- rule number {i} " + "w " * 10 for i in range(30)),
        pb.H_NOTES + "README note " * 30,
        pb.H_NAMES + ", ".join(f'"t{i}"' for i in range(60)),
        pb.H_SAMPLES + "\n".join(f'"t{i}": {{"a":1}}' for i in range(20)),
        pb.H_SCHEMA + schema,
        pb.H_JOINS + "\n".join(f'"t{i}" <-> "t{i+1}": a' for i in range(30)),
        pb.END_MARKER + " group by label and id, then LIMIT.",
        "Business: dev_x\nQuestion: tổng doanh thu tháng 6?",
    ]), schema


def test_fit_prompt_shrinks_low_value_sections_and_never_touches_schema_or_question():
    from code_member_d.prompt_budget import fit_prompt

    count = lambda text: len(text.split())
    text, schema = _sectioned_prompt()
    assert fit_prompt(text, count, 10_000) == text  # already fits: unchanged
    target = count(text) - 100
    fitted = fit_prompt(text, count, target)
    assert count(fitted) <= target
    assert schema in fitted                      # exact schema intact
    assert fitted.endswith("Question: tổng doanh thu tháng 6?")
    assert "You generate one safe read-only DuckDB SELECT query." in fitted
    # join hints are the first thing to go
    assert fitted.count("<->") < text.count("<->")
    # an impossible target still returns the best effort with schema and question intact
    tiny = fit_prompt(text, count, 5)
    assert schema in tiny and tiny.endswith("tháng 6?")
    # a prompt with a different layout is returned untouched
    assert fit_prompt("no sections here " * 50, count, 5) == "no sections here " * 50


def test_planner_fit_uses_section_shrinking_before_cutting_the_middle():
    from code_member_d.planner import QwenAdapterGenerator

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    gen = QwenAdapterGenerator.__new__(QwenAdapterGenerator)
    gen.tokenizer = Tok()
    text, schema = _sectioned_prompt()
    limit = len(text.split()) - 100
    ids = gen._fit(text, limit)
    assert len(ids) <= limit
    assert " ".join(ids).count("TABLE") == 20    # all 20 schema lines survived, nothing cut blindly


def test_long_prompt_is_cut_in_the_middle_keeping_instruction_and_question():
    from code_member_d.planner import QwenAdapterGenerator

    class Tok:  # one token per word
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    gen = QwenAdapterGenerator.__new__(QwenAdapterGenerator)  # skip the GPU model load
    gen.tokenizer = Tok()
    words = ["FORMAT-INSTRUCTION"] + [f"mid{i}" for i in range(500)] + ["THE-QUESTION"]
    ids = gen._fit(" ".join(words), 100)
    assert len(ids) == 100
    assert ids[0] == "FORMAT-INSTRUCTION" and ids[-1] == "THE-QUESTION"
    assert ids[: round(100 * 0.6)] == words[: round(100 * 0.6)]  # 60% head kept
    assert gen._fit("a b c", 100) == ["a", "b", "c"]  # short prompts untouched


class BatchEngine:
    """Content-based fake of a batching engine (like vLLM): behaves by question text."""

    model_id = "batch-fake"
    GOOD = {"answer_type": "number", "sql": "SELECT COUNT(*) FROM ban_hang"}
    BAD = {"answer_type": "number", "sql": "SELECT SUM(khong_co) FROM ban_hang"}

    def __init__(self):
        self.calls: list[int] = []

    def generate_batch(self, prompts):
        self.calls.append(len(prompts))
        out = []
        for p in prompts:
            retry = "CORRECTION REQUIRED" in p
            if "UNANS" in p:
                out.append({"answer_type": "unanswerable", "reason": "thiếu dữ liệu giá vốn"})
            elif "NEVER" in p or ("BADSQL" in p and not retry):
                out.append(dict(self.BAD))
            else:
                out.append(dict(self.GOOD))
        return out

    def generate(self, prompt):  # sequential adapter around the same behaviour
        return self.generate_batch([prompt])[0]


def _bq(qid, marker):
    return {"id": qid, "business_id": "dev_x", "question": f"tổng {marker}?", "as_of_date": "2026-01-31"}


def test_batch_mode_gives_the_same_rows_as_one_by_one_and_retries_together(cache):
    from code_member_d.inference import answer_batch, answer_question

    questions = [_bq("q1", "OK"), _bq("q2", "BADSQL"), _bq("q3", "UNANS"), _bq("q4", "NEVER")]
    for policy in ("keep", "abstain"):
        engine = BatchEngine()
        batch = answer_batch(questions, cache, engine, DuckDBExecutor(),
                             max_attempts=2, unanswerable_policy=policy)
        one_by_one = [answer_question(q, cache, BatchEngine(), DuckDBExecutor(),
                                      max_attempts=2, unanswerable_policy=policy) for q in questions]
        assert [r for r, _ in batch] == [r for r, _ in one_by_one]
        assert engine.calls == [4, 2]  # round 1: all four; round 2: only the two that failed
    rows = {r["id"]: r for r, _ in answer_batch(questions, cache, BatchEngine(), DuckDBExecutor(),
                                               unanswerable_policy="keep")}
    assert rows["q1"]["answer"] == 2 and rows["q2"]["answer"] == 2
    assert rows["q3"] == {"id": "q3", "type": "unanswerable", "reason": "thiếu dữ liệu giá vốn"}
    assert rows["q4"] == {"id": "q4", "type": "abstain"}
    logs = {l["id"]: l for _, l in answer_batch(questions, cache, BatchEngine(), DuckDBExecutor())}
    assert len(logs["q2"]["attempts"]) == 2 and "error" in logs["q2"]["attempts"][0]


def test_batch_engine_failure_abstains_instead_of_crashing(cache):
    from code_member_d.inference import answer_batch

    class Broken:
        def generate_batch(self, prompts):
            raise RuntimeError("engine died")

    results = answer_batch([_bq("q1", "OK"), _bq("q2", "OK")], cache, Broken(), DuckDBExecutor())
    assert [r for r, _ in results] == [{"id": "q1", "type": "abstain"}, {"id": "q2", "type": "abstain"}]
    assert "engine died" in results[0][1]["attempts"][0]["error"]

    class WrongCount:
        def generate_batch(self, prompts):
            return [{"answer_type": "unanswerable"}]  # one answer for two prompts

    results = answer_batch([_bq("q1", "OK"), _bq("q2", "OK")], cache, WrongCount(), DuckDBExecutor())
    assert all(r["type"] == "abstain" for r, _ in results)


def test_run_uses_batches_writes_every_row_and_resumes(cache, tmp_path):
    out, log = tmp_path / "sub.jsonl", tmp_path / "log.jsonl"
    qs = [_bq(f"q{i}", "OK") for i in range(1, 6)]
    engine = BatchEngine()
    done = run(qs, engine, cache, out, log, batch_size=2)
    assert sorted(done) == [f"q{i}" for i in range(1, 6)]
    assert engine.calls == [2, 2, 1]                      # chunks of 2, 2, 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5
    engine2 = BatchEngine()
    run(qs, engine2, cache, out, log, batch_size=2)       # everything done: nothing to generate
    assert engine2.calls == []
    # batch_size 1 (or an engine without generate_batch) keeps the one-by-one path
    solo = BatchEngine()
    out2 = tmp_path / "sub2.jsonl"
    run(qs[:2], solo, cache, out2, tmp_path / "log2.jsonl", batch_size=1)
    assert solo.calls == [1, 1]


def test_suspicious_results_are_retried_then_abstained_not_submitted(cache):
    from code_member_d.inference import suspicious_reason

    for kind, value in (("number", 0), ("percent", 0.0), ("list", []), ("date", "None"),
                        ("date", None), ("string", ""), ("table", {"columns": ["a"], "rows": []})):
        assert suspicious_reason(kind, value), (kind, value)
    for kind, value in (("number", 2), ("percent", -1.0), ("boolean", False), ("list", ["a"]),
                        ("date", "2026-01-31"), ("string", "Cà phê sữa")):
        assert suspicious_reason(kind, value) is None, (kind, value)

    zero = {"answer_type": "number", "sql": "SELECT COUNT(*) FROM ban_hang WHERE ngay = 'nope'"}
    good = {"answer_type": "number", "sql": "SELECT COUNT(*) FROM ban_hang"}
    # 0 -> retried with the reason in the prompt -> a real answer on the second try
    gen = Scripted(zero, good)
    row, log = _answer(cache, gen, max_attempts=2)
    assert row == {"id": "q1", "type": "number", "answer": 2}
    assert "Suspicious result" in gen.prompts[1] and log["attempts"][0]["rejected_answer"] == 0
    # 0 twice -> abstain, never a submitted 0
    assert _answer(cache, Scripted(zero, zero), max_attempts=2)[0] == {"id": "q1", "type": "abstain"}
    # the old behaviour is still available
    assert _answer(cache, Scripted(zero), max_attempts=1, reject_suspicious=False)[0]["answer"] == 0


def test_git_head_is_read_without_the_git_binary(tmp_path):
    from code_member_d.run_manifest import read_git_head

    assert read_git_head(tmp_path) == {"branch": None, "commit": None}          # no .git at all
    git = tmp_path / ".git"
    (git / "refs" / "heads" / "feat").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/feat/D-infra\n", encoding="utf-8")
    assert read_git_head(tmp_path) == {"branch": "feat/D-infra", "commit": None}  # ref not found
    (git / "packed-refs").write_text("# pack-refs\n" + "a" * 40 + " refs/heads/feat/D-infra\n", encoding="utf-8")
    assert read_git_head(tmp_path)["commit"] == "a" * 40                          # packed ref
    (git / "refs" / "heads" / "feat" / "D-infra").write_text("b" * 40 + "\n", encoding="utf-8")
    assert read_git_head(tmp_path)["commit"] == "b" * 40                          # loose ref wins
    (git / "HEAD").write_text("c" * 40 + "\n", encoding="utf-8")
    assert read_git_head(tmp_path) == {"branch": None, "commit": "c" * 40}        # detached HEAD


def test_manifest_records_each_run_and_warns_when_the_configuration_changes(tmp_path):
    from code_member_d import run_manifest as rm

    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_version": "0.21.2"}), encoding="utf-8")
    q = tmp_path / "q.jsonl"
    q.write_text("{}\n", encoding="utf-8")
    base = {"engine": "hf", "model_id": "Qwen/x", "adapter": str(adapter), "max_input_tokens": 3072,
            "fallback_input_tokens": 2048, "rules_chars": 1600, "max_attempts": 2,
            "unanswerable": "abstain", "allow_suspicious": False}

    record = rm.build_run_record(base, tmp_path, q, tmp_path / "cache", "public_test")
    assert record["adapter_sha256"] == rm.sha256_file(adapter / "adapter_model.safetensors")
    assert record["adapter_peft_version"] == "0.21.2" and len(record["prompt_hash"]) == 12
    assert record["prompt_hash"] == rm.prompt_hash()                      # stable across calls
    assert record["packages"]["duckdb"] and record["python"]
    assert "gpu" in record and "git" in record and record["questions_sha256"]

    manifest = tmp_path / "out.manifest.json"
    assert rm.append_run(manifest, record) == []                           # first run
    assert rm.append_run(manifest, record) == []                           # resumed, same config
    changed = rm.build_run_record({**base, "rules_chars": 3500}, tmp_path, q, tmp_path / "cache", "public_test")
    warnings = rm.append_run(manifest, changed)
    assert len(warnings) == 1 and "rules_chars: 1600 -> 3500" in warnings[0] and "new --out" in warnings[0]
    rm.finish_run(manifest, {"counts": {"abstain": 3}})
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(data["runs"]) == 3 and data["runs"][-1]["counts"] == {"abstain": 3}
    assert "finished_at" in data["runs"][-1] and "finished_at" not in data["runs"][0]
    rm.finish_run(tmp_path / "missing.json", {})                           # never raises


def test_heldout_never_contains_a_training_example_and_keeps_unanswerable(tmp_path):
    from code_member_d.make_heldout import main as heldout_main, trained_ids

    questions, answers, qlora = [], [], []
    def add(qid, biz, kind, in_qlora):
        questions.append({"id": qid, "business_id": biz, "question": "q", "as_of_date": "2026-01-01"})
        answers.append({"id": qid, "type": kind, "answer": None if kind == "unanswerable" else 1})
        if in_qlora:
            target = {"answer_type": kind, "sql": None if kind == "unanswerable" else "SELECT 1 FROM t"}
            qlora.append({"id": qid, "messages": [{"role": "user", "content": "p"},
                                                  {"role": "assistant", "content": json.dumps(target)}]})
    for i in range(40):
        add(f"n{i}", f"b{i % 4}", "number", i < 20)          # 20 of 40 numbers are in the training file
    for i in range(30):
        add(f"u{i}", f"b{i % 4}", "unanswerable", i < 25)    # 25 unanswerable, only 20 kept for training
    files = {}
    for name, rows in (("questions", questions), ("answers", answers), ("qlora_train", qlora)):
        files[name] = tmp_path / f"{name}.jsonl"
        files[name].write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    used = trained_ids(files["qlora_train"])
    assert used <= {r["id"] for r in qlora} and len([i for i in used if i.startswith("u")]) <= 20 - 2

    out = tmp_path / "held"
    argv = ["--questions", str(files["questions"]), "--answers", str(files["answers"]),
            "--qlora-train", str(files["qlora_train"]), "--out-dir", str(out), "--n", "30",
            "--min-unanswerable", "8"]
    assert heldout_main(argv) == 0
    held = [json.loads(l) for l in (out / "questions_heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    gold = [json.loads(l) for l in (out / "answers_heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = {q["id"] for q in held}
    assert ids.isdisjoint(used) and ids == {a["id"] for a in gold} and len(ids) <= 30
    assert sum(a["type"] == "unanswerable" for a in gold) >= 8        # verdicts can be measured
    assert len({q["business_id"] for q in held}) == 4                 # spread over businesses
    first = (out / "questions_heldout.jsonl").read_text(encoding="utf-8")
    heldout_main(argv)
    assert (out / "questions_heldout.jsonl").read_text(encoding="utf-8") == first   # deterministic


def test_filter_submission_abstains_only_the_suspicious_rows(tmp_path):
    from code_member_d.filter_submission import filter_rows

    rows = [
        {"id": "a", "type": "number", "answer": 0},
        {"id": "b", "type": "number", "answer": 1250000},
        {"id": "c", "type": "list", "answer": []},
        {"id": "d", "type": "date", "answer": "None"},
        {"id": "e", "type": "boolean", "answer": False},
        {"id": "f", "type": "unanswerable", "reason": "thiếu dữ liệu"},
        {"id": "g", "type": "abstain"},
    ]
    src, dst = tmp_path / "s.jsonl", tmp_path / "f.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    changed, answered, _ = filter_rows(src, dst)
    out = {r["id"]: r for r in map(json.loads, dst.read_text(encoding="utf-8").splitlines())}
    assert (changed, answered) == (3, 5)
    assert out["b"]["answer"] == 1250000 and out["e"]["answer"] is False
    assert all(out[i] == {"id": i, "type": "abstain"} for i in "acd")
    assert out["f"]["type"] == "unanswerable" and out["g"] == {"id": "g", "type": "abstain"}


def test_check_env_flags_peft_older_than_the_adapter(tmp_path, monkeypatch):
    from code_member_d import check_env

    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"peft_version": "0.21.2"}), encoding="utf-8")
    monkeypatch.setattr(check_env, "PROBLEMS", [])
    monkeypatch.setattr(check_env, "version", lambda name: "0.13.0")
    check_env.check_peft_version(cfg)
    assert check_env.PROBLEMS and "pip install -U peft" in check_env.PROBLEMS[0]

    monkeypatch.setattr(check_env, "PROBLEMS", [])
    monkeypatch.setattr(check_env, "version", lambda name: "0.21.2")
    check_env.check_peft_version(cfg)
    assert not check_env.PROBLEMS
    assert check_env._numbers("0.21.2.dev0") == (0, 21, 2)


def test_redo_abstain_reruns_only_abstained_and_replaces_old_rows(cache, tmp_path):
    out, log = tmp_path / "sub.jsonl", tmp_path / "log.jsonl"
    ok = {"answer_type": "number", "sql": "SELECT COUNT(*) FROM ban_hang"}
    boom = {"answer_type": "number", "sql": "DROP TABLE ban_hang"}
    run([_q("q1"), _q("q2")], Scripted(ok, boom, boom), cache, out, log, max_attempts=2)
    assert read_done(out)["q2"]["type"] == "abstain"

    # plain resume leaves the abstained question alone
    gen = Scripted()
    run([_q("q1"), _q("q2")], gen, cache, out, log)
    assert gen.prompts == []

    # --redo-abstain re-runs q2 only; q1 (a real answer) is untouched
    gen = Scripted(ok)
    done = run([_q("q1"), _q("q2")], gen, cache, out, log, redo_abstain=True)
    assert len(gen.prompts) == 1 and done["q1"]["answer"] == 2 and done["q2"]["answer"] == 2
    assert read_done(out)["q2"] == {"id": "q2", "type": "number", "answer": 2}
    write_final(out, ["q1", "q2"], done)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_resume_after_crash_leaves_truncated_last_line(cache, tmp_path):
    out, log = tmp_path / "sub.jsonl", tmp_path / "log.jsonl"
    good = json.dumps({"id": "q1", "type": "number", "answer": 2})
    out.write_text(good + "\n" + '{"id": "q2", "type": "num', encoding="utf-8")  # cut mid-write
    ok = {"answer_type": "number", "sql": "SELECT COUNT(*) FROM ban_hang"}
    done = run([_q("q1"), _q("q2"), _q("q3")], Scripted(ok, ok), cache, out, log)
    assert sorted(done) == ["q1", "q2", "q3"]  # q1 kept, q2 (cut) and q3 redone
    # every line on disk is valid JSON again, so a second crash cannot lose q3 either
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip().endswith("}")]
    assert {r["id"] for r in rows} == {"q1", "q2", "q3"}
    assert len(read_done(out)) == 3


def test_merge_shards_fills_missing(tmp_path):
    q = tmp_path / "questions.jsonl"
    q.write_text("\n".join(json.dumps({"id": i, "business_id": "b"}) for i in ("a", "b", "c")),
                 encoding="utf-8")
    s1, s2, out = tmp_path / "s1.jsonl", tmp_path / "s2.jsonl", tmp_path / "final.jsonl"
    s1.write_text(json.dumps({"id": "a", "type": "number", "answer": 1}) + "\n", encoding="utf-8")
    s2.write_text(json.dumps({"id": "c", "type": "boolean", "answer": True}) + "\n", encoding="utf-8")
    assert submission_main(["--questions", str(q), "--merge-from", str(s1), str(s2), "--out", str(out)]) == 0
    rows = {r["id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    assert rows["b"] == {"id": "b", "type": "abstain"} and rows["c"]["answer"] is True
    assert submission_main(["--questions", str(q), "--validate", str(out)]) == 0
    assert read_questions(q)[0]["id"] == "a"
