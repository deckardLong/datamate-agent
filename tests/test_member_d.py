from __future__ import annotations

import json

import duckdb
import pytest

from code_member_b.executor import DuckDBExecutor
from code_member_d.inference import ContextCache, answer_question, build_prompt, render_rules
from code_member_d.make_baseline_submission import main as submission_main
from code_member_d.run_all import read_questions, run, write_final


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
