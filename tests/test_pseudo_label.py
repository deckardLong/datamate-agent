from __future__ import annotations

import json

from code_member_b.pseudo_label import (
    PseudoLabelPipeline,
    TrainingRecord,
    answers_match,
    export_qlora_jsonl,
    reject_constant_only_sql,
    stratified_sample,
)


class FakeGenerator:
    model_id = "fake-teacher"

    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    def generate(self, prompt, candidate_count):
        assert "expected_answer" not in prompt
        self.calls += 1
        return self.candidates[:candidate_count]


def _record(record_id="q1", answer_type="number", expected=30):
    return TrainingRecord(
        id=record_id,
        business_id="biz_001",
        question="Tổng doanh thu?",
        as_of_date="2026-10-30",
        answer_type=answer_type,
        expected_answer=expected,
        aliases=[],
    )


def _write_context(root):
    import duckdb

    business = root / "train" / "biz_001"
    business.mkdir(parents=True)
    connection = duckdb.connect(str(business / "db.duckdb"))
    connection.execute("CREATE TABLE sales(amount INTEGER)")
    connection.execute("INSERT INTO sales VALUES (10), (20)")
    connection.close()
    (business / "profile.json").write_text(
        json.dumps(
            {
                "business_id": "biz_001",
                "tables": {
                    "sales": {
                        "row_count": 2,
                        "columns": {"amount": {"type": "INTEGER"}},
                        "samples": [{"amount": 10}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_answer_matching_supports_numeric_tolerance_alias_and_collections():
    assert answers_match(10.001, 10, "number")
    assert answers_match(" Hà Nội ", "hà nội", "string")
    assert answers_match("HCM", "Sài Gòn", "string", aliases=["hcm"])
    assert answers_match(["A", 2], ["a", 2.0], "list")
    assert answers_match(
        ["SP0003", "SP0017"],
        ["Cà phê sữa", "Trà đào"],
        "list",
        aliases=[["SP0003"], ["SP0017"]],
    )
    assert answers_match(
        {"columns": ["name"], "rows": [["A"]]},
        {"columns": ["name"], "rows": [["a"]]},
        "table",
    )
    assert answers_match(
        {
            "columns": ["ma_khach_hang", "so_tien_no"],
            "rows": [["KH0124", 404500]],
        },
        {
            "columns": ["khach_hang", "so_tien_no"],
            "rows": [["Đỗ Đức An", 404500]],
        },
        "table",
        aliases=[["KH0124"]],
    )


def test_constant_only_sql_is_rejected():
    try:
        reject_constant_only_sql("SELECT 30")
    except ValueError as exc:
        assert "does not read" in str(exc)
    else:
        raise AssertionError("Expected constant-only SQL to be rejected")
    reject_constant_only_sql("SELECT SUM(amount) FROM sales")


def test_stratified_sample_is_deterministic_and_spread():
    records = [
        _record(f"q{i}", "number" if i % 2 else "string", i)
        for i in range(10)
    ]
    first = stratified_sample(records, 4, seed=7)
    second = stratified_sample(records, 4, seed=7)
    assert [row.id for row in first] == [row.id for row in second]
    assert {row.answer_type for row in first} == {"number", "string"}


def test_pipeline_verifies_shortest_matching_sql_and_resumes(tmp_path):
    _write_context(tmp_path)
    generator = FakeGenerator(
        [
            {
                "answer_type": "number",
                "sql": "SELECT SUM(amount) + 1 FROM sales",
                "confidence": 0.5,
            },
            {
                "answer_type": "number",
                "sql": "SELECT SUM(amount) FROM sales",
                "confidence": 0.9,
            },
        ]
    )
    pipeline = PseudoLabelPipeline(
        generator,
        data_root=tmp_path,
        output_dir=tmp_path / "out",
        candidate_count=2,
    )
    first = pipeline.run([_record()])
    assert first["accepted_this_run"] == 1
    accepted = json.loads(pipeline.accepted_path.read_text(encoding="utf-8"))
    assert accepted["sql"] == "SELECT SUM(amount) FROM sales"
    assert accepted["observed_answer"] == 30
    assert "expected_answer" not in accepted["prompt"]
    assert "gold answer" not in accepted["prompt"].casefold()
    assert "Required answer_type" not in accepted["prompt"]

    second = pipeline.run([_record()])
    assert second["processed_this_run"] == 0
    assert second["skipped_existing"] == 1
    assert generator.calls == 1


def test_unanswerable_needs_no_teacher_and_qlora_excludes_gold(tmp_path):
    _write_context(tmp_path)
    generator = FakeGenerator([])
    pipeline = PseudoLabelPipeline(
        generator,
        data_root=tmp_path,
        output_dir=tmp_path / "out",
    )
    record = _record("q-un", "unanswerable", expected=None)
    report = pipeline.run([record])
    assert report["accepted_this_run"] == 1
    assert generator.calls == 0

    qlora_path = tmp_path / "qlora.jsonl"
    assert export_qlora_jsonl(pipeline.accepted_path, qlora_path) == 1
    exported = json.loads(qlora_path.read_text(encoding="utf-8"))
    assert "expected_answer" not in json.dumps(exported)
    completion = json.loads(exported["messages"][1]["content"])
    assert completion["answer_type"] == "unanswerable"
    assert completion["sql"] is None
