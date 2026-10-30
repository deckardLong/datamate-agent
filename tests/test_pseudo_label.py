from __future__ import annotations

import json

from code_member_b.pseudo_label import (
    PseudoLabelPipeline,
    TrainingRecord,
    answers_match,
    build_teacher_prompt,
    export_qlora_jsonl,
    reject_constant_only_sql,
    stratified_sample,
)
from code_member_b.context_adapter import load_business_context


class FakeGenerator:
    model_id = "fake-teacher"

    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0
        self.prompts = []

    def generate(self, prompt, candidate_count):
        assert "expected_answer" not in prompt
        assert candidate_count == 1
        self.prompts.append(prompt)
        candidate = self.candidates[self.calls]
        self.calls += 1
        return [candidate]


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
    connection.execute(
        "CREATE TABLE sales(amount INTEGER, customer_id VARCHAR, customer_name VARCHAR)"
    )
    connection.execute("INSERT INTO sales VALUES (10, 'C1', 'An'), (20, 'C2', 'Binh')")
    connection.close()
    (business / "profile.json").write_text(
        json.dumps(
            {
                "business_id": "biz_001",
                "tables": {
                    "sales": {
                        "row_count": 2,
                        "columns": {
                            "amount": {"type": "INTEGER"},
                            "customer_id": {"type": "VARCHAR"},
                            "customer_name": {"type": "VARCHAR"},
                        },
                        "samples": [
                            {"amount": 10, "customer_id": "C1", "customer_name": "An"}
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_header_order_context(root):
    import duckdb

    business = root / "train" / "biz_001"
    business.mkdir(parents=True)
    connection = duckdb.connect(str(business / "db.duckdb"))
    connection.execute(
        "CREATE TABLE orders_06_2025("
        "trang_thai_don_hang VARCHAR, dat_hang_da_duyet VARCHAR, "
        "unnamed_2 VARCHAR, unnamed_7 VARCHAR)"
    )
    connection.execute(
        "INSERT INTO orders_06_2025 VALUES "
        "('ORD1', '01/06/2025', 'Đã hoàn thành', '100,000'), "
        "('ORD2', '02/06/2025', 'Đã hủy', '50,000')"
    )
    connection.close()
    (business / "profile.json").write_text(
        json.dumps(
            {
                "business_id": "biz_001",
                "tables": {
                    "orders_06_2025": {
                        "row_count": 2,
                        "columns": {
                            "trang_thai_don_hang": {"type": "VARCHAR"},
                            "dat_hang_da_duyet": {"type": "VARCHAR"},
                            "unnamed_2": {"type": "VARCHAR"},
                            "unnamed_7": {"type": "VARCHAR"},
                        },
                        "samples": [
                            {
                                "trang_thai_don_hang": "Mã đơn hàng",
                                "dat_hang_da_duyet": "Ngày đặt hàng",
                                "unnamed_2": "Trạng thái đơn hàng",
                                "unnamed_7": "Tổng tiền",
                            }
                        ],
                    }
                },
            },
            ensure_ascii=False,
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


def test_small_stratified_sample_does_not_repeat_type_before_cycle():
    records = []
    for answer_type in ("number", "percent", "string"):
        for index in range(4):
            records.append(_record(f"{answer_type}-{index}", answer_type, index))
    selected = stratified_sample(records, 3, seed=11)
    assert len({row.answer_type for row in selected}) == 3


def test_compact_teacher_prompt_keeps_exact_schema_question_and_hides_gold(tmp_path):
    _write_context(tmp_path)
    record = _record(expected=999999)
    context = load_business_context("train", "biz_001", tmp_path)
    prompt = build_teacher_prompt(record, context)
    assert 'TABLE "sales"' in prompt
    assert '"amount" INTEGER' in prompt
    assert record.question in prompt
    assert str(record.expected_answer) not in prompt


def test_teacher_prompt_prioritizes_question_month_tables(tmp_path):
    _write_context(tmp_path)
    profile_path = tmp_path / "train" / "biz_001" / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["tables"] = {
        **{
            f"orders_{month:02d}_2025": {
                "row_count": 1,
                "columns": {"amount": {"type": "INTEGER"}},
                "samples": [{"amount": month}],
            }
            for month in range(1, 9)
        },
        "orders_09_2025": {
            "row_count": 1,
            "columns": {"amount": {"type": "INTEGER"}},
            "samples": [{"amount": 9}],
        },
        "orders_10_2025": {
            "row_count": 1,
            "columns": {"amount": {"type": "INTEGER"}},
            "samples": [{"amount": 10}],
        },
        "orders_11_2025": {
            "row_count": 1,
            "columns": {"amount": {"type": "INTEGER"}},
            "samples": [{"amount": 11}],
        },
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    context = load_business_context("train", "biz_001", tmp_path)
    record = TrainingRecord(
        id="q-months",
        business_id="biz_001",
        question="Doanh thu thang 9/2025, 10/2025 va 11/2025 co tang lien tuc?",
        as_of_date="2026-01-01",
        answer_type="boolean",
        expected_answer=False,
        aliases=[],
    )
    prompt = build_teacher_prompt(record, context)
    schema = prompt.split("Exact schema for likely relevant tables:\n", 1)[1]
    assert schema.index('TABLE "orders_09_2025"') < schema.index(
        'TABLE "orders_01_2025"'
    )


def test_pipeline_repairs_failed_verification_and_resumes(tmp_path):
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
    assert generator.calls == 2
    assert "failed private answer verification" in generator.prompts[1]
    assert "SELECT SUM(amount) + 1 FROM sales" in generator.prompts[1]

    second = pipeline.run([_record()])
    assert second["processed_this_run"] == 0
    assert second["skipped_existing"] == 1
    assert generator.calls == 2


def test_pipeline_feeds_duckdb_error_to_next_attempt(tmp_path):
    _write_context(tmp_path)
    generator = FakeGenerator(
        [
            {
                "answer_type": "number",
                "sql": "SELECT SUM(missing) FROM sales",
                "confidence": 0.3,
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
    report = pipeline.run([_record()])
    assert report["accepted_this_run"] == 1
    assert "BinderException" in generator.prompts[1]
    assert "missing" in generator.prompts[1]


def test_pipeline_projects_ranked_list_to_display_column(tmp_path):
    _write_context(tmp_path)
    generator = FakeGenerator(
        [
            {
                "answer_type": "list",
                "sql": (
                    "SELECT customer_id, customer_name, SUM(amount) AS total "
                    "FROM sales GROUP BY customer_id, customer_name "
                    "ORDER BY total DESC, customer_id ASC"
                ),
                "confidence": 0.9,
            }
        ]
    )
    pipeline = PseudoLabelPipeline(
        generator,
        data_root=tmp_path,
        output_dir=tmp_path / "out",
        candidate_count=1,
    )
    record = _record("q-list", "list", expected=["Binh", "An"])
    report = pipeline.run([record])
    assert report["accepted_this_run"] == 1
    accepted = json.loads(pipeline.accepted_path.read_text(encoding="utf-8"))
    assert accepted["observed_answer"] == ["Binh", "An"]
    assert accepted["sql"].startswith('SELECT "customer_name" FROM (')


def test_schema_template_uses_header_mapping_for_cancellation_rate(tmp_path):
    _write_header_order_context(tmp_path)
    generator = FakeGenerator([])
    pipeline = PseudoLabelPipeline(
        generator,
        data_root=tmp_path,
        output_dir=tmp_path / "out",
        candidate_count=1,
    )
    record = TrainingRecord(
        id="q-cancel-rate",
        business_id="biz_001",
        question="Thang 6/2025 ti le don bi huy la bao nhieu %?",
        as_of_date="2025-06-30",
        answer_type="percent",
        expected_answer=50.0,
        aliases=[],
    )
    report = pipeline.run([record])
    assert report["accepted_this_run"] == 1
    assert generator.calls == 0
    accepted = json.loads(pipeline.accepted_path.read_text(encoding="utf-8"))
    assert accepted["teacher_model"] == "schema-template"
    assert accepted["observed_answer"] == 50.0


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
