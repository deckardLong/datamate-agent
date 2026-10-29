from __future__ import annotations

import json

import pytest

from code_member_b.agent import AgentCore
from code_member_b.context_adapter import load_business_context
from code_member_b.contracts import QueryPlan
from code_member_b.executor import QueryResult
from code_member_b.formatter import format_query_result
from code_member_b.sql_guard import UnsafeSQLError, validate_read_only_sql
from code_member_b.llm_client import StaticPlanner


def test_query_plan_requires_sql_for_answerable_type():
    with pytest.raises(ValueError):
        QueryPlan.from_mapping({"answer_type": "number", "confidence": 0.9})


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM sales",
        "SELECT 1; DROP TABLE sales",
        "SELECT * FROM read_csv_auto('secret.csv')",
        "INSTALL sqlite",
    ],
)
def test_sql_guard_rejects_unsafe_queries(sql):
    with pytest.raises(UnsafeSQLError):
        validate_read_only_sql(sql)


def test_sql_guard_accepts_select_and_cte():
    assert validate_read_only_sql("SELECT COUNT(*) FROM sales;") == "SELECT COUNT(*) FROM sales"
    assert validate_read_only_sql("WITH x AS (SELECT 1 AS n) SELECT n FROM x").startswith("WITH")


def test_format_scalar_and_collection_answers():
    scalar = QueryResult(columns=["total"], rows=[(10.0,)])
    assert format_query_result(scalar, "number") == 10

    items = QueryResult(columns=["name"], rows=[("A",), ("B",)])
    assert format_query_result(items, "list") == ["A", "B"]

    table = QueryResult(columns=["name", "total"], rows=[("A", 10)])
    assert format_query_result(table, "table") == {
        "columns": ["name", "total"],
        "rows": [["A", 10]],
    }


def test_context_adapter_supports_member_a_database_name(tmp_path):
    business = tmp_path / "train" / "biz_001"
    business.mkdir(parents=True)
    (business / "biz_001.duckdb").touch()
    (business / "profile.json").write_text(
        json.dumps(
            {
                "business_id": "biz_001",
                "tables": {
                    "sales": {
                        "row_count": 2,
                        "columns": {"amount": {"type": "BIGINT"}},
                        "samples": [{"amount": 10}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    context = load_business_context("train", "biz_001", tmp_path)
    assert context.db_path.name == "biz_001.duckdb"
    assert "sales" in context.schema_card


def test_agent_retries_an_unsafe_query_then_answers(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    business = tmp_path / "train" / "biz_001"
    business.mkdir(parents=True)
    db_path = business / "biz_001.duckdb"
    connection = duckdb.connect(str(db_path))
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
    planner = StaticPlanner(
        [
            {"answer_type": "number", "sql": "DELETE FROM sales", "confidence": 0.2},
            {
                "answer_type": "number",
                "sql": "SELECT SUM(amount) FROM sales",
                "confidence": 0.9,
            },
        ]
    )

    result = AgentCore(planner, data_root=tmp_path).answer(
        "Tổng doanh thu?", "biz_001", "2026-10-30"
    )

    assert result["type"] == "number"
    assert result["answer"] == 30
    assert result["trace"]["attempts"] == 2
