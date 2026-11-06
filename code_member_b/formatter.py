"""Convert query results to the exact competition answer shapes."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from .contracts import AnswerType
from .executor import QueryResult


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


def _single_value(result: QueryResult) -> Any:
    if len(result.rows) != 1 or len(result.rows[0]) != 1:
        raise ValueError("Scalar answer requires exactly one row and one column")
    return _plain(result.rows[0][0])


def format_query_result(result: QueryResult, answer_type: AnswerType) -> Any:
    if answer_type in {"number", "percent"}:
        value = _single_value(result)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected a numeric value, got {value!r}")
        rounded = round(float(value), 2)
        if answer_type == "number" and rounded.is_integer():
            return int(rounded)
        return rounded

    if answer_type == "string":
        value = _single_value(result)
        if value is None:
            raise ValueError("String answer cannot be null")
        return str(value)

    if answer_type == "date":
        value = _single_value(result)
        if isinstance(value, (date, datetime)):
            return _plain(value)
        return str(value)

    if answer_type == "boolean":
        value = _single_value(result)
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"Expected a boolean value, got {value!r}")

    if answer_type == "list":
        if len(result.columns) != 1:
            raise ValueError("List answer requires exactly one column")
        return [_plain(row[0]) for row in result.rows]

    if answer_type == "table":
        return {
            "columns": result.columns,
            "rows": [[_plain(value) for value in row] for row in result.rows],
        }

    raise ValueError(f"No query-result formatter for answer type {answer_type!r}")
