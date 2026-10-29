"""Read-only DuckDB execution with bounded result size."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Timer
from typing import Any

from .sql_guard import validate_read_only_sql


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]


class DuckDBExecutor:
    def __init__(self, max_rows: int = 1000, timeout_seconds: float = 30.0):
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

    def execute(self, db_path: str | Path, sql: str) -> QueryResult:
        query = validate_read_only_sql(sql)
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "duckdb is not installed; run: pip install -r code_member_a/requirements.txt"
            ) from exc

        connection = duckdb.connect(str(db_path), read_only=True)
        timer = Timer(self.timeout_seconds, connection.interrupt)
        timer.daemon = True
        try:
            timer.start()
            cursor = connection.execute(query)
            columns = [item[0] for item in (cursor.description or [])]
            rows = cursor.fetchmany(self.max_rows + 1)
            if len(rows) > self.max_rows:
                raise ValueError(
                    f"Query returned more than {self.max_rows} rows; aggregate or limit it"
                )
            return QueryResult(columns=columns, rows=rows)
        finally:
            timer.cancel()
            connection.close()
