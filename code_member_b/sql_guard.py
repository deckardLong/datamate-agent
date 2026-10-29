"""Conservative SQL validation for model-generated DuckDB queries."""

from __future__ import annotations

import re


class UnsafeSQLError(ValueError):
    pass


_FORBIDDEN = re.compile(
    r"\b(attach|call|copy|create|delete|detach|drop|export|import|insert|install|"
    r"load|pragma|replace|truncate|update|vacuum)\b",
    re.IGNORECASE,
)
_FILE_READERS = re.compile(
    r"\b(read_csv|read_csv_auto|read_json|read_ndjson|read_parquet|sqlite_scan|"
    r"postgres_scan|httpfs)\s*\(",
    re.IGNORECASE,
)


def validate_read_only_sql(sql: str) -> str:
    query = sql.strip()
    if not query:
        raise UnsafeSQLError("SQL is empty")

    # Permit one optional trailing semicolon, never multiple statements.
    without_trailing = query[:-1].rstrip() if query.endswith(";") else query
    if ";" in without_trailing:
        raise UnsafeSQLError("Multiple SQL statements are not allowed")
    if not re.match(r"^(select|with)\b", without_trailing, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT or WITH ... SELECT is allowed")
    if _FORBIDDEN.search(without_trailing):
        raise UnsafeSQLError("SQL contains a forbidden operation")
    if _FILE_READERS.search(without_trailing):
        raise UnsafeSQLError("SQL cannot read external files or databases")
    return without_trailing
