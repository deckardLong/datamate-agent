"""Prompt assembly kept separate so it can be benchmarked and versioned."""

from __future__ import annotations

import json

from .context_adapter import BusinessContext


SYSTEM_PROMPT = """You are DataMate, an offline Vietnamese structured-data agent.
Return one JSON object with: answer_type, sql, confidence, reason.
answer_type must be one of number, percent, string, date, boolean, list, table,
unanswerable, abstain. For answerable questions, generate one read-only DuckDB
SELECT query. Never invent a table or column. Respect the supplied as-of date and
business rules. For ranking ties, sort by code ascending, then name ascending.
Use unanswerable when the necessary data is absent; use abstain when uncertain.
"""


def build_prompt(
    question: str,
    as_of_date: str,
    context: BusinessContext,
    previous_sql: str | None = None,
    execution_error: str | None = None,
) -> str:
    sections = [
        SYSTEM_PROMPT,
        f"Business: {context.business_id}",
        f"As-of date: {as_of_date}",
        "Schema:\n" + context.schema_card,
        "Rules:\n" + json.dumps(context.rules, ensure_ascii=False, default=str),
        "Question:\n" + question,
    ]
    if execution_error:
        sections.append(
            "The previous query failed. Return a corrected plan.\n"
            f"Previous SQL: {previous_sql}\nError: {execution_error}"
        )
    return "\n\n".join(sections)
