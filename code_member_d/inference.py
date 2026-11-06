"""Answer one question: prompt -> SQL -> DuckDB -> formatted submission row.

Uses the same compact prompt member B trained the adapter on
(`build_teacher_prompt(..., include_answer_type=False)`), with the rulebook text
re-rendered from member A's rules.json so it is not cut off at 1,600 chars.
No logic depends on question or business ids.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Protocol

from code_member_a.rules import compact_rules
from code_member_b.context_adapter import BusinessContext, load_business_context
from code_member_b.contracts import QueryPlan
from code_member_b.executor import DuckDBExecutor
from code_member_b.formatter import format_query_result
from code_member_b.pseudo_label import (
    TrainingRecord,
    _list_projection_sql,
    _retry_teacher_prompt,
    build_teacher_prompt,
    reject_constant_only_sql,
)

# Topics first that change the numbers most; the rest fills what budget is left.
_TOPIC_ORDER = (
    "revenue", "cancel_return", "valid_order", "as_of_date", "week_definition",
    "this_month", "same_period", "best_selling", "ranking_tiebreak", "rounding",
    "tet_window", "missing_vs_zero", "unanswerable",
)
_TYPE_ALIASES = {
    "percentage": "percent", "numeric": "number", "integer": "number",
    "float": "number", "text": "string", "bool": "boolean",
}
_RULES_HEADER = "Business rules (possibly abbreviated):\n"


class Generator(Protocol):
    model_id: str

    def generate(self, prompt: str) -> dict[str, Any]: ...


def render_rules(rules: dict[str, Any], max_chars: int) -> str:
    """Round-robin over rulebook topics until the character budget is used."""
    raw = compact_rules(rules, max_per_topic=12, max_chars=260)
    # Table header rows ("Luật Mặc định", "Cách nói Nghĩa") are not rules: drop them
    # first so each topic's real first rule is at the front of its list.
    compact = {
        topic: [t for t in (x.replace("**", "").strip() for x in texts) if len(t) >= 30]
        for topic, texts in raw.items()
    }
    topics = [t for t in _TOPIC_ORDER if compact.get(t)] + [
        t for t in compact if t not in _TOPIC_ORDER and compact[t]
    ]
    lines: list[str] = []
    seen: set[str] = set()  # one rulebook row can be tagged under several topics
    used = 0
    depth = max((len(v) for v in compact.values()), default=0)
    for level in range(depth):
        for topic in topics:
            if level >= len(compact[topic]):
                continue
            text = compact[topic][level]
            if text in seen:
                continue
            seen.add(text)
            line = f"- {text}"
            if used + len(line) + 1 > max_chars:
                return "\n".join(lines)
            lines.append(line)
            used += len(line) + 1
    return "\n".join(lines)


def build_prompt(record: TrainingRecord, context: BusinessContext, rules_chars: int) -> str:
    # Build with empty rules, then swap in the larger rendering of the real ones.
    prompt = build_teacher_prompt(
        record, dataclasses.replace(context, rules={}), include_answer_type=False
    )
    marker = _RULES_HEADER + "{}"
    text = render_rules(context.rules, rules_chars) if context.rules else ""
    if marker in prompt:
        return prompt.replace(marker, _RULES_HEADER + (text or "{}"), 1)
    return prompt


class ContextCache:
    def __init__(self, cache_root: str | Path, split: str):
        self.cache_root, self.split = Path(cache_root), split
        self._items: dict[str, BusinessContext] = {}

    def get(self, business_id: str) -> BusinessContext:
        if business_id not in self._items:
            self._items[business_id] = load_business_context(
                self.split, business_id, self.cache_root
            )
        return self._items[business_id]


def _abstain(qid: str) -> dict[str, Any]:
    return {"id": qid, "type": "abstain"}


def answer_question(
    question: dict[str, Any],
    contexts: ContextCache,
    generator: Generator,
    executor: DuckDBExecutor,
    *,
    max_attempts: int = 2,
    rules_chars: int = 3500,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (submission row, log record). Never raises."""
    qid = str(question["id"])
    started = time.perf_counter()
    log: dict[str, Any] = {"id": qid, "business_id": question.get("business_id"),
                           "question": question.get("question"), "attempts": []}

    def finish(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        log["final"] = row
        log["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        return row, log

    try:
        context = contexts.get(str(question["business_id"]))
        record = TrainingRecord(
            id=qid, business_id=str(question["business_id"]),
            question=str(question["question"]), as_of_date=str(question["as_of_date"]),
            answer_type="unknown", expected_answer=None, aliases=[],
        )
        base_prompt = build_prompt(record, context, rules_chars)
    except Exception as exc:  # noqa: BLE001
        log["error"] = f"setup: {type(exc).__name__}: {exc}"
        return finish(_abstain(qid))

    log["prompt_chars"] = len(base_prompt)
    prompt = base_prompt
    for attempt_no in range(1, max_attempts + 1):
        attempt: dict[str, Any] = {"n": attempt_no}
        log["attempts"].append(attempt)
        candidate: dict[str, Any] = {}
        try:
            candidate = generator.generate(prompt)
            attempt["plan"] = candidate
            kind = str(candidate.get("answer_type", candidate.get("type", ""))).lower()
            kind = _TYPE_ALIASES.get(kind, kind)
            if kind == "unanswerable":
                reason = str(candidate.get("reason") or "Dữ liệu không đủ để trả lời")
                return finish({"id": qid, "type": "unanswerable", "reason": reason})
            candidate = {**candidate, "answer_type": kind}
            plan = QueryPlan.from_mapping(candidate)  # validates type and SQL presence
            if plan.answer_type == "abstain":
                raise ValueError("model abstained")
            sql = plan.sql or ""
            reject_constant_only_sql(sql)
            result = executor.execute(context.db_path, sql)
            if plan.answer_type == "list" and len(result.columns) > 1:
                sql = _list_projection_sql(sql, result.columns, result.rows)
                result = executor.execute(context.db_path, sql)
            value = format_query_result(result, plan.answer_type)
            attempt.update(sql=sql, answer=value)
            return finish({"id": qid, "type": plan.answer_type, "answer": value})
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            attempt["error"] = error
            prompt = _retry_teacher_prompt(base_prompt, candidate, context, record, error=error)

    return finish(_abstain(qid))
