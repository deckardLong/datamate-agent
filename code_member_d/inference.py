"""Answer one question: prompt -> SQL -> DuckDB -> formatted submission row.

Uses the same compact prompt member B trained the adapter on
(`build_teacher_prompt(..., include_answer_type=False)`), with the rulebook text
re-rendered from member A's rules.json so it is not cut off at 1,600 chars.
No logic depends on question or business ids.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
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


DEFAULT_REASON = "Dữ liệu của cửa hàng không có đủ thông tin để trả lời câu hỏi này."


def clean_reason(value: Any) -> str:
    """A short natural-language sentence for an unanswerable row.

    The rulebook only requires `reason` to be a string (it is not scored), but model
    output can be empty, JSON or SQL: keep the model's text only when it reads as a
    plain sentence, otherwise use a fixed Vietnamese one.
    """
    if not isinstance(value, str):
        return DEFAULT_REASON
    text = " ".join(value.split())
    looks_technical = text[:1] in "{[" or text.lower().startswith(("select", "with ", "invalid_json"))
    if len(text) < 8 or len(text) > 300 or looks_technical:
        return DEFAULT_REASON
    return text


def assert_finite(value: Any) -> None:
    """NaN/Infinity are not valid JSON; a row containing one could make the whole
    upload unreadable. Treat such a result as an execution failure (retry/abstain)."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Answer contains a non-finite number (NaN/Infinity)")
    if isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)


def suspicious_reason(answer_type: str, value: Any) -> str | None:
    """Why a formatted answer is almost certainly wrong, or None.

    Basis: in the 3,000 reference answers of the training split only 0.16% of numbers and
    percents are 0, and none of the 193 lists is empty. A query that returns 0, an empty
    list or an empty date nearly always has a wrong period/status/join, so it is retried
    once with this message and, if it still fails, abstained (0 points instead of -1).
    """
    if answer_type in {"number", "percent"} and isinstance(value, (int, float)) \
            and not isinstance(value, bool) and value == 0:
        return "the result is 0, which is almost never the true answer; recheck the table, " \
               "the date range and the status filters"
    if answer_type in {"date", "string"} and (
        value is None or str(value).strip().lower() in {"", "none", "nan", "nat", "null"}
    ):
        return f"the {answer_type} result is empty"
    if answer_type == "list" and (not isinstance(value, list) or len(value) == 0):
        return "the list result is empty; recheck the filters"
    if answer_type == "table" and (
        not isinstance(value, dict) or not value.get("rows")
    ):
        return "the table result has no rows; recheck the filters"
    return None


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


@dataclass
class _Work:
    """State of one question while it is being answered (shared by both modes)."""

    question: dict[str, Any]
    qid: str
    log: dict[str, Any]
    started: float
    context: BusinessContext | None = None
    record: TrainingRecord | None = None
    base_prompt: str = ""
    prompt: str = ""
    final: dict[str, Any] | None = None


def _prepare(question: dict[str, Any], contexts: ContextCache, rules_chars: int) -> _Work:
    qid = str(question["id"])
    work = _Work(
        question=question, qid=qid, started=time.perf_counter(),
        log={"id": qid, "business_id": question.get("business_id"),
             "question": question.get("question"), "attempts": []},
    )
    try:
        work.context = contexts.get(str(question["business_id"]))
        work.record = TrainingRecord(
            id=qid, business_id=str(question["business_id"]),
            question=str(question["question"]), as_of_date=str(question["as_of_date"]),
            answer_type="unknown", expected_answer=None, aliases=[],
        )
        work.base_prompt = work.prompt = build_prompt(work.record, work.context, rules_chars)
        work.log["prompt_chars"] = len(work.base_prompt)
    except Exception as exc:  # noqa: BLE001
        work.log["error"] = f"setup: {type(exc).__name__}: {exc}"
        work.final = _abstain(qid)
    return work


def _apply(work: _Work, attempt_no: int, produced: Any, executor: DuckDBExecutor,
           policy: str, reject_suspicious: bool = True) -> None:
    """Evaluate one model output (a plan dict, or the exception raised while generating).

    Sets work.final on success or on a final verdict; otherwise records the error and
    prepares the retry prompt.
    """
    attempt: dict[str, Any] = {"n": attempt_no}
    work.log["attempts"].append(attempt)
    candidate: dict[str, Any] = {}
    try:
        if isinstance(produced, BaseException):
            raise produced
        candidate = produced
        attempt["plan"] = candidate
        kind = str(candidate.get("answer_type", candidate.get("type", ""))).lower()
        kind = _TYPE_ALIASES.get(kind, kind)
        if kind == "unanswerable":
            if policy == "abstain":
                attempt["note"] = "model said unanswerable; policy=abstain"
                work.final = _abstain(work.qid)
            else:
                work.final = {"id": work.qid, "type": "unanswerable",
                              "reason": clean_reason(candidate.get("reason"))}
            return
        candidate = {**candidate, "answer_type": kind}
        plan = QueryPlan.from_mapping(candidate)  # validates type and SQL presence
        if plan.answer_type == "abstain":
            raise ValueError("model abstained")
        sql = plan.sql or ""
        reject_constant_only_sql(sql)
        assert work.context is not None
        result = executor.execute(work.context.db_path, sql)
        if plan.answer_type == "list" and len(result.columns) > 1:
            sql = _list_projection_sql(sql, result.columns, result.rows)
            result = executor.execute(work.context.db_path, sql)
        value = format_query_result(result, plan.answer_type)
        assert_finite(value)
        why = suspicious_reason(plan.answer_type, value) if reject_suspicious else None
        if why:
            attempt["rejected_answer"] = value
            raise ValueError(f"Suspicious result: {why}")
        attempt.update(sql=sql, answer=value)
        work.final = {"id": work.qid, "type": plan.answer_type, "answer": value}
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        attempt["error"] = error
        work.prompt = _retry_teacher_prompt(
            work.base_prompt, candidate, work.context, work.record, error=error
        )


def _finish(work: _Work, elapsed: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if work.final is None:
        work.final = _abstain(work.qid)
    work.log["final"] = work.final
    work.log["elapsed_seconds"] = round(
        elapsed if elapsed is not None else time.perf_counter() - work.started, 2
    )
    return work.final, work.log


def answer_question(
    question: dict[str, Any],
    contexts: ContextCache,
    generator: Generator,
    executor: DuckDBExecutor,
    *,
    max_attempts: int = 2,
    rules_chars: int = 1600,
    unanswerable_policy: str = "keep",
    reject_suspicious: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (submission row, log record). Never raises.

    unanswerable_policy: "keep" submits the model's unanswerable verdict;
    "abstain" scores 0 instead of risking -1 until the verdicts are validated.
    """
    work = _prepare(question, contexts, rules_chars)
    for attempt_no in range(1, max_attempts + 1):
        if work.final is not None:
            break
        try:
            produced: Any = generator.generate(work.prompt)
        except Exception as exc:  # noqa: BLE001
            produced = exc
        _apply(work, attempt_no, produced, executor, unanswerable_policy, reject_suspicious)
    return _finish(work)


def answer_batch(
    questions: list[dict[str, Any]],
    contexts: ContextCache,
    generator: Any,
    executor: DuckDBExecutor,
    *,
    max_attempts: int = 2,
    rules_chars: int = 1600,
    unanswerable_policy: str = "keep",
    reject_suspicious: bool = True,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Same result as answer_question for each question, but every attempt round is one
    generator.generate_batch(prompts) call, so a batching engine (vLLM) can run the whole
    chunk concurrently. Questions that failed are retried together in the next round."""
    started = time.perf_counter()
    works = [_prepare(q, contexts, rules_chars) for q in questions]
    for attempt_no in range(1, max_attempts + 1):
        pending = [w for w in works if w.final is None]
        if not pending:
            break
        try:
            outputs: list[Any] = list(generator.generate_batch([w.prompt for w in pending]))
            if len(outputs) != len(pending):
                raise RuntimeError(f"generate_batch returned {len(outputs)} for {len(pending)} prompts")
        except Exception as exc:  # noqa: BLE001
            outputs = [exc] * len(pending)
        for work, produced in zip(pending, outputs):
            _apply(work, attempt_no, produced, executor, unanswerable_policy, reject_suspicious)
    share = (time.perf_counter() - started) / max(1, len(works))
    return [_finish(w, share) for w in works]
