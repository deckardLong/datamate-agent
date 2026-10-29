"""Small dependency-free contracts shared inside member B's agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AnswerType = Literal[
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
]

ANSWER_TYPES = {
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
}


@dataclass(frozen=True)
class QueryPlan:
    """Structured output expected from the language model."""

    answer_type: AnswerType
    sql: str | None = None
    confidence: float = 0.0
    reason: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "QueryPlan":
        answer_type = str(value.get("answer_type", value.get("type", ""))).lower()
        if answer_type not in ANSWER_TYPES:
            raise ValueError(f"Unsupported answer type: {answer_type!r}")

        confidence = float(value.get("confidence", 0.0))
        confidence = min(1.0, max(0.0, confidence))
        sql = value.get("sql")
        if answer_type not in {"unanswerable", "abstain"} and not sql:
            raise ValueError(f"A SQL query is required for answer type {answer_type!r}")

        return cls(
            answer_type=answer_type,  # type: ignore[arg-type]
            sql=str(sql).strip() if sql else None,
            confidence=confidence,
            reason=str(value["reason"]) if value.get("reason") else None,
        )


@dataclass
class AgentAnswer:
    type: AnswerType
    answer: Any = None
    confidence: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.type in {"unanswerable", "abstain"}:
            result.pop("answer", None)
        if not self.reason:
            result.pop("reason", None)
        return result
