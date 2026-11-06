"""Question-to-SQL orchestration for member B's Agent Core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .context_adapter import load_business_context
from .contracts import AgentAnswer, QueryPlan
from .executor import DuckDBExecutor
from .formatter import format_query_result
from .llm_client import Planner
from .prompt_builder import build_prompt


class AgentCore:
    def __init__(
        self,
        planner: Planner,
        data_root: str | Path = "data/datamate-corpus",
        max_attempts: int = 3,
        max_rows: int = 1000,
        query_timeout_seconds: float = 30.0,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.planner = planner
        self.data_root = Path(data_root)
        self.max_attempts = max_attempts
        self.executor = DuckDBExecutor(
            max_rows=max_rows,
            timeout_seconds=query_timeout_seconds,
        )

    def answer(
        self,
        question: str,
        business_id: str,
        as_of_date: str,
        split: str = "train",
    ) -> dict[str, Any]:
        context = load_business_context(split, business_id, self.data_root)
        previous_sql: str | None = None
        error: str | None = None

        for attempt in range(1, self.max_attempts + 1):
            prompt = build_prompt(question, as_of_date, context, previous_sql, error)
            try:
                plan = QueryPlan.from_mapping(self.planner.plan(prompt))
            except Exception as exc:
                error = f"Invalid planner output: {exc}"
                continue

            if plan.answer_type in {"unanswerable", "abstain"}:
                return AgentAnswer(
                    type=plan.answer_type,
                    confidence=plan.confidence,
                    reason=plan.reason,
                    trace={"attempts": attempt},
                ).to_dict()

            previous_sql = plan.sql
            try:
                result = self.executor.execute(context.db_path, plan.sql or "")
                value = format_query_result(result, plan.answer_type)
                return AgentAnswer(
                    type=plan.answer_type,
                    answer=value,
                    confidence=plan.confidence,
                    trace={"sql": plan.sql, "attempts": attempt},
                ).to_dict()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        return AgentAnswer(
            type="abstain",
            confidence=0.0,
            reason=error or "Agent could not produce a valid answer",
            trace={"sql": previous_sql, "attempts": self.max_attempts},
        ).to_dict()


def answer(
    question: str,
    business_id: str,
    as_of_date: str,
    *,
    planner: Planner,
    split: str = "train",
    data_root: str | Path = "data/datamate-corpus",
) -> dict[str, Any]:
    """Convenience contract requested by the team plan."""

    return AgentCore(planner=planner, data_root=data_root).answer(
        question=question,
        business_id=business_id,
        as_of_date=as_of_date,
        split=split,
    )
