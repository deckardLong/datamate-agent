"""Planner interface; concrete vLLM integration is deliberately pluggable."""

from __future__ import annotations

from typing import Any, Protocol


class Planner(Protocol):
    def plan(self, prompt: str) -> dict[str, Any]: ...


class StaticPlanner:
    """Deterministic planner used by tests and early end-to-end checks."""

    def __init__(self, plans: list[dict[str, Any]]):
        self._plans = iter(plans)

    def plan(self, prompt: str) -> dict[str, Any]:
        del prompt
        return next(self._plans)
