# Member B — Agent Core

This package implements the question-to-SQL core owned by member B. It does
not modify member A's cache. `context_adapter.py` supports the cache layout
currently present in this repository and the layout proposed in the team plan.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code_member_a\requirements.txt pytest
```

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Current contract

```python
from code_member_b import AgentCore

agent = AgentCore(planner=my_planner)
result = agent.answer(
    question="Tháng này bán được bao nhiêu?",
    business_id="biz_101",
    as_of_date="2026-08-24",
    split="train",
)
```

A planner must implement `plan(prompt) -> dict` and return:

```json
{
  "answer_type": "number",
  "sql": "SELECT ...",
  "confidence": 0.9,
  "reason": null
}
```

The concrete offline-model integration is intentionally pending a team model
decision. Tests use `StaticPlanner` so the data, safety, execution, retry, and
formatting paths can already be verified end to end.
