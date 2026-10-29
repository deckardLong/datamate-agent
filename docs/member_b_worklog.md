# Member B work log

## 2026-10-30

- Created branch `feat/B-agent-core` (already active when implementation started).
- Added `code_member_b` without modifying member A's implementation.
- Added a compatibility context adapter for both `db.duckdb` and
  `<business_id>.duckdb`, with schema-card fallback from `profile.json`.
- Added dependency-free answer and query-plan contracts.
- Added a conservative read-only SQL guard.
- Added a bounded, read-only DuckDB executor.
- Added output formatting for number, percent, string, date, boolean, list,
  and table answers.
- Added the Agent Core orchestration loop with up to three attempts and safe
  abstention after repeated failures.
- Added a planner protocol and deterministic static planner for tests.
- Added prompt construction with schema, rules, question, and as-of date.
- Added unit tests for contracts, SQL safety, formatting, and member A cache
  compatibility.
- Created a repository-local `.venv` and installed member A's declared
  dependencies plus pytest. Removed the accidentally created `D:\.venv`.
- Added a 30-second default query timeout using DuckDB interruption.
- Ran an end-to-end query against the real `biz_101` cache. The query
  `SELECT COUNT(*) FROM ban_hang` returned `56501` as a normalized number.
- Added an integration test proving that an unsafe first query is rejected and
  a corrected second query can succeed.
- Added local `data/`, `cache/`, `outputs/`, and `models/` to `.gitignore` to
  prevent accidental commits of datasets, generated artifacts, and weights.

## Decisions still required

- Choose the first offline LLM checkpoint and inference configuration.
- Decide whether member A will standardize the cache contract or member B's
  adapter will remain the integration boundary.
- Obtain the missing Train `questions.jsonl`, `answers.jsonl`, `rulebook.md`,
  and `calendar.csv` before accuracy evaluation.
