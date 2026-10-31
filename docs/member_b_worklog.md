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
- Added a T4-compatible Qwen 7B NF4 teacher and execution-guided SQL
  pseudo-label CLI.
- Added strict question/answer ID joining, deterministic stratified pilots,
  three-attempt execution-guided repair, read-only execution, typed gold/alias
  matching, and constant-only SQL rejection.
- Added compact exact-schema prompts, inferred shared-column join hints, stable
  ranking guidance, DuckDB error feedback, and private mismatch feedback. Raised
  the single-sequence output allowance to 512 tokens while retaining the T4-safe
  4,096-token input bound.
- Added question-aware schema retrieval for month-sharded and domain-specific
  tables. Retry prompts now repeat exact schemas and samples for both the most
  relevant tables and tables referenced by failed SQL, preventing truncation
  from hiding date, header/detail, customer, and payment columns.
- Enabled low-temperature single-sequence sampling so corrected attempts do not
  deterministically repeat the same invalid SQL.
- Replaced unstable sampling with deterministic generation after the five-case
  comparison showed a regression in a previously correct ranked list.
- Added verified schema-driven candidates for recurring month-sharded revenue,
  POS gross-profit, and completed-order receivables questions. These templates
  derive physical tables/columns from each profile and never use gold values.
- Added safe ranked-list projection so a correct ID/name/metric ranking is
  normalized to the required one-column label result before verification.
- Fixed the read-only SQL guard to allow DuckDB's safe `REPLACE()` scalar
  function inside an already validated `SELECT` query.
- Executed the three new templates against the real `biz_103`, `biz_123`, and
  `biz_106` databases; their formatted results exactly matched the boolean,
  product name, and five-row receivables gold outputs from the smoke traces.
- Analyzed the 100-question pilot: 32% overall acceptance but only 20/88
  answerable examples produced verified SQL; percent was 0/12. Most rejected
  plans executed but encoded the wrong aggregation or business entity.
- Added schema-driven templates for single-table continuous monthly revenue,
  peak daily revenue, top identified customers, distinct-order cancellation
  rate, and morning-shift order count. Header-row semantics now take precedence
  over misleading physical column names in malformed Excel exports.
- Added mixed ISO/day-first/Excel-serial date parsing and numeric text parsing
  for comma decimals, currency suffixes, and dot-grouped Vietnamese amounts.
- Verified the five new template families against real `biz_128`, `biz_122`,
  `biz_129`, `biz_121`, `biz_102`, and `biz_115` databases; all representative
  outputs matched the pilot gold values.
- Added append-only accepted/rejected checkpoints with resume and interrupted
  final-line recovery.
- Added QLoRA JSONL export that excludes gold answers, traces, and teacher-only
  answer-type guidance.
- Expanded the member B test suite to cover matching, safety, verification,
  resume, unanswerable examples, and QLoRA export.
- Completed the second deterministic 100-question pilot: 40/100 overall and
  28/88 answerable questions produced verified SQL, up from 32/100 and 20/88
  in the first pilot. All 12 unanswerable cases remained correct.
- Analyzed all 60 rejected traces. The dominant execution failures were
  invented/misbound identifiers (61 attempts), missing or malformed scalar SQL
  (28), numeric conversion failures (6), and non-DuckDB functions/tables (4).
- Annotated compact schemas with logical first-row header labels when malformed
  spreadsheet imports stored their headers as data. Strengthened teacher
  guidance to distinguish logical labels from physical SQL identifiers and to
  use DuckDB-safe numeric conversion instead of `TO_NUMBER`/dialect-specific
  functions.
- Added a CPU-only deterministic pass (`--template-only`) and support for trying
  multiple independent schema candidates before invoking the teacher. Expanded
  period parsing to cover Vietnamese "tháng M năm YYYY" phrasing and relative
  day/week/month ranges, plus initial period-revenue, order-count, top-product,
  and top-customer semantic templates.
- Ran the deterministic pass over all 3,000 examples locally in about 80
  seconds. It verified 286 high-precision labels (including all 70
  unanswerable cases) but only 9.5% overall. This establishes that deterministic
  templates are useful as a cheap first stage, not sufficient as the only
  pseudo-label source; schema-specific joins and broader business intents still
  require a semantic compiler or teacher fallback.

## Decisions still required

- Validate the Qwen pseudo-label pilot acceptance rate on a Colab T4 before
  launching all 3,000 examples.
- Decide whether member A will standardize the cache contract or member B's
  adapter will remain the integration boundary.
- Review the remaining rejected pilot traces to decide whether any business
  rules need deterministic templates before full-corpus generation.
