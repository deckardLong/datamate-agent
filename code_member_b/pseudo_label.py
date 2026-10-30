"""Execution-guided SQL pseudo-label generation for DataMate training data."""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

from .context_adapter import BusinessContext, load_business_context
from .contracts import QueryPlan
from .executor import DuckDBExecutor
from .formatter import format_query_result
from .prompt_builder import build_prompt


class CandidateGenerator(Protocol):
    """Generate structured query-plan candidates for one prompt."""

    model_id: str

    def generate(self, prompt: str, candidate_count: int) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class TrainingRecord:
    id: str
    business_id: str
    question: str
    as_of_date: str
    answer_type: str
    expected_answer: Any
    aliases: list[Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_training_records(
    data_root: str | Path = "data/datamate-corpus",
    split: str = "train",
) -> list[TrainingRecord]:
    """Join questions and answers by ID and validate the local corpus contract."""

    split_dir = Path(data_root) / split
    questions = _read_jsonl(split_dir / "questions.jsonl")
    answers = _read_jsonl(split_dir / "answers.jsonl")
    question_by_id = {str(row["id"]): row for row in questions}
    answer_by_id = {str(row["id"]): row for row in answers}

    if len(question_by_id) != len(questions):
        raise ValueError("Duplicate question IDs")
    if len(answer_by_id) != len(answers):
        raise ValueError("Duplicate answer IDs")
    if question_by_id.keys() != answer_by_id.keys():
        missing_answers = sorted(question_by_id.keys() - answer_by_id.keys())[:5]
        missing_questions = sorted(answer_by_id.keys() - question_by_id.keys())[:5]
        raise ValueError(
            "Question/answer IDs differ: "
            f"missing answers={missing_answers}, missing questions={missing_questions}"
        )

    records = []
    for question_id, question in question_by_id.items():
        answer = answer_by_id[question_id]
        records.append(
            TrainingRecord(
                id=question_id,
                business_id=str(question["business_id"]),
                question=str(question["question"]),
                as_of_date=str(question["as_of_date"]),
                answer_type=str(answer["type"]).lower(),
                expected_answer=answer.get("answer"),
                aliases=list(answer.get("alias") or []),
            )
        )
    return records


def stratified_sample(
    records: Iterable[TrainingRecord],
    size: int,
    seed: int = 42,
) -> list[TrainingRecord]:
    """Select a deterministic pilot spread across answer types and businesses."""

    items = list(records)
    if size <= 0 or size >= len(items):
        return items

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[TrainingRecord]] = defaultdict(list)
    for record in items:
        buckets[(record.answer_type, record.business_id)].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    # First spread each answer type across businesses, then cycle answer types.
    # This prevents a small smoke test from accidentally containing only one or
    # two types while retaining broad business coverage within every type.
    type_records: dict[str, list[TrainingRecord]] = defaultdict(list)
    answer_types = sorted({key[0] for key in buckets})
    for answer_type in answer_types:
        business_keys = sorted(key for key in buckets if key[0] == answer_type)
        rng.shuffle(business_keys)
        while any(buckets[key] for key in business_keys):
            for key in business_keys:
                if buckets[key]:
                    type_records[answer_type].append(buckets[key].pop())

    rng.shuffle(answer_types)
    selected: list[TrainingRecord] = []
    while len(selected) < size:
        progressed = False
        for answer_type in answer_types:
            if type_records[answer_type]:
                selected.append(type_records[answer_type].pop())
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            break
    return selected


def _plain(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _normalized_text(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _sequence_match(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, (list, tuple)) or not isinstance(expected, (list, tuple)):
        return False
    if len(observed) != len(expected):
        return False
    return all(_generic_match(left, right) for left, right in zip(observed, expected))


def _mapping_match(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    if set(observed) != set(expected):
        return False
    return all(_generic_match(observed[key], expected[key]) for key in observed)


def _generic_match(observed: Any, expected: Any) -> bool:
    observed, expected = _plain(observed), _plain(expected)
    if isinstance(observed, bool) or isinstance(expected, bool):
        return observed is expected
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=0.01)
    if isinstance(observed, (list, tuple)) or isinstance(expected, (list, tuple)):
        return _sequence_match(observed, expected)
    if isinstance(observed, dict) or isinstance(expected, dict):
        return _mapping_match(observed, expected)
    if observed is None or expected is None:
        return observed is expected
    return _normalized_text(observed) == _normalized_text(expected)


def answers_match(
    observed: Any,
    expected: Any,
    answer_type: str,
    aliases: Iterable[Any] = (),
) -> bool:
    """Compare a formatted query result with the gold answer and its aliases."""

    aliases = list(aliases)
    candidates = [expected, *aliases]
    if answer_type in {"number", "percent"}:
        for candidate in candidates:
            if isinstance(observed, bool) or isinstance(candidate, bool):
                continue
            try:
                if math.isclose(
                    float(observed), float(candidate), rel_tol=1e-9, abs_tol=0.01
                ):
                    return True
            except (TypeError, ValueError):
                pass
        return False
    if answer_type == "list" and isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            return False
        for index, (observed_item, expected_item) in enumerate(zip(observed, expected)):
            item_aliases = aliases[index] if index < len(aliases) else []
            if not isinstance(item_aliases, list):
                item_aliases = [item_aliases]
            if not any(
                _generic_match(observed_item, candidate)
                for candidate in [expected_item, *item_aliases]
            ):
                return False
        return True
    if answer_type == "table" and isinstance(observed, dict) and isinstance(expected, dict):
        observed_rows, expected_rows = observed.get("rows"), expected.get("rows")
        if not isinstance(observed_rows, list) or not isinstance(expected_rows, list):
            return False
        if len(observed_rows) != len(expected_rows):
            return False
        for row_index, (observed_row, expected_row) in enumerate(
            zip(observed_rows, expected_rows)
        ):
            if not isinstance(observed_row, list) or not isinstance(expected_row, list):
                return False
            if len(observed_row) != len(expected_row):
                return False
            row_aliases = aliases[row_index] if row_index < len(aliases) else []
            if not isinstance(row_aliases, list):
                row_aliases = [row_aliases]
            for column_index, (observed_cell, expected_cell) in enumerate(
                zip(observed_row, expected_row)
            ):
                cell_candidates = [expected_cell]
                if column_index == 0:
                    cell_candidates.extend(row_aliases)
                if not any(
                    _generic_match(observed_cell, candidate)
                    for candidate in cell_candidates
                ):
                    return False
        return True
    return any(_generic_match(observed, candidate) for candidate in candidates)


_FROM = re.compile(r"\bfrom\b", re.IGNORECASE)


def reject_constant_only_sql(sql: str) -> None:
    """Reject answer leakage such as ``SELECT 123`` that never reads the database."""

    if not _FROM.search(sql):
        raise ValueError("SQL does not read a database table")


def _ascii_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _relevant_table_names(
    record: TrainingRecord,
    context: BusinessContext,
    limit: int = 10,
) -> list[str]:
    """Rank tables using question vocabulary and explicit month/year hints."""

    question = _ascii_key(record.question)
    question_tokens = set(re.findall(r"[a-z0-9]+", question))
    month_years = {
        (int(month), year)
        for month, year in re.findall(r"(\d{1,2})\s*/\s*(\d{4})", question)
        if 1 <= int(month) <= 12
    }
    concepts = {
        "revenue": (
            ("doanh thu", "tong tien", "ban duoc"),
            ("donhang", "don_hang", "hoadon", "hoa_don", "ban_hang"),
        ),
        "profit": (
            ("lai gop", "loi nhuan", "gia von"),
            ("chitiet", "chi_tiet", "sanpham", "san_pham", "don_hang"),
        ),
        "debt": (
            ("cong no", "dang no", "con no"),
            ("hoadon", "hoa_don", "khachhang", "khach_hang", "soquy", "thanh_toan"),
        ),
    }

    scored: list[tuple[int, int, str]] = []
    for position, table_name in enumerate(context.profile.get("tables", {})):
        key = _ascii_key(str(table_name))
        compact_key = re.sub(r"[^a-z0-9]", "", key)
        table_tokens = set(re.findall(r"[a-z0-9]+", key))
        score = 2 * len(question_tokens & table_tokens)
        for month, year in month_years:
            date_patterns = (
                f"{month:02d}_{year}",
                f"{month}_{year}",
                f"t{month:02d}_{year}",
                f"{year}_{month:02d}",
            )
            if any(pattern in key for pattern in date_patterns):
                score += 80
            elif str(year) in key and f"{month:02d}" in key:
                score += 40
        for question_phrases, table_markers in concepts.values():
            if any(phrase in question for phrase in question_phrases):
                if any(
                    re.sub(r"[^a-z0-9]", "", marker) in compact_key
                    for marker in table_markers
                ):
                    score += 18
        scored.append((score, -position, str(table_name)))

    scored.sort(reverse=True)
    return [table_name for _, _, table_name in scored[:limit]]


def _compact_schema(
    context: BusinessContext,
    table_names: Iterable[str] | None = None,
    max_chars: int = 7500,
) -> str:
    lines = []
    tables = context.profile.get("tables", {})
    names = list(table_names) if table_names is not None else list(tables)
    for table_name in names:
        table = tables.get(table_name, {})
        columns = []
        for column_name, metadata in table.get("columns", {}).items():
            column_type = (
                metadata.get("type", metadata.get("semantic_type", "UNKNOWN"))
                if isinstance(metadata, dict)
                else metadata
            )
            columns.append(f'"{column_name}" {column_type}')
        line = f'TABLE "{table_name}" ({", ".join(columns)})'
        if sum(len(item) + 1 for item in lines) + len(line) > max_chars:
            remaining = max_chars - sum(len(item) + 1 for item in lines)
            if remaining > 80:
                lines.append(line[:remaining])
            break
        lines.append(line)
    return "\n".join(lines)


def _compact_samples(
    context: BusinessContext,
    table_names: Iterable[str] | None = None,
    max_chars: int = 1800,
) -> str:
    samples = []
    tables = context.profile.get("tables", {})
    names = list(table_names) if table_names is not None else list(tables)
    for table_name in names:
        table = tables.get(table_name, {})
        rows = table.get("samples") or []
        if not rows:
            continue
        rendered = f'"{table_name}": ' + json.dumps(
            rows[0], ensure_ascii=False, default=str, separators=(",", ":")
        )
        used = sum(len(item) + 1 for item in samples)
        if used + len(rendered) > max_chars:
            remaining = max_chars - used
            if remaining > 80:
                samples.append(rendered[:remaining])
            break
        samples.append(rendered)
    return "\n".join(samples)


def _compact_relationships(
    context: BusinessContext,
    table_names: Iterable[str] | None = None,
    max_chars: int = 1400,
) -> str:
    """Infer bounded join hints from exact column names shared by tables."""

    table_columns: dict[str, set[str]] = {}
    tables = context.profile.get("tables", {})
    names = list(table_names) if table_names is not None else list(tables)
    for table_name in names:
        table = tables.get(table_name, {})
        table_columns[str(table_name)] = {
            str(column_name)
            for column_name in table.get("columns", {})
            if not str(column_name).endswith("_raw")
        }

    hints: list[str] = []
    table_names = list(table_columns)
    for left_index, left in enumerate(table_names):
        for right in table_names[left_index + 1 :]:
            shared = sorted(table_columns[left] & table_columns[right])
            if not shared:
                continue
            line = f'"{left}" <-> "{right}": {", ".join(shared)}'
            used = sum(len(item) + 1 for item in hints)
            if used + len(line) > max_chars:
                return "\n".join(hints)
            hints.append(line)
    return "\n".join(hints) or "No shared column names were detected; inspect schema and samples."


def build_teacher_prompt(record: TrainingRecord, context: BusinessContext) -> str:
    """Build a compact schema-first prompt that fits a T4 without gold values."""

    rules = json.dumps(
        context.rules, ensure_ascii=False, default=str, separators=(",", ":")
    )[:1600]
    notes = str(context.profile.get("readme_notes") or "")[:900]
    relevant_tables = _relevant_table_names(record, context)
    all_table_names = ", ".join(
        f'"{name}"' for name in context.profile.get("tables", {})
    )[:3000]
    shape = {
        "number": "exactly one numeric row and one column",
        "percent": "exactly one numeric row and one column, expressed in percent units",
        "string": "exactly one text row and one column",
        "date": "exactly one ISO date row and one column",
        "boolean": "exactly one BOOLEAN row and one column",
        "list": "exactly one column with ordered result rows; do not return metric columns",
        "table": "the requested columns and ordered rows",
    }.get(record.answer_type, "the requested result")
    return "\n\n".join(
        [
            "You generate one safe read-only DuckDB SELECT query for Vietnamese "
            "business questions. Use only the exact tables and columns below. "
            "Never invent identifiers and never hard-code the answer. Return one "
            "JSON object only with keys answer_type, sql, confidence, reason.",
            "Business rules (possibly abbreviated):\n" + rules,
            "Business notes (possibly abbreviated):\n" + notes,
            "All exact table names (never shorten them):\n" + all_table_names,
            "Representative rows for likely relevant tables (values are examples only):\n"
            + _compact_samples(context, relevant_tables),
            "Exact schema for likely relevant tables:\n"
            + _compact_schema(context, relevant_tables),
            "Possible joins inferred from identical column names (verify semantics):\n"
            + _compact_relationships(context, relevant_tables),
            "Query discipline: join header/detail or lookup tables on exact ID columns. "
            "For ranked lists, group by both stable ID and display label, order by the "
            "requested aggregate DESC and then a stable ID/label ASC tie-breaker, and "
            "apply LIMIT only after aggregation. Avoid WHERE IN for ordered rankings. "
            "Keep the JSON and SQL concise.",
            f"Business: {record.business_id}\nAs-of date: {record.as_of_date}\n"
            f"Required answer_type: {record.answer_type}\n"
            f"Required SQL result shape: {shape}\n"
            f"Question: {record.question}",
        ]
    )


_SQL_TABLE = re.compile(r"\b(?:from|join)\s+[\"`]?([a-zA-Z0-9_]+)", re.IGNORECASE)


def _repair_table_names(
    context: BusinessContext,
    candidate: dict[str, Any],
    limit: int = 6,
) -> list[str]:
    """Select exact and closely related tables used by a failed SQL plan."""

    tables = context.profile.get("tables", {})
    table_names = list(tables)
    by_key = {_ascii_key(str(name)): str(name) for name in table_names}
    referenced = _SQL_TABLE.findall(str(candidate.get("sql") or ""))
    selected: list[str] = []
    for name in referenced:
        key = _ascii_key(name)
        exact = by_key.get(key)
        if exact and exact not in selected:
            selected.append(exact)
            continue
        compact = re.sub(r"[^a-z0-9]", "", key)
        close = sorted(
            table_names,
            key=lambda item: (
                compact not in re.sub(r"[^a-z0-9]", "", _ascii_key(str(item))),
                abs(len(str(item)) - len(name)),
            ),
        )
        if close and close[0] not in selected:
            selected.append(str(close[0]))

    base_columns = set().union(
        *(
            set(tables[name].get("columns", {}))
            for name in selected
            if name in tables
        )
    ) if selected else set()
    related = sorted(
        (name for name in table_names if name not in selected),
        key=lambda name: len(base_columns & set(tables[name].get("columns", {}))),
        reverse=True,
    )
    selected.extend(
        str(name)
        for name in related
        if base_columns & set(tables[name].get("columns", {}))
    )
    return selected[:limit]


def _retry_teacher_prompt(
    base_prompt: str,
    candidate: dict[str, Any],
    context: BusinessContext,
    record: TrainingRecord,
    *,
    error: str | None = None,
    observed: Any = None,
) -> str:
    """Ask the teacher to repair one failed attempt without revealing gold data."""

    previous_sql = str(candidate.get("sql") or "<no valid SQL>")[:2200]
    if error is not None:
        feedback = (
            "The previous candidate could not be validated. DuckDB/validator feedback: "
            + error[:1200]
        )
    else:
        rendered_observed = json.dumps(
            observed, ensure_ascii=False, default=str, separators=(",", ":")
        )[:800]
        feedback = (
            "The query executed, but failed private answer verification. The private "
            "answer is intentionally hidden. Reconsider date boundaries, joins, filters, "
            "aggregation, ranking/ties, units, and the required output shape. Observed "
            f"result from the previous SQL: {rendered_observed}"
        )
    repair_tables = list(
        dict.fromkeys(
            _relevant_table_names(record, context, limit=6)
            + _repair_table_names(context, candidate, limit=6)
        )
    )[:8]
    repair_schema = _compact_schema(context, repair_tables, max_chars=4800)
    repair_samples = _compact_samples(context, repair_tables, max_chars=1200)
    return (
        base_prompt
        + "\n\nCORRECTION REQUIRED\nPrevious SQL:\n"
        + previous_sql
        + "\nFeedback:\n"
        + feedback
        + "\nDo not reuse an identifier named in the error. Copy identifiers exactly "
        "from this repair schema:\n"
        + repair_schema
        + "\nRelevant example rows:\n"
        + repair_samples
        + "\nQuestion: "
        + record.question
        + "\nReturn one corrected JSON object only. Do not repeat the same SQL."
    )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _header_mapping(table: dict[str, Any]) -> dict[str, str]:
    """Map human header labels stored in the first data row to physical columns."""

    rows = table.get("samples") or []
    if not rows or not isinstance(rows[0], dict):
        return {}
    return {
        _ascii_key(str(label)).strip(): str(column)
        for column, label in rows[0].items()
        if label is not None
    }


def _find_header_column(mapping: dict[str, str], *labels: str) -> str | None:
    for label in labels:
        expected = _ascii_key(label)
        for actual, column in mapping.items():
            if actual == expected or expected in actual:
                return column
    return None


def _numeric_text_sql(expression: str) -> str:
    return (
        "TRY_CAST(REPLACE(CAST(" + expression
        + " AS VARCHAR), ',', '') AS DOUBLE)"
    )


def _completed_status_sql(expression: str) -> str:
    return (
        f"LOWER(CAST({expression} AS VARCHAR)) IN "
        "('completed', 'complete', 'xong', 'hoàn thành', 'ok', 'đã hoàn thành')"
    )


def _month_years(question: str) -> list[tuple[int, int]]:
    return [
        (int(month), int(year))
        for month, year in re.findall(
            r"(\d{1,2})\s*/\s*(\d{4})", _ascii_key(question)
        )
        if 1 <= int(month) <= 12
    ]


def _monthly_revenue_candidate(
    record: TrainingRecord, context: BusinessContext
) -> dict[str, Any] | None:
    question = _ascii_key(record.question)
    periods = _month_years(record.question)
    if "doanh thu" not in question or "tang lien tuc" not in question or len(periods) < 2:
        return None

    tables = context.profile.get("tables", {})
    expressions: list[str] = []
    for month, year in periods:
        patterns = (f"{month:02d}_{year}", f"{month}_{year}", f"t{month:02d}_{year}")
        table_name = next(
            (
                str(name)
                for name in tables
                if any(pattern in _ascii_key(str(name)) for pattern in patterns)
            ),
            None,
        )
        if table_name is None:
            return None
        mapping = _header_mapping(tables[table_name])
        amount = _find_header_column(mapping, "tổng tiền", "thành tiền")
        status = _find_header_column(mapping, "trạng thái đơn hàng", "trạng thái")
        if amount is None:
            return None
        amount_sql = _numeric_text_sql(_quote_identifier(amount))
        where = ""
        if status is not None:
            status_sql = f"LOWER(COALESCE(CAST({_quote_identifier(status)} AS VARCHAR), ''))"
            where = (
                f" WHERE {status_sql} NOT LIKE '%hủy%'"
                f" AND {status_sql} NOT LIKE '%huy%'"
                f" AND {status_sql} NOT LIKE '%trả%'"
                f" AND {status_sql} NOT LIKE '%tra%'"
            )
        expressions.append(
            f"(SELECT SUM({amount_sql}) FROM {_quote_identifier(table_name)}{where})"
        )

    comparisons = [
        f"{left} < {right}" for left, right in zip(expressions, expressions[1:])
    ]
    return {
        "answer_type": "boolean",
        "sql": "SELECT " + " AND ".join(comparisons) + " AS is_increasing",
        "confidence": 1.0,
        "reason": "schema_template: month-sharded continuous revenue",
    }


def _gross_profit_candidate(
    record: TrainingRecord, context: BusinessContext
) -> dict[str, Any] | None:
    question = _ascii_key(record.question)
    periods = _month_years(record.question)
    if not any(phrase in question for phrase in ("lai gop", "loi nhuan gop")) or not periods:
        return None

    tables = context.profile.get("tables", {})
    column_sets = {
        str(name): set(table.get("columns", {})) for name, table in tables.items()
    }
    detail = next(
        (
            name
            for name, columns in column_sets.items()
            if {"mahoadon", "mahanghoa", "trigia", "giavonlucban", "sl"} <= columns
        ),
        None,
    )
    header = next(
        (
            name
            for name, columns in column_sets.items()
            if "mahoadon" in columns and {"ngaylap", "tinhtrang"} <= columns
        ),
        None,
    )
    product = next(
        (
            name
            for name, columns in column_sets.items()
            if {"mahanghoa", "tengoi"} <= columns and name != detail
        ),
        None,
    )
    if not all((detail, header, product)):
        return None

    month, year = periods[0]
    start = f"{year:04d}-{month:02d}-01"
    next_month, next_year = (1, year + 1) if month == 12 else (month + 1, year)
    end = f"{next_year:04d}-{next_month:02d}-01"
    detail_columns = column_sets[detail]
    header_columns = column_sets[header]
    product_columns = column_sets[product]
    detail_header_keys = [
        key for key in ("idcuahang", "mahoadon") if key in detail_columns & header_columns
    ]
    detail_product_keys = [
        key for key in ("idcuahang", "mahanghoa") if key in detail_columns & product_columns
    ]
    if "mahoadon" not in detail_header_keys or "mahanghoa" not in detail_product_keys:
        return None
    join_header = " AND ".join(f"d.{key} = h.{key}" for key in detail_header_keys)
    join_product = " AND ".join(f"d.{key} = p.{key}" for key in detail_product_keys)
    sql = (
        f"SELECT p.tengoi FROM {_quote_identifier(detail)} d "
        f"JOIN {_quote_identifier(header)} h ON {join_header} "
        f"JOIN {_quote_identifier(product)} p ON {join_product} "
        f"WHERE TRY_CAST(h.ngaylap AS DATE) >= DATE '{start}' "
        f"AND TRY_CAST(h.ngaylap AS DATE) < DATE '{end}' "
        f"AND {_completed_status_sql('h.tinhtrang')} "
        "GROUP BY p.mahanghoa, p.tengoi "
        "ORDER BY SUM(d.trigia - d.giavonlucban * d.sl) DESC, p.mahanghoa ASC LIMIT 1"
    )
    return {
        "answer_type": record.answer_type,
        "sql": sql,
        "confidence": 1.0,
        "reason": "schema_template: POS gross profit",
    }


def _debt_candidate(
    record: TrainingRecord, context: BusinessContext
) -> dict[str, Any] | None:
    question = _ascii_key(record.question)
    if not any(phrase in question for phrase in ("dang no", "con no", "cong no")):
        return None

    for table_name, table in context.profile.get("tables", {}).items():
        mapping = _header_mapping(table)
        customer_id = _find_header_column(mapping, "mã kh", "mã khách hàng")
        customer_name = _find_header_column(mapping, "khách hàng", "tên khách hàng")
        total = _find_header_column(mapping, "tổng sau giảm giá", "tổng thanh toán")
        paid = _find_header_column(mapping, "khách đã trả")
        status = _find_header_column(mapping, "trạng thái")
        created_at = _find_header_column(mapping, "thời gian tạo")
        if not all((customer_id, customer_name, total, paid, status)):
            continue
        limit_match = re.search(r"\b(\d+)\s*(?:khach|nguoi)", question)
        limit = int(limit_match.group(1)) if limit_match else 5
        id_sql = _quote_identifier(customer_id)
        name_sql = _quote_identifier(customer_name)
        debt_sql = (
            f"{_numeric_text_sql(_quote_identifier(total))} - "
            f"{_numeric_text_sql(_quote_identifier(paid))}"
        )
        filters = [
            f"{id_sql} IS NOT NULL",
            _completed_status_sql(_quote_identifier(status)),
        ]
        if created_at:
            filters.append(
                f"TRY_CAST({_quote_identifier(created_at)} AS TIMESTAMP) < "
                f"DATE '{record.as_of_date}' + INTERVAL 1 DAY"
            )
        sql = (
            f"SELECT ANY_VALUE({name_sql}) AS khach_hang, SUM({debt_sql}) AS so_tien_no "
            f"FROM {_quote_identifier(str(table_name))} WHERE "
            + " AND ".join(filters)
            + f" GROUP BY {id_sql} HAVING SUM({debt_sql}) > 0 "
            + f"ORDER BY so_tien_no DESC, {id_sql} ASC LIMIT {limit}"
        )
        return {
            "answer_type": "table",
            "sql": sql,
            "confidence": 1.0,
            "reason": "schema_template: completed-order receivables",
        }
    return None


def build_schema_candidate(
    record: TrainingRecord, context: BusinessContext
) -> dict[str, Any] | None:
    """Build high-precision plans for recurring schema-defined business metrics."""

    return (
        _monthly_revenue_candidate(record, context)
        or _gross_profit_candidate(record, context)
        or _debt_candidate(record, context)
    )


def _list_projection_sql(sql: str, columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Project a ranked multi-column query down to its most likely display label."""

    def score(index: int, column: str) -> tuple[int, int]:
        key = _ascii_key(column)
        value = 0
        if any(marker in key for marker in ("ten", "name", "label", "khach_hang", "san_pham")):
            value += 12
        if any(marker in key for marker in ("tong", "sum", "count", "amount", "so_tien")):
            value -= 10
        if key == "id" or key.startswith("id_") or key.startswith("ma_") or key.endswith("_id"):
            value -= 8
        non_null = [row[index] for row in rows if index < len(row) and row[index] is not None]
        if non_null and all(isinstance(item, str) for item in non_null):
            value += 3
        return value, -index

    selected = max(range(len(columns)), key=lambda index: score(index, columns[index]))
    inner = sql.strip().removesuffix(";").rstrip()
    return (
        f"SELECT {_quote_identifier(columns[selected])} FROM ({inner}) "
        "AS _datamate_ranked"
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size:
        with path.open("rb+") as binary:
            binary.seek(-1, 2)
            if binary.read(1) != b"\n":
                binary.seek(0, 2)
                binary.write(b"\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _generated_rows(path: Path) -> list[dict[str, Any]]:
    """Read valid checkpoint rows, tolerating a partial line after interruption."""

    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring incomplete checkpoint line {path}:{line_number}")
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _completed_ids(paths: Iterable[Path]) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        for row in _generated_rows(path):
            if row.get("id"):
                completed.add(str(row["id"]))
    return completed


def _accepted_row(
    record: TrainingRecord,
    sql: str | None,
    observed: Any,
    model_id: str,
    candidate_count: int,
    prompt: str,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        **asdict(record),
        "sql": sql,
        "observed_answer": observed,
        "verified": True,
        "teacher_model": model_id,
        "candidate_count": candidate_count,
        "confidence": confidence,
        "prompt": prompt,
    }


class PseudoLabelPipeline:
    """Generate, execute, verify, and checkpoint SQL pseudo-labels."""

    def __init__(
        self,
        generator: CandidateGenerator,
        data_root: str | Path = "data/datamate-corpus",
        split: str = "train",
        output_dir: str | Path = "outputs/pseudo_labels",
        candidate_count: int = 3,
        max_rows: int = 1000,
        query_timeout_seconds: float = 30.0,
    ):
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        self.generator = generator
        self.data_root = Path(data_root)
        self.split = split
        self.output_dir = Path(output_dir)
        self.candidate_count = candidate_count
        self.executor = DuckDBExecutor(max_rows, query_timeout_seconds)
        self.accepted_path = self.output_dir / "accepted.jsonl"
        self.rejected_path = self.output_dir / "rejected.jsonl"

    def _process(self, record: TrainingRecord) -> tuple[bool, dict[str, Any]]:
        context = load_business_context(self.split, record.business_id, self.data_root)
        training_prompt = build_prompt(record.question, record.as_of_date, context)
        teacher_prompt = build_teacher_prompt(record, context)
        if record.answer_type == "unanswerable":
            return True, _accepted_row(
                record,
                sql=None,
                observed=None,
                model_id="gold-unanswerable",
                candidate_count=0,
                prompt=training_prompt,
            )

        attempts: list[dict[str, Any]] = []
        current_prompt = teacher_prompt
        schema_candidate = build_schema_candidate(record, context)
        for index in range(1, self.candidate_count + 1):
            source = "schema_template" if index == 1 and schema_candidate else "teacher"
            if source == "schema_template":
                candidate = schema_candidate
            else:
                try:
                    generated = self.generator.generate(current_prompt, 1)
                    if not generated:
                        raise ValueError("Teacher returned no candidate")
                    candidate = generated[0]
                except Exception as exc:
                    generation_error = f"{type(exc).__name__}: {exc}"
                    attempts.append(
                        {"candidate_index": index, "generation_error": generation_error}
                    )
                    return False, {
                        **asdict(record),
                        "verified": False,
                        "teacher_model": self.generator.model_id,
                        "reason": f"generation_error: {generation_error}",
                        "attempts": attempts,
                    }

            attempt: dict[str, Any] = {
                "candidate_index": index,
                "source": source,
                "plan": candidate,
            }
            try:
                plan_data = dict(candidate)
                plan_data["answer_type"] = record.answer_type
                plan = QueryPlan.from_mapping(plan_data)
                sql = plan.sql or ""
                reject_constant_only_sql(sql)
                query_result = self.executor.execute(context.db_path, sql)
                if record.answer_type == "list" and len(query_result.columns) > 1:
                    original_sql = sql
                    sql = _list_projection_sql(
                        sql, query_result.columns, query_result.rows
                    )
                    query_result = self.executor.execute(context.db_path, sql)
                    attempt["normalized_from_sql"] = original_sql
                observed = format_query_result(query_result, record.answer_type)  # type: ignore[arg-type]
                matched = answers_match(
                    observed,
                    record.expected_answer,
                    record.answer_type,
                    record.aliases,
                )
                attempt.update(sql=sql, observed_answer=observed, matched=matched)
                if matched:
                    attempts.append(attempt)
                    row = _accepted_row(
                        record,
                        sql=sql,
                        observed=observed,
                        model_id=(
                            "schema-template"
                            if source == "schema_template"
                            else self.generator.model_id
                        ),
                        candidate_count=index,
                        prompt=training_prompt,
                        confidence=plan.confidence,
                    )
                    row["attempts"] = attempts
                    return True, row
                current_prompt = _retry_teacher_prompt(
                    teacher_prompt,
                    candidate,
                    context,
                    record,
                    observed=observed,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempt["error"] = error
                current_prompt = _retry_teacher_prompt(
                    teacher_prompt,
                    candidate,
                    context,
                    record,
                    error=error,
                )
            attempts.append(attempt)

        return False, {
            **asdict(record),
            "verified": False,
            "teacher_model": self.generator.model_id,
            "reason": "no_candidate_matched_gold",
            "attempts": attempts,
        }

    def run(self, records: Iterable[TrainingRecord]) -> dict[str, Any]:
        records = list(records)
        completed = _completed_ids([self.accepted_path, self.rejected_path])
        accepted_count = rejected_count = skipped_count = 0
        by_type: dict[str, dict[str, int]] = defaultdict(
            lambda: {"accepted": 0, "rejected": 0, "skipped": 0}
        )

        for index, record in enumerate(records, start=1):
            if record.id in completed:
                skipped_count += 1
                by_type[record.answer_type]["skipped"] += 1
                continue
            accepted, row = self._process(record)
            if accepted:
                _append_jsonl(self.accepted_path, row)
                accepted_count += 1
                by_type[record.answer_type]["accepted"] += 1
            else:
                _append_jsonl(self.rejected_path, row)
                rejected_count += 1
                by_type[record.answer_type]["rejected"] += 1
            print(
                f"[{index}/{len(records)}] {record.id} {record.answer_type}: "
                f"{'accepted' if accepted else 'rejected'}",
                flush=True,
            )

        processed = accepted_count + rejected_count
        accepted_rows = _generated_rows(self.accepted_path)
        rejected_rows = _generated_rows(self.rejected_path)
        total_by_type: dict[str, dict[str, int]] = defaultdict(
            lambda: {"accepted": 0, "rejected": 0}
        )
        for row in accepted_rows:
            total_by_type[str(row["answer_type"])]["accepted"] += 1
        for row in rejected_rows:
            total_by_type[str(row["answer_type"])]["rejected"] += 1
        total_processed = len(accepted_rows) + len(rejected_rows)
        return {
            "selected": len(records),
            "processed_this_run": processed,
            "accepted_this_run": accepted_count,
            "rejected_this_run": rejected_count,
            "skipped_existing": skipped_count,
            "acceptance_rate": accepted_count / processed if processed else None,
            "by_type_this_run": dict(by_type),
            "total_processed": total_processed,
            "total_accepted": len(accepted_rows),
            "total_rejected": len(rejected_rows),
            "total_acceptance_rate": (
                len(accepted_rows) / total_processed if total_processed else None
            ),
            "by_type_total": dict(total_by_type),
            "accepted_path": str(self.accepted_path),
            "rejected_path": str(self.rejected_path),
        }


def export_qlora_jsonl(
    accepted_path: str | Path,
    output_path: str | Path,
) -> int:
    """Export verified labels without gold answers or execution traces."""

    accepted_path, output_path = Path(accepted_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in _generated_rows(accepted_path):
            completion = {
                "answer_type": row["answer_type"],
                "sql": row.get("sql"),
                "confidence": row.get("confidence", 1.0),
                "reason": None,
            }
            example = {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(completion, ensure_ascii=False),
                    },
                ],
                "id": row["id"],
                "business_id": row["business_id"],
            }
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count
