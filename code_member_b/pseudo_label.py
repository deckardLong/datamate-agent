"""Execution-guided SQL pseudo-label generation for DataMate training data."""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

from .context_adapter import load_business_context
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


def build_training_prompt(record: TrainingRecord, data_root: Path, split: str) -> str:
    """Build the inference-time prompt used later for QLoRA examples."""

    context = load_business_context(split, record.business_id, data_root)
    return build_prompt(record.question, record.as_of_date, context)


def build_teacher_prompt(training_prompt: str, answer_type: str) -> str:
    """Add gold type guidance for SQL generation without exposing the gold value."""

    return (
        training_prompt
        + "\n\n"
        + f"Required answer_type: {answer_type}.\n"
        + "Generate SQL with exactly the result shape required by that type.\n"
        + "Return one JSON object only, without Markdown or commentary.\n"
        + "The SQL must derive its answer from the supplied tables; never return a "
        + "hard-coded answer constant."
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
        training_prompt = build_training_prompt(record, self.data_root, self.split)
        teacher_prompt = build_teacher_prompt(training_prompt, record.answer_type)
        if record.answer_type == "unanswerable":
            return True, _accepted_row(
                record,
                sql=None,
                observed=None,
                model_id="gold-unanswerable",
                candidate_count=0,
                prompt=training_prompt,
            )

        context = load_business_context(self.split, record.business_id, self.data_root)
        attempts: list[dict[str, Any]] = []
        accepted: list[tuple[int, dict[str, Any], Any, QueryPlan]] = []
        try:
            candidates = self.generator.generate(teacher_prompt, self.candidate_count)
        except Exception as exc:
            return False, {
                **asdict(record),
                "verified": False,
                "teacher_model": self.generator.model_id,
                "reason": f"generation_error: {type(exc).__name__}: {exc}",
                "attempts": attempts,
            }

        for index, candidate in enumerate(candidates, start=1):
            attempt: dict[str, Any] = {"candidate_index": index, "plan": candidate}
            try:
                plan_data = dict(candidate)
                plan_data["answer_type"] = record.answer_type
                plan = QueryPlan.from_mapping(plan_data)
                sql = plan.sql or ""
                reject_constant_only_sql(sql)
                query_result = self.executor.execute(context.db_path, sql)
                observed = format_query_result(query_result, record.answer_type)  # type: ignore[arg-type]
                matched = answers_match(
                    observed,
                    record.expected_answer,
                    record.answer_type,
                    record.aliases,
                )
                attempt.update(sql=sql, observed_answer=observed, matched=matched)
                if matched:
                    accepted.append((len(sql), attempt, observed, plan))
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)

        if not accepted:
            return False, {
                **asdict(record),
                "verified": False,
                "teacher_model": self.generator.model_id,
                "reason": "no_candidate_matched_gold",
                "attempts": attempts,
            }

        _, chosen_attempt, observed, plan = min(accepted, key=lambda item: item[0])
        row = _accepted_row(
            record,
            sql=chosen_attempt["sql"],
            observed=observed,
            model_id=self.generator.model_id,
            candidate_count=len(candidates),
            prompt=training_prompt,
            confidence=plan.confidence,
        )
        row["attempts"] = attempts
        return True, row

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
