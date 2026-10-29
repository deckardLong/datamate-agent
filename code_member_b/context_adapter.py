"""Compatibility adapter for the cache currently produced by member A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BusinessContext:
    business_id: str
    db_path: Path
    profile: dict[str, Any]
    schema_card: str
    rules: dict[str, Any]
    calendar_path: Path | None


def _find_business_dir(data_root: Path, split: str, business_id: str) -> Path:
    candidates = (
        data_root / split / business_id,
        data_root / business_id,
        data_root / split / "data" / business_id,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find {business_id!r}. Searched: {searched}")


def _find_database(business_dir: Path, business_id: str) -> Path:
    for name in ("db.duckdb", f"{business_id}.duckdb"):
        candidate = business_dir / name
        if candidate.is_file():
            return candidate

    databases = sorted(business_dir.glob("*.duckdb"))
    if len(databases) == 1:
        return databases[0]
    raise FileNotFoundError(f"Cannot select a DuckDB database in {business_dir}")


def build_compatible_schema_card(profile: dict[str, Any]) -> str:
    """Build a compact card from both the old and proposed profile schemas."""

    business_id = profile.get("business_id", "unknown")
    lines = [f"# {business_id} schema"]
    for table_name, table in profile.get("tables", {}).items():
        row_count = table.get("row_count", "?")
        lines.append(f"\n## {table_name} ({row_count} rows)")
        columns = table.get("columns", {})
        rendered = []
        for column_name, metadata in columns.items():
            if isinstance(metadata, dict):
                column_type = metadata.get("type", metadata.get("semantic_type", "UNKNOWN"))
            else:
                column_type = metadata
            rendered.append(f"{column_name} {column_type}")
        lines.append("Columns: " + ", ".join(rendered))
        samples = table.get("samples", [])
        if samples:
            lines.append("Sample: " + json.dumps(samples[0], ensure_ascii=False, default=str))

    notes = profile.get("readme_notes")
    if notes:
        lines.append("\n## Business notes\n" + str(notes)[:4000])
    return "\n".join(lines)[:12000]


def load_business_context(
    split: str,
    business_id: str,
    data_root: str | Path = "data/datamate-corpus",
) -> BusinessContext:
    root = Path(data_root)
    business_dir = _find_business_dir(root, split, business_id)
    db_path = _find_database(business_dir, business_id)

    profile_path = business_dir / "profile.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"Missing profile: {profile_path}")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    card_path = business_dir / "schema_card.md"
    schema_card = (
        card_path.read_text(encoding="utf-8")
        if card_path.is_file()
        else build_compatible_schema_card(profile)
    )

    split_dir = business_dir.parent
    rules_path = split_dir / "rules.json"
    rules = (
        json.loads(rules_path.read_text(encoding="utf-8"))
        if rules_path.is_file()
        else profile.get("rules_summary", {})
    )
    calendar_path = split_dir / "calendar.csv"

    return BusinessContext(
        business_id=business_id,
        db_path=db_path,
        profile=profile,
        schema_card=schema_card,
        rules=rules,
        calendar_path=calendar_path if calendar_path.is_file() else None,
    )
