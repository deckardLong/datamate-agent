from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from .rules import compact_rules
from .schema import TOPIC_KW
from .utils import has_kw, sanitize_value

DB_FILENAME = "db.duckdb"


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _topics_for_table(table: str, cols: list[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for topic, kws in TOPIC_KW.items():
        hits = [c for c in cols if has_kw(c, kws)]
        if has_kw(table, kws):
            hits.insert(0, f"<table:{table}>")
        if hits:
            found[topic] = hits
    return found


def _representative_samples(conn, table: str, n_rows: int, order_col: Optional[str], k: int = 6) -> list[dict]:
    if n_rows == 0:
        return []
    idx = sorted({int(x) for x in np.linspace(1, n_rows, min(k, n_rows))})
    df = conn.execute(
        f"SELECT * EXCLUDE (_rn) FROM (SELECT *, row_number() OVER () AS _rn FROM {_q(table)}) "
        f"WHERE _rn IN ({','.join(map(str, idx))})").df()
    frames = [df]
    if order_col:
        frames.append(conn.execute(
            f"SELECT * FROM {_q(table)} WHERE {_q(order_col)} IS NOT NULL ORDER BY {_q(order_col)} DESC LIMIT 1").df())
    allv = pd.concat(frames, ignore_index=True).astype(object).drop_duplicates(
        subset=[c for c in df.columns if df[c].map(lambda v: isinstance(v, (str, int, float, type(None), date, datetime, pd.Timestamp))).all()] or None)
    return [{k: sanitize_value(v) for k, v in r.items()} for r in allv.to_dict(orient="records")][: k + 1]


def build_profile(conn, business_id: str, metas: dict[str, dict], dirties: dict[str, dict],
                  sources: dict[str, str], readme: str, rules: dict) -> dict:
    tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
    prof_tables: dict[str, dict] = {}
    pk: dict[str, list[str]] = {}
    uniq_cols: dict[str, set[str]] = {}
    date_range: dict[str, dict] = {}
    inventory_cols: dict[str, dict] = {"money": {}, "quantity": {}, "date": {}, "id": {}, "name": {}}

    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {_q(t)}").fetchone()[0]
        info = conn.execute(f"PRAGMA table_info({_q(t)})").fetchall()
        cols: dict[str, dict] = {}
        uniq_cols[t] = set()
        for _, cname, ctype, *_ in info:
            m = dict(metas.get(t, {}).get(cname, {"semantic_type": "text"}))
            if cname.endswith("_raw") and cname[:-4] in metas.get(t, {}):
                m = {"semantic_type": "raw_text", "normalized_column": cname[:-4]}
            m["type"] = ctype
            stats = conn.execute(
                f"SELECT COUNT({_q(cname)}), approx_count_distinct({_q(cname)}) FROM {_q(t)}").fetchone()
            m["non_null"], m["approx_distinct"] = int(stats[0]), int(stats[1])
            sem = m.get("semantic_type")
            if sem in ("id", "text", "status", "unit", "name") and ctype in ("VARCHAR",) and 0 < m["approx_distinct"] <= 30 \
                    and sem in ("status", "unit", "text"):
                vals = conn.execute(
                    f"SELECT {_q(cname)}, COUNT(*) c FROM {_q(t)} WHERE {_q(cname)} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY c DESC LIMIT 15").fetchall()
                m["top_values"] = [{"value": v, "count": int(c)} for v, c in vals]
            if sem == "id" and n > 0:
                exact = conn.execute(f"SELECT COUNT(DISTINCT {_q(cname)}) FROM {_q(t)}").fetchone()[0]
                m["is_unique"] = bool(exact == n and m["non_null"] == n)
                if m["is_unique"]:
                    uniq_cols[t].add(cname)
            if ctype == "DATE" and m["non_null"]:
                lo, hi = conn.execute(f"SELECT MIN({_q(cname)}), MAX({_q(cname)}) FROM {_q(t)}").fetchone()
                m["min"], m["max"] = str(lo), str(hi)
                date_range.setdefault(t, {})[cname] = {"min": str(lo), "max": str(hi)}
            if sem in inventory_cols:
                inventory_cols[sem].setdefault(t, []).append(cname)
            cols[cname] = m
        pk[t] = sorted(uniq_cols[t], key=lambda c: (not c.startswith("ma_"), c))[:1]
        date_cols = [c for c, m in cols.items() if m.get("type") == "DATE"]
        money_cols = [c for c, m in cols.items() if m.get("semantic_type") == "money"]
        order_col = money_cols[0] if money_cols else None
        samples = _representative_samples(conn, t, n, order_col)
        prof_tables[t] = {
            "source": sources.get(t),
            "row_count": n,
            "columns": cols,
            "primary_key": pk[t],
            "date_columns": date_cols,
            "topics": _topics_for_table(t, list(cols)),
            "description": _describe_table(t, n, cols, date_range.get(t, {})),
            "samples": samples,
        }

    # --- quan hệ join (FK dự đoán) ---
    relationships: list[dict] = []
    for a in tables:
        for ca, ma in prof_tables[a]["columns"].items():
            if ma.get("semantic_type") != "id":
                continue
            for b in tables:
                if a == b or not pk[b]:
                    continue
                pb = pk[b][0]
                if ca != pb and not (ca in uniq_cols[b]):
                    # khác tên: chỉ xét khi cả hai đều là id và giá trị phủ gần như hoàn toàn
                    if prof_tables[b]["columns"][pb].get("semantic_type") != "id":
                        continue
                    thr = 0.98
                else:
                    thr = 0.9
                try:
                    n_a, matched = conn.execute(
                        f"SELECT COUNT(*), COUNT(b.v) FROM (SELECT DISTINCT CAST({_q(ca)} AS VARCHAR) v FROM {_q(a)} "
                        f"WHERE {_q(ca)} IS NOT NULL) x LEFT JOIN (SELECT DISTINCT CAST({_q(pb)} AS VARCHAR) v "
                        f"FROM {_q(b)}) b ON x.v = b.v").fetchone()
                except Exception:  # noqa: BLE001
                    continue
                ratio = matched / n_a if n_a else 0
                if n_a >= 2 and ratio >= thr:
                    relationships.append({
                        "from_table": a, "from_column": ca, "to_table": b, "to_column": pb,
                        "match_ratio": round(ratio, 4),
                        "cardinality": "one-to-one" if ma.get("is_unique") else "many-to-one",
                        "join": f"{a}.{ca} = {b}.{pb}",
                    })

    all_dates = [v for d in date_range.values() for r in d.values() for v in (r["min"], r["max"])]
    units = {"currency": "VND (giả định: không thấy ký hiệu tiền tệ khác)", "unit_columns": {}, "quantity_unit_suffixes": {}}
    for t, tp in prof_tables.items():
        for c, m in tp["columns"].items():
            if m.get("semantic_type") == "unit" and "top_values" in m:
                units["unit_columns"][f"{t}.{c}"] = [v["value"] for v in m["top_values"]]
            if m.get("unit_suffixes"):
                units["quantity_unit_suffixes"][f"{t}.{c}"] = m["unit_suffixes"]
    topic_index: dict[str, list[str]] = defaultdict(list)
    for t, tp in prof_tables.items():
        for topic, hits in tp["topics"].items():
            topic_index[topic].extend(f"{t}.{h}" if not h.startswith("<table:") else t for h in hits)

    summary = {
        "tables": len(tables),
        "duplicate_rows_total": sum(d["duplicate_rows_count"] for d in dirties.values()),
        "canceled_orders_total": sum(d["canceled_orders_count"] for d in dirties.values()),
        "returned_items_total": sum(d["returned_items_count"] for d in dirties.values()),
        "tables_with_format_conflicts": [t for t, d in dirties.items() if d["format_conflicts"]],
    }
    return {
        "business_id": business_id,
        "database_path": DB_FILENAME,
        "tables": prof_tables,
        "relationships": relationships,
        "date_range": {"overall": {"min": min(all_dates), "max": max(all_dates)} if all_dates else {},
                       "by_table": date_range},
        "column_inventory": inventory_cols,
        "units": units,
        "topic_index": dict(topic_index),
        "readme_notes": readme,
        "rules_summary": compact_rules(rules),
        "dirty_data_report": {"summary": summary, "tables": dirties},
        "meta": {"pipeline_version": "2.0"},
    }


def _describe_table(t: str, n: int, cols: dict, dr: dict) -> str:
    sem = defaultdict(list)
    for c, m in cols.items():
        sem[m.get("semantic_type")].append(c)
    parts = [f"{n:,} dòng"]
    for k, label in (("id", "mã"), ("name", "tên"), ("date", "ngày"), ("quantity", "số lượng"), ("money", "tiền")):
        if sem.get(k):
            parts.append(f"{label}: {', '.join(sem[k][:4])}")
    if dr:
        c = next(iter(dr))
        parts.append(f"{dr[c]['min']}→{dr[c]['max']}")
    return "; ".join(parts)


def build_schema_card(profile: dict, max_cols: int = 14, max_chars: int = 7000) -> str:
    """Schema card ngắn gọn để nhét vào prompt thay cho profile đầy đủ."""
    L = [f"# {profile['business_id']} — schema card",
         f"DB: `{profile['database_path']}` (DuckDB). Cột ngày là DATE, cột tiền/số lượng là DOUBLE, giữ nguyên tên; "
         f"giá trị gốc nằm ở cột `<tên>_raw` (không dùng để tính toán)."]
    ov = profile["date_range"].get("overall")
    if ov:
        L.append(f"Khoảng ngày dữ liệu: {ov['min']} → {ov['max']}")
    for t, tp in profile["tables"].items():
        L.append(f"\n## {t} ({tp['row_count']:,} dòng)  {tp['description']}")
        shown = [(c, m) for c, m in tp["columns"].items() if m.get("semantic_type") != "raw_text"]
        items = []
        for c, m in shown[:max_cols]:
            tag = m.get("semantic_type", "text")
            extra = ""
            if m.get("top_values"):
                extra = ": " + ", ".join(repr(v["value"]) for v in m["top_values"][:6])
            items.append(f"{c}[{tag}{extra}]")
        if len(shown) > max_cols:
            items.append(f"... +{len(shown) - max_cols} cột")
        L.append("Cột: " + ", ".join(items))
        if tp["primary_key"]:
            L.append("Khoá chính (dự đoán): " + ", ".join(tp["primary_key"]))
        for s in tp["samples"][:2]:
            short = {k: (str(v)[:30] if v is not None else None) for k, v in list(s.items())[:max_cols]
                     if not k.endswith("_raw")}
            L.append("Mẫu: " + json.dumps(short, ensure_ascii=False))
    if profile["relationships"]:
        L.append("\n## Quan hệ join")
        for r in profile["relationships"]:
            L.append(f"- {r['join']} ({r['cardinality']}, khớp {r['match_ratio']:.0%})")
    warn = []
    for t, d in profile["dirty_data_report"]["tables"].items():
        w = []
        if d["duplicate_rows_count"]:
            w.append(f"{d['duplicate_rows_count']} dòng trùng (KHÔNG bị xoá)")
        if d["canceled_orders_count"]:
            cond = "; ".join(f"{c['column']} ∈ {c['values']}" for c in d["canceled_rows_condition"])
            w.append(f"{d['canceled_orders_count']} dòng huỷ ({cond})")
        if d["returned_items_count"]:
            cond = "; ".join(f"{c['column']} ∈ {c['values']}" for c in d["returned_rows_condition"])
            w.append(f"{d['returned_items_count']} dòng trả hàng ({cond or 'số lượng âm'})")
        for fc in d["format_conflicts"][:3]:
            w.append(f"lỗi định dạng {fc['column']} ({fc['kind']})")
        if d["aliases"]:
            w.append("alias tên/mã ở: " + ", ".join(d["aliases"]))
        if d["mixed_type_columns"]:
            w.append("kiểu lẫn lộn: " + ", ".join(x["column"] for x in d["mixed_type_columns"]))
        if w:
            warn.append(f"- {t}: " + "; ".join(w))
    if warn:
        L.append("\n## Cảnh báo dữ liệu bẩn")
        L.extend(warn)
    card = "\n".join(L)
    return card if len(card) <= max_chars else card[: max_chars - 40] + "\n...[card bị cắt, xem profile.json]"
