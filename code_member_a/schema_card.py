"""Module schema card."""
from __future__ import annotations

import json


def build_schema_card(profile: dict, max_cols: int = 14, max_chars: int = 7000) -> str:
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
