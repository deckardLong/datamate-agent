from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from .parsers import (
    money_style,
    parse_date_series,
    parse_number_series,
    sql_parse_expr,
)
from .utils import NULL_TOKENS, _is_textual, clean_name, fold, has_kw, make_unique

PARSE_OK_RATE = 0.8

ID_KW = {"id", "ma", "code", "sku", "barcode", "so_hd", "so_hoa_don", "so_chung_tu"}
DATE_KW = {"ngay", "date", "thoi_gian", "time", "created_at", "updated_at", "datetime", "timestamp"}
QTY_KW = {"sluong", "so_luong", "soluong", "quantity", "qty", "sl", "khoi_luong", "trong_luong",
          "ton_kho", "ton", "stock", "sl_tra", "so_luong_tra"}
MONEY_KW = {"gia", "gia_ban", "tien", "thanh_tien", "tong", "tong_tien", "doanh_thu", "chi_phi",
            "loi_nhuan", "don_gia", "price", "amount", "total", "cost", "revenue", "discount",
            "giam_gia", "chiet_khau", "thue", "vat", "cong_no", "phai_thu", "phai_tra"}
STATUS_KW = {"trang_thai", "status", "tinh_trang", "state"}
UNIT_KW = {"don_vi", "dvt", "unit", "uom"}
NAME_KW = {"ten", "name", "title"}
CANCEL_COL_KW = {"huy", "cancel", "cancelled", "canceled", "da_huy"}
RETURN_COL_KW = {"tra_hang", "hang_tra", "return", "returned", "refund", "hoan_tien", "hoan_hang",
                 "sl_tra", "so_luong_tra"}
CANCEL_RE = re.compile(r"\b(huy|cancel(l?ed)?|void)\b")
RETURN_RE = re.compile(r"\b(tra hang|tra lai|hoan hang|hoan tien|hoan tra|return(ed)?|refund(ed)?)\b")
TRUTHY = {"1", "true", "yes", "y", "co", "x", "t", "1.0"}

TOPIC_KW = {
    "cancel_return": CANCEL_COL_KW | RETURN_COL_KW | {"hoan", "tra_lai"},
    "inventory": {"ton_kho", "kho", "stock", "inventory", "nhap_kho", "xuat_kho", "ton"},
    "debt": {"cong_no", "no", "phai_thu", "phai_tra", "debt", "receivable", "payable"},
    "sales": {"don_hang", "hoa_don", "order", "orders", "invoice", "ban_hang", "doanh_thu", "sales"},
    "customer": {"khach_hang", "customer", "kh"},
    "product": {"san_pham", "hang_hoa", "product", "sku", "mat_hang"},
}


# Tên cột viết liền không dấu gạch (tongtien, giabanle, dongianhap, soluongton...): so khớp chuỗi con
# trên tên đã bỏ "_", chỉ dùng các từ ghép đủ dài để tránh khớp nhầm (vd 'gia' trong 'giaohang').
MONEY_COMPACT = ("tongtien", "thanhtien", "dongia", "giaban", "giavon", "gianhap", "giadonvi", "giale",
                 "giasi", "giasanpham", "gianiemyet", "doanhthu", "tienhang", "tientra", "trigia",
                 "chietkhau", "giamgia", "congno", "phaithu", "phaitra", "loinhuan", "chiphi", "tongbill")
QTY_COMPACT = ("soluong", "sluong", "tonkho", "khoiluong", "trongluong")
DATE_COMPACT = ("ngaytao", "ngayban", "ngaydat", "ngaygiao", "ngaythanhtoan", "ngaychungtu", "ngayhoadon",
                "thoigian", "ngaynhap", "ngayxuat")
ID_COMPACT = re.compile(r"^ma(hd|hoadon|sp|sanpham|kh|khachhang|don|donhang|nv|nhanvien|hang|vach|giaodich|ct)")
# Cột "tổng điểm", "điểm tích lũy"... là điểm thưởng, không phải tiền dù có chữ 'tong'
NOT_MONEY_KW = {"diem", "point", "points", "ty_le", "phan_tram", "percent", "pct"}


def classify_column(col: str) -> str:
    compact = col.replace("_", "")
    if has_kw(col, ID_KW) or ID_COMPACT.match(compact):
        return "id"
    if has_kw(col, DATE_KW) or any(k in compact for k in DATE_COMPACT):
        return "date"
    if has_kw(col, QTY_KW) or any(k in compact for k in QTY_COMPACT):
        return "quantity"
    if has_kw(col, NOT_MONEY_KW):
        return "other"
    if has_kw(col, MONEY_KW) or any(k in compact for k in MONEY_COMPACT):
        return "money"
    if has_kw(col, STATUS_KW):
        return "status"
    if has_kw(col, UNIT_KW):
        return "unit"
    if has_kw(col, NAME_KW):
        return "name"
    return "other"


def _clean_text(s: pd.Series) -> pd.Series:
    def f(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        t = str(v).strip()
        return None if t.lower() in NULL_TOKENS else t
    return pd.Series([f(v) for v in s], index=s.index, dtype=object)


def _infer_generic(s: pd.Series) -> pd.Series:
    """Ép số cho cột chưa rõ ngữ nghĩa nếu 100% giá trị là số và không có số 0 đứng đầu (mã)."""
    nn = s.dropna().astype(str)
    if nn.empty:
        return s
    if nn.str.match(r"^-?\d+$").all() and not nn.str.match(r"^-?0\d+").any():
        v = pd.to_numeric(nn)
        if v.abs().max() < 9e18:
            return pd.to_numeric(s, errors="coerce").astype("Int64")
    elif nn.str.match(r"^-?\d+\.\d+$").all() and not nn.str.match(r"^-?0\d+").any():
        return pd.to_numeric(s, errors="coerce")
    return s


def _flag_mask(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0) > 0
    f = s.map(lambda v: fold(v) if pd.notna(v) else "")
    return f.isin(TRUTHY) | f.map(lambda x: bool(CANCEL_RE.search(x) or RETURN_RE.search(x)))


def _alias_groups(s: pd.Series, limit: int = 200_000) -> dict:
    nn = s.dropna().astype(str)
    uniq = nn.unique()
    if len(uniq) > limit:
        return {}
    groups: dict[str, set] = defaultdict(set)
    for u in uniq:
        key = re.sub(r"[^a-z0-9]", "", fold(u))
        if key:
            groups[key].add(u)
    multi = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    if not multi:
        return {}
    return {"alias_group_count": len(multi), "examples": list(multi.values())[:5]}


# Dòng tổng / tổng phụ lẫn trong dữ liệu ("Cộng", "Tổng ngày 01/04", "Tổng cộng", "Total").
# So khớp CÓ DẤU để không nhầm với họ "Tống" (Tống Khánh Quang); bản không dấu chỉ nhận cụm rõ nghĩa.
SUMMARY_ACCENTED_RE = re.compile(r"^\s*(tổng|cộng|lũy kế|luỹ kế)(\s|:|$)", re.I)
SUMMARY_FOLDED_RE = re.compile(r"^\s*(tong cong|tong ngay|tong thang|tong tuan|tong quy|tong nam|tong ca|"
                               r"cong don|luy ke|total|sub ?total|grand total|sum)(\s|:|$)")


def find_summary_rows(df: pd.DataFrame) -> pd.Series:
    """Mask các dòng tổng: có nhãn tổng VÀ (không có mã định danh HOẶC trống quá nửa số cột)."""
    mask = pd.Series(False, index=df.index)
    if df.empty:
        return mask
    text_cols = [c for c in df.columns if _is_textual(df[c])]
    if not text_cols:
        return mask
    label = pd.Series(False, index=df.index)
    for c in text_cols:
        s = df[c].astype("string").fillna("").str.strip()
        label |= s.str.match(SUMMARY_ACCENTED_RE) | s.map(lambda v: bool(SUMMARY_FOLDED_RE.match(fold(v))) if v else False)
    if not label.any():
        return mask
    id_cols = [c for c in df.columns if classify_column(c) == "id"]
    no_id = df[id_cols].isna().all(axis=1) if id_cols else pd.Series(True, index=df.index)
    sparse = df.notna().sum(axis=1) <= max(2, df.shape[1] // 2)
    return label & (no_id & (sparse | bool(id_cols)))


def normalize_table(df: pd.DataFrame, table: str) -> tuple[pd.DataFrame, dict, dict]:
    """Làm sạch + chuẩn hoá ngày/tiền/số lượng, đồng thời tính báo cáo dữ liệu bẩn trên TOÀN BỘ bảng."""
    df = df.copy()
    df.columns = make_unique([clean_name(c) for c in df.columns])
    n_raw = len(df)
    df = df.dropna(how="all")
    summary = find_summary_rows(df)
    summary_report = {"count": int(summary.sum()),
                      "examples": [{k: str(v) for k, v in r.items() if pd.notna(v)}
                                   for r in df[summary].head(5).astype(object).to_dict("records")]}
    df = df[~summary]
    dirty: dict[str, Any] = {
        "row_count_raw": n_raw,
        "empty_rows_dropped": n_raw - len(df) - summary_report["count"],
        "summary_rows_removed": summary_report,
        "duplicate_rows_count": 0,
        "format_conflicts": [],
        "mixed_type_columns": [],
        "null_counts": {},
        "canceled_orders_count": 0,
        "returned_items_count": 0,
        "canceled_rows_condition": [],
        "returned_rows_condition": [],
        "aliases": {},
        "invalid_values": {},
        "outliers": {},
        "date_ranges": {},
    }
    # Dòng trùng: tính trên dữ liệu thô (đã stringify), KHÔNG xoá.
    try:
        dirty["duplicate_rows_count"] = int(df.astype(str).duplicated().sum())
    except Exception:  # noqa: BLE001
        pass

    out: dict[str, pd.Series] = {}
    meta: dict[str, dict] = {}
    cancel_mask = pd.Series(False, index=df.index)
    return_mask = pd.Series(False, index=df.index)

    for col in df.columns:
        s = df[col]
        if _is_textual(s):
            s = _clean_text(s)
        nn = int(s.notna().sum())
        sem = classify_column(col)
        m: dict[str, Any] = {"semantic_type": sem if sem != "other" else "text"}
        textual = _is_textual(s)
        if nn == 0:
            out[col] = s
            meta[col] = {"semantic_type": "empty"}
            continue

        if sem == "date":
            if pd.api.types.is_datetime64_any_dtype(s):
                out[col] = s.dt.normalize()
                m.update(raw_format="native", normalized_column=col)
            else:
                parsed, fmts = parse_date_series(s)
                rate = parsed.notna().sum() / nn
                if rate >= PARSE_OK_RATE:
                    fmt = fmts.most_common(1)[0][0] if fmts else "unknown"
                    out[col] = parsed
                    out[f"{col}_raw"] = s
                    m.update(raw_format=fmt, raw_column=f"{col}_raw", normalized_column=col,
                             parse_rate=round(float(rate), 4),
                             sql_parse_expr=sql_parse_expr("date", fmt, f"{col}_raw"))
                    if len(fmts) > 1:
                        m["raw_format_mix"] = dict(fmts)
                    bad = s[parsed.isna() & s.notna()]
                    if len(bad):
                        dirty["format_conflicts"].append(
                            {"column": col, "kind": "unparsable_date", "count": int(len(bad)),
                             "examples": [str(x) for x in bad.unique()[:5]]})
                    yrs = parsed.dt.year.dropna()
                    bad_year = int(((yrs < 2000) | (yrs > 2100)).sum())
                    if bad_year:
                        dirty["invalid_values"][col] = {"year_out_of_range": bad_year}
                else:
                    out[col] = s
                    m.update(semantic_type="text", semantic_hint="date", parse_rate=round(float(rate), 4))
                    dirty["format_conflicts"].append(
                        {"column": col, "kind": "date_column_not_normalized", "parse_rate": round(float(rate), 4)})
                    meta[col] = m
                    continue
            rng = out[col].dropna()
            if len(rng):
                dirty["date_ranges"][col] = {"min": str(rng.min().date()), "max": str(rng.max().date())}

        elif sem in ("money", "quantity"):
            if textual:
                num = parse_number_series(s, money=(sem == "money"))
                rate = num.notna().sum() / nn
                if rate >= PARSE_OK_RATE:
                    style = money_style(s)
                    out[col] = num
                    out[f"{col}_raw"] = s
                    m.update(raw_format=style, raw_column=f"{col}_raw", normalized_column=col,
                             parse_rate=round(float(rate), 4),
                             sql_parse_expr=sql_parse_expr(sem, style, f"{col}_raw"))
                    bad = s[num.isna() & s.notna()]
                    if len(bad):
                        dirty["format_conflicts"].append(
                            {"column": col, "kind": f"unparsable_{sem}", "count": int(len(bad)),
                             "examples": [str(x) for x in bad.unique()[:5]]})
                    if sem == "quantity":
                        suf = s.dropna().astype(str).str.extract(r"\d\s*([^\d\s.,]+)\s*$", expand=False).dropna()
                        if len(suf):
                            m["unit_suffixes"] = dict(Counter(suf.str.lower()).most_common(8))
                else:
                    out[col] = s
                    m.update(semantic_type="text", semantic_hint=sem, parse_rate=round(float(rate), 4))
                    dirty["format_conflicts"].append(
                        {"column": col, "kind": f"{sem}_column_not_normalized", "parse_rate": round(float(rate), 4),
                         "examples": [str(x) for x in s.dropna().unique()[:5]]})
                    meta[col] = m
                    continue
            else:
                out[col] = pd.to_numeric(s, errors="coerce")
                m.update(raw_format="native", normalized_column=col)
            v = out[col].dropna()
            if len(v):
                inv = {"negative": int((v < 0).sum())}
                if sem == "money":
                    inv["zero"] = int((v == 0).sum())
                dirty["invalid_values"][col] = {k: c for k, c in inv.items() if c}
                if not dirty["invalid_values"][col]:
                    del dirty["invalid_values"][col]
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    n_out = int(((v > q3 + 3 * iqr) | (v < q1 - 3 * iqr)).sum())
                    if n_out:
                        dirty["outliers"][col] = {"count": n_out, "min": float(v.min()), "max": float(v.max()),
                                                  "q1": float(q1), "q3": float(q3)}
                m["min"], m["max"] = float(v.min()), float(v.max())
                if sem == "quantity" and (v < 0).any():
                    return_mask |= out[col].fillna(0) < 0
            if sem == "money" and has_kw(col, {"huy", "tra", "hoan"}):
                pass

        else:
            out[col] = s if sem == "id" or not textual else _infer_generic(s)
            if textual:
                nnv = s.dropna().astype(str)
                numeric_like = nnv.str.match(r"^-?[\d.,]+$").mean()
                if 0.05 < numeric_like < 0.95 and sem not in ("id", "name", "status", "unit"):
                    dirty["mixed_type_columns"].append(
                        {"column": col, "numeric_like_ratio": round(float(numeric_like), 3),
                         "examples": [str(x) for x in nnv[~nnv.str.match(r"^-?[\d.,]+$")].unique()[:3]]})

        # --- trạng thái / cờ huỷ / trả hàng ---
        if has_kw(col, STATUS_KW) and textual:
            f = out[col].map(lambda v: fold(v) if pd.notna(v) else "")
            c_m, r_m = f.map(lambda x: bool(CANCEL_RE.search(x))), f.map(lambda x: bool(RETURN_RE.search(x)))
            if c_m.any():
                cancel_mask |= c_m
                vals = sorted(set(out[col][c_m].dropna().astype(str)))
                dirty["canceled_rows_condition"].append({"column": col, "values": vals})
            if r_m.any():
                return_mask |= r_m
                vals = sorted(set(out[col][r_m].dropna().astype(str)))
                dirty["returned_rows_condition"].append({"column": col, "values": vals})
        elif has_kw(col, CANCEL_COL_KW):
            fm = _flag_mask(out[col])
            if fm.any():
                cancel_mask |= fm
                dirty["canceled_rows_condition"].append({"column": col, "values": "truthy"})
        elif has_kw(col, RETURN_COL_KW):
            fm = _flag_mask(out[col])
            if fm.any():
                return_mask |= fm
                dirty["returned_rows_condition"].append({"column": col, "values": "truthy / >0"})

        # --- alias tên / mã ---
        if sem in ("name", "id") and textual:
            al = _alias_groups(out[col])
            if al:
                dirty["aliases"][col] = al

        meta[col] = m

    dirty["canceled_orders_count"] = int(cancel_mask.sum())
    dirty["returned_items_count"] = int(return_mask.sum())
    out_df = pd.DataFrame(out, index=df.index).reset_index(drop=True)
    # cột raw: ép về str để DuckDB nhận diện kiểu ổn định
    for c in out_df.columns:
        if out_df[c].dtype == object:
            out_df[c] = out_df[c].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    nulls = out_df.isna().sum()
    dirty["null_counts"] = {c: {"count": int(n), "ratio": round(float(n) / max(len(out_df), 1), 4)}
                            for c, n in nulls.items() if n > 0}
    dirty["row_count"] = len(out_df)
    return out_df, meta, dirty
