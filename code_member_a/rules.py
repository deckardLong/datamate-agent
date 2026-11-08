from __future__ import annotations

import re

from pathlib import Path

from typing import Optional

from .utils import clean_name, fold

RULE_TOPICS: dict[str, list[str]] = {
    "valid_order": [r"hop le", r"\bvalid\b"],
    "revenue": [r"doanh thu", r"revenue", r"thanh tien", r"tong tien"],
    "cancel_return": [r"\bhuy\b", r"tra hang", r"\bhoan\b", r"\breturn", r"cancel", r"refund"],
    "best_selling": [r"ban chay", r"best.?sell", r"\btop\b"],
    "week_definition": [r"\btuan\b", r"thu hai", r"chu nhat", r"\bweek\b"],
    "this_month": [r"thang nay", r"thang truoc", r"thang rồi", r"\bmonth\b", r"tinh den"],
    "as_of_date": [r"as.?of", r"ngay tham chieu"],
    "same_period": [r"cung ky"],
    "tet_window": [r"\btet\b", r"thang chap", r"mung 10"],
    "ranking_tiebreak": [r"dong hang", r"\btie\b", r"\bhoa\b", r"xep hang", r"uu tien"],
    "rounding": [r"lam tron", r"\bround", r"chu so thap phan", r"decimal"],
    "missing_vs_zero": [r"thieu du lieu", r"khong co du lieu", r"\bnull\b", r"bang 0", r"missing"],
    "unanswerable": [r"khong the tra loi", r"khong tra loi", r"unanswerable", r"tu choi",
                     r"khong du thong tin", r"ngoai pham vi"],
}
_RULE_RE = {k: [re.compile(p) for p in v] for k, v in RULE_TOPICS.items()}


def _split_units(block: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỵ0-9\"“(])", block.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_rulebook(rulebook_path: Optional[Path]) -> dict:
    """Parse rulebook.md: lấy cả bullet, đoạn văn, bảng; gán chủ đề theo từ khoá (có bỏ dấu).

    Output:
      topics[topic] = {"rules": [{"text","section"}], "text": "..."}   # dùng trực tiếp cho prompt
      sections[section] = [các câu/ý trong mục]                         # tương thích bản cũ
      missing_topics = chủ đề chưa tìm thấy câu nào -> người phụ trách cần đọc lại rulebook
      hints = vài giá trị rút ra tự động (thứ bắt đầu tuần, số chữ số làm tròn...)
    """
    empty = {"version": 2, "topics": {}, "sections": {}, "global_rules": [], "hints": {},
             "missing_topics": list(RULE_TOPICS), "raw": ""}
    if not rulebook_path or not Path(rulebook_path).exists():
        return empty
    content = Path(rulebook_path).read_text(encoding="utf-8")

    sections: dict[str, list[str]] = {"general": []}
    titles: dict[str, str] = {"general": "general"}
    cur = "general"
    para: list[str] = []

    def flush():
        nonlocal para
        if para:
            sections.setdefault(cur, []).extend(_split_units(" ".join(para)))
            para = []

    for raw in content.splitlines():
        line = raw.strip()
        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h:
            flush()
            cur = clean_name(h[2], "section")
            titles[cur] = h[2].strip()
            sections.setdefault(cur, [])
        elif not line:
            flush()
        elif re.match(r"^[-*+]\s+", line) or re.match(r"^\d+[.)]\s+", line):
            flush()
            sections.setdefault(cur, []).extend(_split_units(re.sub(r"^([-*+]|\d+[.)])\s+", "", line)))
        elif line.startswith("|"):
            flush()
            if not re.match(r"^\|[\s:|-]+\|$", line):
                sections.setdefault(cur, []).append(" ".join(c.strip() for c in line.strip("|").split("|")))
        else:
            para.append(line)
    flush()

    topics: dict[str, dict] = {k: {"rules": [], "text": ""} for k in RULE_TOPICS}
    for sec, units in sections.items():
        sec_f = fold(titles.get(sec, sec))
        for u in units:
            uf = fold(u)
            for topic, regs in _RULE_RE.items():
                if any(r.search(uf) for r in regs) or any(r.search(sec_f) for r in regs):
                    topics[topic]["rules"].append({"text": u, "section": titles.get(sec, sec)})
    for t in topics.values():
        seen, uniq = set(), []
        for r in t["rules"]:
            if r["text"] not in seen:
                seen.add(r["text"])
                uniq.append(r)
        t["rules"] = uniq
        t["text"] = " ".join(r["text"] for r in uniq)

    folded = fold(content)
    hints: dict[str, Any] = {}
    if re.search(r"thu hai.{0,40}chu nhat|monday.{0,40}sunday", folded):
        hints["week_start"] = "monday"
        hints["week_end"] = "sunday"
    m = re.search(r"lam tron.{0,40}?(\d+)\s*chu so", folded)
    if m:
        hints["round_digits"] = int(m[1])

    return {
        "version": 2,
        "topics": topics,
        "sections": {titles.get(k, k): v for k, v in sections.items() if v},
        "global_rules": sections.get("general", []),
        "hints": hints,
        "missing_topics": [k for k, v in topics.items() if not v["rules"]],
        "raw": content,
    }


def compact_rules(rules: dict, max_per_topic: int = 12, max_chars: int = 400) -> dict:
    return {k: [r["text"][:max_chars] for r in v["rules"][:max_per_topic]]
            for k, v in rules.get("topics", {}).items() if v["rules"]}
