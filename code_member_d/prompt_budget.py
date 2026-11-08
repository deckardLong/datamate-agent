"""Fit a prompt into a token budget without losing the exact schema or the question.

B's teacher prompt is assembled from fixed sections. When it is longer than the
context we allow (small GPU, OOM), cutting a slice out of the middle drops the first
tables of "Exact schema" (the most relevant ones), and the model then invents column
names. Instead, shrink the sections that matter least, in this order, and leave the
instruction, the exact schema, the query discipline and the task/question untouched.
"""

from __future__ import annotations

from typing import Callable

H_RULES = "Business rules (possibly abbreviated):\n"
H_NOTES = "Business notes (possibly abbreviated):\n"
H_NAMES = "All exact table names (never shorten them):\n"
H_SAMPLES = "Representative rows for likely relevant tables (values are examples only):\n"
H_SCHEMA = "Exact schema for likely relevant tables:\n"
H_JOINS = "Possible joins inferred from identical column names (verify semantics):\n"
END_MARKER = "Query discipline:"

# Section order in B's prompt, and which ones may be shrunk (first = shrunk first).
_ORDER = (("rules", H_RULES), ("notes", H_NOTES), ("names", H_NAMES),
          ("samples", H_SAMPLES), ("schema", H_SCHEMA), ("joins", H_JOINS))
_SACRIFICE = ("joins", "names", "samples", "rules", "notes")
_OMITTED = "(omitted to fit the context window)"


def _spans(text: str) -> list[tuple[str, int, int]] | None:
    """[(section, body_start, body_end)] in prompt order, or None if the layout differs."""
    starts = []
    cursor = 0
    for name, header in _ORDER:
        pos = text.find(header, cursor)
        if pos < 0:
            return None
        starts.append((name, pos, pos + len(header)))
        cursor = pos + len(header)
    end_marker = text.find(END_MARKER, cursor)
    if end_marker < 0:
        return None
    spans = []
    for index, (name, _, body_start) in enumerate(starts):
        next_start = starts[index + 1][1] if index + 1 < len(starts) else end_marker
        spans.append((name, body_start, max(body_start, next_start - 2)))  # minus "\n\n"
    return spans


def _keep_lines(body: str, fraction: float) -> str:
    if fraction <= 0 or not body.strip():
        return _OMITTED
    budget = len(body) * fraction
    kept, used = [], 0
    for line in body.split("\n"):
        if kept and used + len(line) > budget:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)


def fit_ids(encode: Callable[[str], list[int]], rendered: str, limit: int,
            head_fraction: float = 0.6) -> list[int]:
    """Token ids of `rendered` within `limit`: shrink low-value sections first; only if that
    is not enough, cut the middle keeping both ends (head = output-format instruction,
    tail = task and question; the same 60/40 split B used for training prompts)."""
    ids = encode(rendered)
    if len(ids) <= limit:
        return ids
    ids = encode(fit_prompt(rendered, lambda text: len(encode(text)), limit))
    if len(ids) <= limit:
        return ids
    head = round(limit * head_fraction)
    return ids[:head] + ids[-(limit - head):]


def fit_prompt(text: str, count_tokens: Callable[[str], int], limit: int) -> str:
    """Return `text` shrunk until count_tokens(text) <= limit (best effort)."""
    if count_tokens(text) <= limit:
        return text
    spans = _spans(text)
    if spans is None:
        return text
    original = {name: text[start:end] for name, start, end in spans}
    bodies = dict(original)

    def render() -> str:
        out, cursor = [], 0
        for name, start, end in spans:
            out.append(text[cursor:start])
            out.append(bodies[name])
            cursor = end
        out.append(text[cursor:])
        return "".join(out)

    for name in _SACRIFICE:
        for fraction in (0.5, 0.25, 0.0):
            bodies[name] = _keep_lines(original[name], fraction)
            candidate = render()
            if count_tokens(candidate) <= limit:
                return candidate
    return render()
