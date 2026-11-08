"""Run manifest: what exactly produced a submission file (plan: "ghi commit hash, model, prompt").

run_all.py writes `<out>.manifest.json` next to the results. Every (re)start of the same --out
appends a record, so a resumed run is visible, and a changed configuration is reported
because rows already written by the old configuration are kept (they would silently mix).
No `git` binary is needed: the commit is read straight from .git.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "vllm", "duckdb",
            "pandas", "numpy", "huggingface_hub", "safetensors", "tokenizers")

# Settings that change the answers; a difference between two runs on one --out is a warning.
CONFIG_KEYS = ("engine", "model_id", "adapter_sha256", "max_input_tokens", "fallback_input_tokens",
               "rules_chars", "max_attempts", "unanswerable", "allow_suspicious",
               "vllm_dtype", "vllm_quantization", "batch_size", "prompt_hash")


def read_git_head(repo_root: Path) -> dict[str, str | None]:
    """{"branch", "commit"} read from .git without running git; Nones if unavailable."""
    git = Path(repo_root) / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return {"branch": None, "commit": None}
    if not head.startswith("ref:"):
        return {"branch": None, "commit": head}  # detached HEAD
    ref = head.split(" ", 1)[1].strip()
    branch = ref.removeprefix("refs/heads/")
    loose = git / ref
    if loose.is_file():
        return {"branch": branch, "commit": loose.read_text(encoding="utf-8").strip()}
    packed = git / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return {"branch": branch, "commit": line.split(" ", 1)[0]}
    return {"branch": branch, "commit": None}


def sha256_file(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def prompt_hash() -> str:
    """Short hash of the code that builds the prompt (ours and B's), i.e. the prompt version."""
    from code_member_b import pseudo_label as b
    from code_member_d import inference, prompt_budget

    parts = [inference.build_prompt, inference.render_rules, prompt_budget.fit_prompt,
             prompt_budget._spans, b.build_teacher_prompt, b._compact_schema, b._compact_samples,
             b._compact_relationships, b._relevant_table_names, b._retry_teacher_prompt]
    source = "\n".join(inspect.getsource(part) for part in parts)
    source += repr((prompt_budget._ORDER, prompt_budget._SACRIFICE, inference._TOPIC_ORDER))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def _gpu_name() -> str | None:
    torch = sys.modules.get("torch")
    try:
        return torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - informational only
        return None


def build_run_record(args: dict[str, Any], repo_root: Path, questions: Path,
                     cache_root: Path, split: str) -> dict[str, Any]:
    adapter = Path(args["adapter"]) / "adapter_model.safetensors" if args.get("adapter") else None
    adapter_cfg: dict[str, Any] = {}
    if args.get("adapter") and (Path(args["adapter"]) / "adapter_config.json").is_file():
        adapter_cfg = json.loads((Path(args["adapter"]) / "adapter_config.json").read_text(encoding="utf-8"))
    rules = Path(cache_root) / split / "rules.json"
    record: dict[str, Any] = {
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": read_git_head(repo_root),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args.items()},
        "model_id": args.get("model_id"),
        "adapter": args.get("adapter"),
        "adapter_sha256": sha256_file(adapter),
        "adapter_peft_version": adapter_cfg.get("peft_version"),
        "prompt_hash": prompt_hash(),
        "questions_sha256": sha256_file(questions),
        "rules_json_sha256": sha256_file(rules),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": _gpu_name(),
        "packages": package_versions(),
    }
    for key in CONFIG_KEYS:
        record.setdefault(key, args.get(key))
    return record


def config_of(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in CONFIG_KEYS}


def append_run(manifest_path: Path, record: dict[str, Any]) -> list[str]:
    """Add a run to the manifest and return warnings about changes vs the previous run."""
    manifest_path = Path(manifest_path)
    runs: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            runs = json.loads(manifest_path.read_text(encoding="utf-8")).get("runs", [])
        except (OSError, ValueError):
            runs = []
    warnings: list[str] = []
    if runs:
        before, now = config_of(runs[-1]), config_of(record)
        changed = [k for k in CONFIG_KEYS if before.get(k) != now.get(k)]
        if changed:
            detail = ", ".join(f"{k}: {before.get(k)!r} -> {now.get(k)!r}" for k in changed)
            warnings.append(
                "configuration differs from the previous run on this --out (" + detail + "). "
                "Rows already written keep the OLD configuration; use a new --out/-OutDir to avoid mixing."
            )
        old_commit, new_commit = runs[-1].get("git", {}).get("commit"), record.get("git", {}).get("commit")
        if old_commit and new_commit and old_commit != new_commit:
            warnings.append(f"git commit changed since the previous run ({old_commit[:8]} -> {new_commit[:8]}).")
    runs.append(record)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    return warnings


def finish_run(manifest_path: Path, summary: dict[str, Any]) -> None:
    """Record how the latest run ended (answer counts etc.)."""
    manifest_path = Path(manifest_path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["runs"][-1].update(summary)
        data["runs"][-1]["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, KeyError, IndexError):
        pass  # the manifest is documentation; never fail a finished run because of it
