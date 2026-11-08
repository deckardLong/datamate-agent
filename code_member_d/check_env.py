"""Pre-flight check before a long local run: GPU, libraries, files. Exit 0 = ready.

    python -m code_member_d.check_env --adapter "D:\\path\\final_adapter"
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

PROBLEMS: list[str] = []
WARNINGS: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    PROBLEMS.append(msg)
    print(f"  [FAIL] {msg}")


def version(name: str) -> str | None:
    try:
        return str(getattr(importlib.import_module(name), "__version__", "?"))
    except Exception as exc:  # noqa: BLE001 - report any import problem
        return None if isinstance(exc, ImportError) else f"import error: {exc}"


def _numbers(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_peft_version(adapter_config: Path) -> None:
    """An adapter saved by a newer PEFT can contain config fields an older PEFT rejects."""
    import json

    try:
        saved = str(json.loads(adapter_config.read_text(encoding="utf-8")).get("peft_version") or "")
    except (OSError, ValueError):
        return
    installed = version("peft")
    if saved and installed and installed[0].isdigit():
        if _numbers(installed) < _numbers(saved):
            fail(f"peft {installed} is older than the {saved} that saved this adapter: "
                 "run  pip install -U peft")
        else:
            ok(f"peft {installed} >= adapter's {saved}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--cache-root", type=Path, default=Path("cache"))
    p.add_argument("--questions", type=Path, default=Path("public_test/questions.jsonl"))
    p.add_argument("--min-free-vram-gb", type=float, default=6.5)
    a = p.parse_args(argv)

    print("Python / libraries")
    py = sys.version_info
    if (py.major, py.minor) >= (3, 14):
        warn(f"Python {py.major}.{py.minor}: PyTorch/bitsandbytes wheels may not exist yet; use 3.11 or 3.12")
    else:
        ok(f"Python {py.major}.{py.minor}.{py.micro}")
    for lib in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "duckdb"):
        v = version(lib)
        if v is None:
            fail(f"{lib} is not installed")
        elif v.startswith("import error"):
            fail(f"{lib}: {v}")
        else:
            ok(f"{lib} {v}")
    tf = version("transformers")
    if tf and tf[0].isdigit():
        major, minor = (int(x) for x in tf.split(".")[:2])
        if (major, minor) < (4, 56) or major >= 5:
            fail(f"transformers {tf}: need >=4.56,<5 (the code passes dtype=)")

    print("GPU")
    try:
        import torch

        if not torch.cuda.is_available():
            fail("torch.cuda.is_available() is False (CPU-only torch build, or NVIDIA driver too old)")
        else:
            free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
            ok(f"{torch.cuda.get_device_name(0)}: {free:.1f} GB free of {total:.1f} GB (CUDA {torch.version.cuda})")
            if free < a.min_free_vram_gb:
                fail(f"only {free:.1f} GB VRAM free (< {a.min_free_vram_gb}); close browsers/games using the GPU")
            if total < 10:
                warn("<10 GB GPU: use --max-input-tokens 3072 --fallback-input-tokens 2048")
    except ImportError:
        pass

    print("Files")
    if a.questions.is_file():
        ok(f"questions: {a.questions}")
    else:
        fail(f"questions file not found: {a.questions}")
    split_dir = a.cache_root / "public_test"
    if (split_dir / "rules.json").is_file() and any(split_dir.glob("*/db.duckdb")):
        ok(f"cache: {len(list(split_dir.glob('*/db.duckdb')))} businesses in {split_dir}")
    else:
        fail(f"cache not found in {split_dir} (run code_member_a.main first)")
    if a.adapter is None:
        warn("no --adapter given: the base model will run without B's LoRA adapter")
    elif (a.adapter / "adapter_config.json").is_file() and (a.adapter / "adapter_model.safetensors").is_file():
        ok(f"adapter: {a.adapter}")
        check_peft_version(a.adapter / "adapter_config.json")
    else:
        fail(f"adapter_config.json / adapter_model.safetensors missing in {a.adapter}")

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cached = list((hf_home / "hub").glob("models--Qwen--Qwen2.5-Coder-7B-Instruct")) if (hf_home / "hub").is_dir() else []
    if cached:
        ok(f"Qwen2.5-Coder-7B-Instruct already in HF cache ({hf_home})")
    else:
        warn(f"Qwen model not cached under {hf_home}: first run downloads ~15 GB (needs internet)")
    free_disk = None
    try:
        import shutil

        free_disk = shutil.disk_usage(hf_home if hf_home.exists() else Path.home()).free / 2**30
        if free_disk < 25 and not cached:
            fail(f"only {free_disk:.0f} GB free on the HF cache drive; need ~20 GB for the model")
    except OSError:
        pass

    print()
    if PROBLEMS:
        print(f"NOT READY: {len(PROBLEMS)} problem(s)")
        return 1
    print("READY" + (f" ({len(WARNINGS)} warning(s))" if WARNINGS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
