from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union

import duckdb

from .io_loaders import load_sources
from .profiler import DB_FILENAME, build_profile, build_schema_card
from .rules import parse_rulebook
from .schema import normalize_table
from .sharding import generate_shards_config

log = logging.getLogger("datamate")

PIPELINE_VERSION = "2.0"
PROFILE_FILENAME = "profile.json"
CARD_FILENAME = "schema_card.md"
PARQUET_DIRNAME = "parquet"
BUSINESS_PREFIXES = ("biz_", "business_")
EXCLUDED_BUSINESSES = {"diz_114"}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _dir_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def _file_hash(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def business_fingerprint(biz_dir: Path) -> str:
    """Hash nội dung (không phụ thuộc tên thư mục) -> phát hiện business trùng như diz_114/biz_114."""
    hs = sorted(_file_hash(f) for f in biz_dir.iterdir() if f.is_file() and f.name.lower() != "readme.md")
    return hashlib.sha1("".join(hs).encode()).hexdigest()


def find_aux_file(name: str, start: Path, levels: int = 3) -> Optional[Path]:
    cur = start.resolve()
    for _ in range(levels):
        for cand in (cur / name, cur / "train" / name):
            if cand.exists():
                return cand
        cur = cur.parent
    return None


class DataMateDataPipeline:
    def __init__(self, split: str = "train", data_dir: Union[str, Path] = "data/train",
                 cache_root: Union[str, Path] = "cache", workers: int = 4, n_shards: int = 4,
                 excluded: Optional[set[str]] = None, force: bool = False):
        self.split = split
        self.data_dir = Path(data_dir)
        self.biz_root = self.data_dir / "data" if (self.data_dir / "data").is_dir() else self.data_dir
        self.cache_split_dir = Path(cache_root) / split
        self.cache_split_dir.mkdir(parents=True, exist_ok=True)
        self.workers, self.n_shards, self.force = workers, n_shards, force
        self.excluded = set(EXCLUDED_BUSINESSES if excluded is None else excluded)
        self.rulebook_path = find_aux_file("rulebook.md", self.data_dir)
        self.calendar_path = find_aux_file("calendar.csv", self.data_dir)
        self.questions_path = find_aux_file("questions.jsonl", self.data_dir)
        self.rules = parse_rulebook(self.rulebook_path)
        if not self.rulebook_path:
            log.warning("Không thấy rulebook.md -> rules rỗng")
        if not self.calendar_path:
            log.warning("Không thấy calendar.csv -> resolve_time_range cho Tết sẽ lỗi")

    # ---- một business ----
    def process_single_business(self, biz_dir: Path) -> dict:
        bid = biz_dir.name
        final = self.cache_split_dir / bid
        sources_fp = hashlib.sha1(json.dumps(
            [PIPELINE_VERSION] + sorted((f.name, f.stat().st_size, int(f.stat().st_mtime))
                                        for f in biz_dir.iterdir() if f.is_file())).encode()).hexdigest()
        prof_path = final / PROFILE_FILENAME
        if not self.force and prof_path.exists() and (final / DB_FILENAME).exists():
            try:
                if json.loads(prof_path.read_text(encoding="utf-8")).get("meta", {}).get("source_fingerprint") == sources_fp:
                    log.info("[%s] bỏ qua (cache còn mới)", bid)
                    return {"business_id": bid, "status": "cached"}
            except Exception:  # noqa: BLE001
                pass

        tmp = self.cache_split_dir / f".{bid}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        (tmp / PARQUET_DIRNAME).mkdir(parents=True)
        conn = duckdb.connect(str(tmp / DB_FILENAME))
        try:
            metas, dirties, srcs = {}, {}, {}
            for name, raw_df, src in load_sources(biz_dir):
                df, meta, dirty = normalize_table(raw_df, name)
                date_cols = [c for c, m in meta.items() if m.get("normalized_column") == c and m.get("semantic_type") == "date"]
                conn.register("_df", df)
                repl = f" REPLACE ({', '.join(f'CAST({_q(c)} AS DATE) AS {_q(c)}' for c in date_cols)})" if date_cols else ""
                conn.execute(f"CREATE TABLE {_q(name)} AS SELECT *{repl} FROM _df")
                conn.unregister("_df")
                metas[name], dirties[name], srcs[name] = meta, dirty, src
            if not metas:
                raise RuntimeError("không nạp được bảng nào")
            for (t,) in conn.execute("SHOW TABLES").fetchall():
                conn.execute(f"COPY {_q(t)} TO '{(tmp / PARQUET_DIRNAME / (t + '.parquet')).as_posix()}' (FORMAT PARQUET)")
            readme = next((p.read_text(encoding="utf-8", errors="replace") for p in biz_dir.iterdir()
                           if p.is_file() and p.name.lower() == "readme.md"), "")
            profile = build_profile(conn, bid, metas, dirties, srcs, readme, self.rules)
            profile["meta"]["source_fingerprint"] = sources_fp
            (tmp / PROFILE_FILENAME).write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
            (tmp / CARD_FILENAME).write_text(build_schema_card(profile), encoding="utf-8")
        except Exception:
            conn.close()
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        conn.close()
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        log.info("[%s] xong: %d bảng", bid, len(metas))
        return {"business_id": bid, "status": "ok", "tables": len(metas)}

    # ---- toàn bộ split ----
    def discover(self) -> tuple[list[Path], list[dict]]:
        skipped, keep = [], []
        for d in sorted(p for p in self.biz_root.iterdir() if p.is_dir()):
            if d.name in self.excluded:
                skipped.append({"business_id": d.name, "reason": "excluded (trùng/chưa xác nhận)"})
            elif not d.name.startswith(BUSINESS_PREFIXES):
                skipped.append({"business_id": d.name, "reason": "tên thư mục không phải business"})
            else:
                keep.append(d)
        return keep, skipped

    def run(self, only: Optional[list[str]] = None) -> dict:
        keep, skipped = self.discover()
        # phát hiện business trùng nội dung trong số business hợp lệ + business bị loại
        fps: dict[str, list[str]] = defaultdict(list)
        for d in sorted(p for p in self.biz_root.iterdir() if p.is_dir() and p.name.startswith(BUSINESS_PREFIXES + ("diz_",))):
            try:
                fps[business_fingerprint(d)].append(d.name)
            except Exception:  # noqa: BLE001
                pass
        dup_groups = [v for v in fps.values() if len(v) > 1]
        for g in dup_groups:
            log.warning("Business trùng dữ liệu: %s", g)
        if only:
            keep = [d for d in keep if d.name in only]
        keep.sort(key=_dir_size)  # nhỏ trước
        log.info("%d business, %d bị bỏ qua", len(keep), len(skipped))

        results, errors = [], []
        priority, rest = keep[:5], keep[5:]
        for d in priority:  # giao sớm cho B/C
            try:
                results.append(self.process_single_business(d))
            except Exception as e:  # noqa: BLE001
                log.exception("[%s] lỗi", d.name)
                errors.append({"business_id": d.name, "error": repr(e)})
        if rest:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self.process_single_business, d): d for d in rest}
                for f in as_completed(futs):
                    d = futs[f]
                    try:
                        results.append(f.result())
                    except Exception as e:  # noqa: BLE001
                        log.error("[%s] lỗi: %r", d.name, e)
                        errors.append({"business_id": d.name, "error": repr(e)})

        # file dùng chung cho B
        if self.rulebook_path:
            shutil.copy2(self.rulebook_path, self.cache_split_dir / "rulebook.md")
        if self.calendar_path:
            shutil.copy2(self.calendar_path, self.cache_split_dir / "calendar.csv")
        (self.cache_split_dir / "rules.json").write_text(json.dumps(self.rules, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.rules.get("missing_topics"):
            log.warning("Rulebook: chưa tìm thấy nội dung cho chủ đề: %s", self.rules["missing_topics"])

        ok_ids = sorted(r["business_id"] for r in results)
        shards = None
        if not only:
            shards = generate_shards_config(self.cache_split_dir, ok_ids, self.questions_path,
                                            self.cache_split_dir / "shards.json", self.n_shards, self.excluded)
        manifest = {"pipeline_version": PIPELINE_VERSION, "split": self.split, "processed": sorted(results, key=lambda r: r["business_id"]),
                    "errors": errors, "skipped": skipped, "duplicate_business_groups": dup_groups,
                    "rulebook_found": bool(self.rulebook_path), "calendar_found": bool(self.calendar_path),
                    "questions_found": bool(self.questions_path)}
        (self.cache_split_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("HOÀN TẤT: %d ok, %d lỗi", len(results), len(errors))
        return manifest
