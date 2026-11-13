"""Dynamic session import: accept new Kaggle result zips / JSONL files, rebuild labelling_data/, sync DB."""
import gzip
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from labelling_tool import build_data, config, db


def _safe_filename(name: str) -> str:
    base = Path(name or "session_results.zip").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return cleaned or "session_results.zip"


def _inspect_and_stage_file(src_path: Path, incoming_dir: Path) -> dict:
    """Validate an incoming .zip, .jsonl, or .jsonl.gz and place into incoming_dir."""
    incoming_dir.mkdir(parents=True, exist_ok=True)
    lower = src_path.name.lower()
    rows_count = 0
    staged_files: list[str] = []

    if lower.endswith(".zip"):
        try:
            with zipfile.ZipFile(src_path, "r") as z:
                names = z.namelist()
                seg_jsons = [n for n in names if n.endswith("segments/transcripts.jsonl")]
                other_jsons = [n for n in names if n.lower().endswith(".jsonl")]
                target_names = seg_jsons or other_jsons
                if not target_names:
                    raise ValueError(f"{src_path.name}: zip contains no segments/transcripts.jsonl or .jsonl file")
                for n in target_names:
                    with z.open(n) as fh:
                        for line in fh:
                            if line.strip():
                                rows_count += 1
                if seg_jsons:
                    dst = incoming_dir / _safe_filename(src_path.name)
                    if src_path.resolve() != dst.resolve():
                        shutil.copy2(src_path, dst)
                    staged_files.append(dst.name)
                else:
                    # Extract non-standard jsonl directly so build_data reads it cleanly
                    for n in target_names:
                        out_name = _safe_filename(f"{src_path.stem}_{Path(n).name}")
                        dst = incoming_dir / out_name
                        with z.open(n) as fin, open(dst, "wb") as fout:
                            shutil.copyfileobj(fin, fout)
                        staged_files.append(dst.name)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{src_path.name}: invalid zip file ({exc})") from exc

    elif lower.endswith(".jsonl.gz"):
        stem = src_path.name[:-3]  # drops .gz -> .jsonl
        dst = incoming_dir / _safe_filename(stem)
        with gzip.open(src_path, "rb") as fin, open(dst, "wb") as fout:
            for line in fin:
                if line.strip():
                    json.loads(line.decode("utf-8"))
                    rows_count += 1
                    fout.write(line)
        staged_files.append(dst.name)

    elif lower.endswith(".jsonl"):
        dst = incoming_dir / _safe_filename(src_path.name)
        with open(src_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    json.loads(line)
                    rows_count += 1
        if src_path.resolve() != dst.resolve():
            shutil.copy2(src_path, dst)
        staged_files.append(dst.name)

    else:
        raise ValueError("Unsupported file format; expected .zip, .jsonl, or .jsonl.gz")

    return {"staged_files": staged_files, "rows_in_source": rows_count}


def _run_build_data(data_dir: Path | None = None):
    target_out = data_dir or config.DATA_DIR
    old_out = build_data.OUT
    old_argv = sys.argv
    try:
        build_data.OUT = target_out
        sys.argv = ["build_data.py"]
        build_data.main()
    finally:
        build_data.OUT = old_out
        sys.argv = old_argv


def import_uploaded_bytes(filename: str, content: bytes, db_path: Path | None = None, run_build: bool = True) -> dict:
    """Save uploaded bytes to incoming/, run build_data, and sync DB (leaving verified pieces untouched)."""
    safe_name = _safe_filename(filename)
    tmp_dir = config.INCOMING_DIR / ".tmp_upload"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / safe_name
    try:
        tmp_file.write_bytes(content)
        info = _inspect_and_stage_file(tmp_file, config.INCOMING_DIR)
    finally:
        tmp_file.unlink(missing_ok=True)
    return finalize_import(info, db_path=db_path, run_build=run_build)


def import_local_path(local_path_str: str, db_path: Path | None = None, run_build: bool = True) -> dict:
    """Import a local .zip/.jsonl/.jsonl.gz file or directory containing Kaggle results."""
    p = Path((local_path_str or "").strip().strip('"').strip("'")).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    targets: list[Path] = []
    if p.is_file():
        targets = [p]
    elif p.is_dir():
        targets = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.name.lower().endswith((".zip", ".jsonl", ".jsonl.gz"))
        )
        if not targets:
            raise ValueError(f"No .zip / .jsonl / .jsonl.gz files found in {p}")
    else:
        raise ValueError(f"Unsupported path: {p}")

    all_staged: list[str] = []
    total_rows = 0
    for t in targets:
        res = _inspect_and_stage_file(t, config.INCOMING_DIR)
        all_staged.extend(res["staged_files"])
        total_rows += res["rows_in_source"]

    return finalize_import({"staged_files": all_staged, "rows_in_source": total_rows}, db_path=db_path, run_build=run_build)


def finalize_import(stage_info: dict, db_path: Path | None = None, run_build: bool = True) -> dict:
    conn = db.connect(db_path)
    try:
        before_drafts = conn.execute("SELECT COUNT(*) FROM pieces WHERE has_draft = 1").fetchone()[0]
        before_verified = conn.execute("SELECT COUNT(*) FROM pieces WHERE status = 'verified'").fetchone()[0]
        before_calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        before_pieces = conn.execute("SELECT COUNT(*) FROM pieces").fetchone()[0]
    finally:
        conn.close()

    if run_build:
        _run_build_data(config.DATA_DIR)
    sync_res = db.sync_from_labelling_data(force=True, db_path=db_path)

    conn = db.connect(db_path)
    try:
        after_drafts = conn.execute("SELECT COUNT(*) FROM pieces WHERE has_draft = 1").fetchone()[0]
        after_verified = conn.execute("SELECT COUNT(*) FROM pieces WHERE status = 'verified'").fetchone()[0]
        after_calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
        after_pieces = conn.execute("SELECT COUNT(*) FROM pieces").fetchone()[0]
    finally:
        conn.close()

    incoming_files = sorted(
        f.name for f in config.INCOMING_DIR.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ) if config.INCOMING_DIR.is_dir() else []

    return {
        "staged_files": stage_info.get("staged_files", []),
        "rows_in_source": stage_info.get("rows_in_source", 0),
        "new_pieces_with_draft": after_drafts - before_drafts,
        "new_calls": after_calls - before_calls,
        "new_pieces": after_pieces - before_pieces,
        "total_calls": after_calls,
        "total_pieces": after_pieces,
        "pieces_with_draft": after_drafts,
        "verified_preserved": after_verified,
        "verified_before": before_verified,
        "incoming_files": incoming_files,
        "sync": sync_res,
    }
