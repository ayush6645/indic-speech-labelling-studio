"""Live exporter that maintains labelling_data/verified/ in exact data/final/dataset.jsonl schema."""
import json
import os
import shutil
import sqlite3
import threading
from pathlib import Path

from labelling_tool import config, db

_EXPORT_LOCK = threading.RLock()


def _lang_dir_name(lang_code: str) -> str:
    c = (lang_code or "").strip().lower()
    if not c:
        return "unknown"
    return config.LANG_FULL.get(c, c)


def _known_lang_dirs() -> list[str]:
    dirs = list(config.LANG_FULL.values()) + ["unknown", "none"]
    for sub in ("audio", "segments", "calls"):
        base = config.VERIFIED_DIR / sub
        if base.is_dir():
            for d in base.iterdir():
                if d.is_dir() and d.name not in dirs:
                    dirs.append(d.name)
    return dirs


def _link_or_copy(src: Path, dst: Path):
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.stat(dst).st_ino == os.stat(src).st_ino and os.stat(dst).st_dev == os.stat(src).st_dev:
                return True
        except OSError:
            pass
        dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        tmp = dst.with_suffix(".tmp" + dst.suffix)
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    return True


def _write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _remove_piece_files(piece_id: str, keep_lang_dir: str | None = None):
    for lang_dir in _known_lang_dirs():
        if keep_lang_dir and lang_dir == keep_lang_dir:
            continue
        seg_json = config.VERIFIED_DIR / "segments" / lang_dir / f"{piece_id}.json"
        seg_json.unlink(missing_ok=True)
        aud_dir = config.VERIFIED_DIR / "audio" / lang_dir
        if aud_dir.is_dir():
            for f in aud_dir.glob(f"{piece_id}.*"):
                f.unlink(missing_ok=True)


def _remove_call_files(call_id: str, keep_lang_dir: str | None = None):
    for lang_dir in _known_lang_dirs():
        if keep_lang_dir and lang_dir == keep_lang_dir:
            continue
        (config.VERIFIED_DIR / "calls" / lang_dir / f"{call_id}.json").unlink(missing_ok=True)


def _piece_src_audio(piece_row: dict) -> Path | None:
    rel = piece_row.get("edited_audio_rel") or piece_row.get("audio_rel") or ""
    if not rel:
        return None
    p = (config.DATA_DIR / rel).resolve()
    return p if p.is_file() else None


def _build_piece_record(p: dict, call: dict, split_val: str, audio_rel_verified: str) -> dict:
    lang_full = _lang_dir_name(p.get("main_language") or call.get("language") or "")
    also_codes = db.parse_also_langs(p.get("also_languages"), p.get("main_language") or "")
    also_full = [_lang_dir_name(c) for c in also_codes]
    src_call_rel = call.get("audio_rel") or ""
    return {
        "segment_id": p["piece_id"],
        "audio_filepath": audio_rel_verified,
        "duration": round(float(p["duration_s"]), 3),
        "language": lang_full,
        "region": call.get("state", ""),
        "split": split_val,
        "text_native": p.get("current_text", ""),
        "text_romanized": p.get("current_text_roman", ""),
        "draft_text_native": p.get("draft_text", ""),
        "draft_text_romanized": p.get("draft_text_roman", ""),
        "draft_language": _lang_dir_name(p["draft_language"]) if p.get("draft_language") else "",
        "human_edited": bool(p.get("human_edited")),
        "also_languages": also_full,
        "verified_by": p.get("verified_by", ""),
        "verified_at": p.get("verified_at", ""),
        "draft_by": p.get("draft_by", ""),
        "lead_id": call.get("lead_id", ""),
        "call_source": call.get("source", ""),
        "edges_edited": bool(p.get("edges_edited")),
        "source": {
            "parent_call_id": p["call_id"],
            "source_filepath": src_call_rel,
            "original_filename": Path(src_call_rel).name if src_call_rel else "",
            "segment_index": int(p["piece_index"]),
            "start_time_s": round(float(p["start_s"]), 3),
            "end_time_s": round(float(p["end_s"]), 3),
        },
    }


def _rewrite_dataset_jsonl(conn: sqlite3.Connection):
    config.VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    rows = conn.execute("""
        SELECT p.*, c.state, c.language AS call_lang, c.lead_id, c.audio_rel AS call_audio_rel,
               c.source AS call_source, COALESCE(sp.split, '') AS assigned_split
        FROM pieces p
        JOIN calls c ON c.call_id = p.call_id
        LEFT JOIN splits sp ON sp.call_id = p.call_id
        WHERE p.status = 'verified'
        ORDER BY p.call_id, p.piece_index
    """).fetchall()
    out_path = config.VERIFIED_DIR / "dataset.jsonl"
    tmp = config.VERIFIED_DIR / "dataset.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            p = dict(r)
            lang_code = p.get("main_language") or p.get("call_lang") or ""
            lang_full = _lang_dir_name(lang_code)
            src_aud = _piece_src_audio(p)
            ext = src_aud.suffix if src_aud else Path(p["audio_rel"]).suffix or ".flac"
            aud_rel = f"audio/{lang_full}/{p['piece_id']}{ext}"
            call_dict = {
                "state": p["state"],
                "language": p["call_lang"],
                "lead_id": p["lead_id"],
                "audio_rel": p["call_audio_rel"],
                "source": p["call_source"],
            }
            rec = _build_piece_record(p, call_dict, p["assigned_split"] or "", aud_rel)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def _export_call_file(conn: sqlite3.Connection, call_id: str):
    call_row = conn.execute("""
        SELECT c.*, COALESCE(sp.split, '') AS assigned_split
        FROM calls c
        LEFT JOIN splits sp ON sp.call_id = c.call_id
        WHERE c.call_id = ?
    """, (call_id,)).fetchone()
    if not call_row:
        _remove_call_files(call_id)
        return
    call = dict(call_row)
    pieces = conn.execute(
        "SELECT * FROM pieces WHERE call_id = ? ORDER BY piece_index",
        (call_id,),
    ).fetchall()
    verified_pieces = [dict(p) for p in pieces if p["status"] == "verified"]
    if not verified_pieces:
        _remove_call_files(call_id)
        return

    call_lang_code = call.get("language") or verified_pieces[0].get("main_language") or ""
    lang_full = _lang_dir_name(call_lang_code)
    _remove_call_files(call_id, keep_lang_dir=lang_full)

    split_val = call.get("assigned_split") or ""
    segs_out = []
    for p in verified_pieces:
        seg_lang_full = _lang_dir_name(p.get("main_language") or call_lang_code)
        src_aud = _piece_src_audio(p)
        ext = src_aud.suffix if src_aud else Path(p["audio_rel"]).suffix or ".flac"
        aud_rel = f"audio/{seg_lang_full}/{p['piece_id']}{ext}"
        also_codes = db.parse_also_langs(p.get("also_languages"), p.get("main_language") or "")
        segs_out.append({
            "segment_id": p["piece_id"],
            "segment_index": int(p["piece_index"]),
            "start_time_s": round(float(p["start_s"]), 3),
            "end_time_s": round(float(p["end_s"]), 3),
            "duration": round(float(p["duration_s"]), 3),
            "audio_filepath": aud_rel,
            "language": seg_lang_full,
            "also_languages": [_lang_dir_name(c) for c in also_codes],
            "text_native": p.get("current_text", ""),
            "text_romanized": p.get("current_text_roman", ""),
            "draft_text_native": p.get("draft_text", ""),
            "draft_text_romanized": p.get("draft_text_roman", ""),
            "draft_language": _lang_dir_name(p["draft_language"]) if p.get("draft_language") else "",
            "draft_by": p.get("draft_by", ""),
            "human_edited": bool(p.get("human_edited")),
            "verified_by": p.get("verified_by", ""),
            "verified_at": p.get("verified_at", ""),
            "edges_edited": bool(p.get("edges_edited")),
        })

    complete = len(pieces) > 0 and len(verified_pieces) == len(pieces)
    also_call = [_lang_dir_name(c) for c in db.parse_also_langs(call.get("also_languages"), call_lang_code)]
    call_json = {
        "call_id": call_id,
        "source": call.get("source", ""),
        "state": call.get("state", ""),
        "region": call.get("state", ""),
        "lead_id": call.get("lead_id", ""),
        "crm_call_id": call.get("crm_call_id", ""),
        "audio_filepath": call.get("audio_rel", ""),
        "language": lang_full,
        "mixed_with": also_call or None,
        "split": split_val,
        "duration_s": round(float(call.get("duration_s") or 0), 3),
        "md5": call.get("md5", ""),
        "segment_count": len(pieces),
        "verified_count": len(verified_pieces),
        "complete": complete,
        "segments": segs_out,
        "text_native": " ".join(s["text_native"].strip() for s in segs_out if s["text_native"].strip()),
        "text_romanized": " ".join(s["text_romanized"].strip() for s in segs_out if s["text_romanized"].strip()),
    }
    _write_json(config.VERIFIED_DIR / "calls" / lang_full / f"{call_id}.json", call_json)


def sync_piece(piece_id: str, db_path: Path | None = None, rewrite_jsonl: bool = True):
    """Synchronize a single piece (verified -> export/update; unverified -> remove) and rewrite dataset.jsonl."""
    with _EXPORT_LOCK:
        conn = db.connect(db_path)
        try:
            row = conn.execute("""
                SELECT p.*, c.state, c.language AS call_lang, c.lead_id, c.audio_rel AS call_audio_rel,
                       c.source AS call_source, COALESCE(sp.split, '') AS assigned_split
                FROM pieces p
                JOIN calls c ON c.call_id = p.call_id
                LEFT JOIN splits sp ON sp.call_id = p.call_id
                WHERE p.piece_id = ?
            """, (piece_id,)).fetchone()
            if not row:
                _remove_piece_files(piece_id)
                if rewrite_jsonl:
                    _rewrite_dataset_jsonl(conn)
                return

            p = dict(row)
            call_id = p["call_id"]
            if p["status"] != "verified":
                _remove_piece_files(piece_id)
                _export_call_file(conn, call_id)
                if rewrite_jsonl:
                    _rewrite_dataset_jsonl(conn)
                return

            lang_code = p.get("main_language") or p.get("call_lang") or ""
            lang_full = _lang_dir_name(lang_code)
            _remove_piece_files(piece_id, keep_lang_dir=lang_full)

            src_aud = _piece_src_audio(p)
            ext = src_aud.suffix if src_aud else Path(p["audio_rel"]).suffix or ".flac"
            aud_rel = f"audio/{lang_full}/{piece_id}{ext}"
            dst_aud = config.VERIFIED_DIR / aud_rel
            if src_aud:
                _link_or_copy(src_aud, dst_aud)
            # Clean any old audio with a different extension in same lang dir (e.g., .wav -> .flac after edge edit)
            for sibling in (config.VERIFIED_DIR / "audio" / lang_full).glob(f"{piece_id}.*"):
                if sibling.name != dst_aud.name:
                    sibling.unlink(missing_ok=True)

            call_dict = {
                "state": p["state"],
                "language": p["call_lang"],
                "lead_id": p["lead_id"],
                "audio_rel": p["call_audio_rel"],
                "source": p["call_source"],
            }
            rec = _build_piece_record(p, call_dict, p["assigned_split"] or "", aud_rel)
            _write_json(config.VERIFIED_DIR / "segments" / lang_full / f"{piece_id}.json", rec)
            _export_call_file(conn, call_id)
            if rewrite_jsonl:
                _rewrite_dataset_jsonl(conn)
        finally:
            conn.close()


def sync_all_verified(db_path: Path | None = None) -> dict:
    """Rebuild all files in labelling_data/verified/ from DB verified pieces (used at startup / after split)."""
    with _EXPORT_LOCK:
        conn = db.connect(db_path)
        try:
            config.VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
            v_rows = conn.execute("SELECT piece_id, call_id FROM pieces WHERE status = 'verified'").fetchall()
            v_pids = {r["piece_id"] for r in v_rows}
            v_cids = {r["call_id"] for r in v_rows}

            # Prune any stale segment/audio/call files no longer verified
            for sub in ("audio", "segments"):
                base = config.VERIFIED_DIR / sub
                if base.is_dir():
                    for f in base.rglob("*"):
                        if f.is_file() and f.stem not in v_pids:
                            f.unlink(missing_ok=True)
            calls_base = config.VERIFIED_DIR / "calls"
            if calls_base.is_dir():
                for f in calls_base.rglob("*.json"):
                    if f.stem not in v_cids:
                        f.unlink(missing_ok=True)

            for r in v_rows:
                sync_piece(r["piece_id"], db_path=db_path, rewrite_jsonl=False)
            _rewrite_dataset_jsonl(conn)
            return {"verified_pieces": len(v_pids), "calls_with_verified": len(v_cids)}
        finally:
            conn.close()
