"""SQLite database management, sync with labelling_data/, history/undo, and daily hot backups."""
import csv
import datetime as dt
import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

from labelling_tool import config
from labelling_tool.metrics import word_diff

_WRITE_LOCK = threading.RLock()
_SYNC_FILES = (
    ("calls_csv", config.CALLS_CSV),
    ("pieces_csv", config.PIECES_CSV),
    ("drafts_jsonl", config.DRAFTS_JSONL),
    ("call_texts_jsonl", config.CALL_TEXTS_JSONL),
)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    p = db_path or config.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def init_db(db_path: Path | None = None):
    with _WRITE_LOCK:
        conn = connect(db_path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    state TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT '',
                    also_languages TEXT NOT NULL DEFAULT '',
                    lead_id TEXT NOT NULL DEFAULT '',
                    crm_call_id TEXT NOT NULL DEFAULT '',
                    duration_s REAL NOT NULL DEFAULT 0,
                    pieces_count INTEGER NOT NULL DEFAULT 0,
                    pieces_with_text INTEGER NOT NULL DEFAULT 0,
                    verified_count INTEGER NOT NULL DEFAULT 0,
                    audio_rel TEXT NOT NULL DEFAULT '',
                    md5 TEXT NOT NULL DEFAULT '',
                    old_split TEXT NOT NULL DEFAULT '',
                    has_call_text INTEGER NOT NULL DEFAULT 0,
                    flags TEXT NOT NULL DEFAULT '',
                    edges_edited INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS pieces (
                    piece_id TEXT PRIMARY KEY,
                    call_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    piece_index INTEGER NOT NULL,
                    piece_count INTEGER NOT NULL,
                    start_s REAL NOT NULL,
                    end_s REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    audio_rel TEXT NOT NULL,
                    silent INTEGER NOT NULL DEFAULT 0,
                    old_split TEXT NOT NULL DEFAULT '',
                    flags TEXT NOT NULL DEFAULT '',
                    draft_text TEXT NOT NULL DEFAULT '',
                    draft_text_roman TEXT NOT NULL DEFAULT '',
                    draft_language TEXT NOT NULL DEFAULT '',
                    draft_by TEXT NOT NULL DEFAULT '',
                    draft_decoded_at TEXT NOT NULL DEFAULT '',
                    draft_loop INTEGER NOT NULL DEFAULT 0,
                    draft_silent INTEGER NOT NULL DEFAULT 0,
                    has_draft INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'unverified',
                    current_text TEXT NOT NULL DEFAULT '',
                    current_text_roman TEXT NOT NULL DEFAULT '',
                    main_language TEXT NOT NULL DEFAULT '',
                    also_languages TEXT NOT NULL DEFAULT '[]',
                    needs_update_script TEXT NOT NULL DEFAULT '',
                    human_edited INTEGER NOT NULL DEFAULT 0,
                    edges_edited INTEGER NOT NULL DEFAULT 0,
                    edited_audio_rel TEXT NOT NULL DEFAULT '',
                    verified_by TEXT NOT NULL DEFAULT '',
                    verified_at TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS deleted_pieces (
                    piece_id TEXT PRIMARY KEY,
                    call_id TEXT NOT NULL,
                    deleted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS call_texts (
                    call_id TEXT PRIMARY KEY,
                    text_native TEXT NOT NULL DEFAULT '',
                    text_romanized TEXT NOT NULL DEFAULT '',
                    v4_model TEXT NOT NULL DEFAULT '',
                    v4_diarized_text TEXT NOT NULL DEFAULT '',
                    v4_english_text TEXT NOT NULL DEFAULT '',
                    project TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    piece_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    old_text TEXT NOT NULL DEFAULT '',
                    new_text TEXT NOT NULL DEFAULT '',
                    old_text_roman TEXT NOT NULL DEFAULT '',
                    new_text_roman TEXT NOT NULL DEFAULT '',
                    old_main_lang TEXT NOT NULL DEFAULT '',
                    new_main_lang TEXT NOT NULL DEFAULT '',
                    old_also_langs TEXT NOT NULL DEFAULT '[]',
                    new_also_langs TEXT NOT NULL DEFAULT '[]',
                    old_status TEXT NOT NULL DEFAULT 'unverified',
                    new_status TEXT NOT NULL DEFAULT 'unverified',
                    word_diff_native TEXT NOT NULL DEFAULT '[]',
                    word_diff_roman TEXT NOT NULL DEFAULT '[]',
                    draft_model TEXT NOT NULL DEFAULT '',
                    saved_by TEXT NOT NULL DEFAULT '',
                    saved_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL DEFAULT '{}',
                    undone INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'reviewer',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audio_fingerprints (
                    call_id TEXT PRIMARY KEY,
                    duration_s REAL NOT NULL,
                    vec_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audio_overlaps (
                    old_call_id TEXT NOT NULL,
                    crm_call_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    dur_diff_s REAL NOT NULL,
                    confirmed INTEGER NOT NULL DEFAULT 1,
                    detected_at TEXT NOT NULL,
                    PRIMARY KEY (old_call_id, crm_call_id)
                );
                CREATE TABLE IF NOT EXISTS splits (
                    call_id TEXT PRIMARY KEY,
                    split TEXT NOT NULL,
                    component_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pieces_call ON pieces(call_id, piece_index);
                CREATE INDEX IF NOT EXISTS idx_pieces_status ON pieces(status);
                CREATE INDEX IF NOT EXISTS idx_pieces_lang ON pieces(main_language);
                CREATE INDEX IF NOT EXISTS idx_calls_filters ON calls(source, state, language);
                CREATE INDEX IF NOT EXISTS idx_history_piece ON history(piece_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_history_call ON history(call_id, id DESC);
            """)
            conn.commit()
        finally:
            conn.close()


def _file_sig(path: Path) -> str:
    if not path.exists():
        return "missing"
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def sync_from_labelling_data(force: bool = False, db_path: Path | None = None) -> dict:
    """Sync calls.csv, pieces.csv, drafts.jsonl, and call_texts.jsonl into SQLite.
    Never modifies verified pieces or human edge edits.
    """
    init_db(db_path)
    with _WRITE_LOCK:
        conn = connect(db_path)
        try:
            sigs = {k: _file_sig(p) for k, p in _SYNC_FILES}
            if not force:
                existing = {
                    r["key"]: r["value"]
                    for r in conn.execute("SELECT key, value FROM meta WHERE key LIKE 'sig_%'")
                }
                if all(existing.get(f"sig_{k}") == v for k, v in sigs.items()):
                    return {"synced": False, "reason": "up_to_date"}

            t0 = time.time()
            # 1. calls.csv
            if config.CALLS_CSV.exists():
                with open(config.CALLS_CSV, encoding="utf-8", newline="") as f:
                    call_rows = list(csv.DictReader(f))
                conn.executemany("""
                    INSERT INTO calls (
                        call_id, source, state, language, also_languages, lead_id, crm_call_id,
                        duration_s, pieces_count, pieces_with_text, audio_rel, md5, old_split,
                        has_call_text, flags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(call_id) DO UPDATE SET
                        source=excluded.source,
                        state=excluded.state,
                        language=CASE WHEN calls.language='' THEN excluded.language ELSE calls.language END,
                        also_languages=CASE WHEN calls.also_languages='' THEN excluded.also_languages ELSE calls.also_languages END,
                        lead_id=excluded.lead_id,
                        crm_call_id=excluded.crm_call_id,
                        duration_s=excluded.duration_s,
                        audio_rel=excluded.audio_rel,
                        md5=excluded.md5,
                        old_split=excluded.old_split,
                        has_call_text=excluded.has_call_text,
                        flags=excluded.flags
                """, [
                    (
                        r["call_id"], r["source"], r["state"], r["language"], r["also_languages"],
                        r["lead_id"], r["crm_call_id"], float(r["duration_s"] or 0),
                        int(r["pieces"] or 0), int(r["pieces_with_text"] or 0),
                        r["audio"], r["md5"], r["old_split"],
                        1 if r["has_call_text"] == "yes" else 0, r["flags"],
                    )
                    for r in call_rows
                ])

            # 2. pieces.csv
            deleted_ids = {r["piece_id"] for r in conn.execute("SELECT piece_id FROM deleted_pieces")}
            call_lang_map = {r["call_id"]: r["language"] for r in conn.execute("SELECT call_id, language FROM calls")}
            if config.PIECES_CSV.exists():
                with open(config.PIECES_CSV, encoding="utf-8", newline="") as f:
                    piece_rows = list(csv.DictReader(f))
                batch = []
                for r in piece_rows:
                    pid = r["piece_id"]
                    if pid in deleted_ids:
                        continue
                    cid = r["call_id"]
                    silent_val = 1 if str(r.get("silent", "")).strip() == "1" else 0
                    default_lang = call_lang_map.get(cid, "")
                    batch.append((
                        pid, cid, r["source"], int(r["index"]), int(r["count"]),
                        float(r["start_s"]), float(r["end_s"]), float(r["duration_s"]),
                        r["audio"], silent_val, r["old_split"], r["flags"],
                        default_lang,
                    ))
                conn.executemany("""
                    INSERT INTO pieces (
                        piece_id, call_id, source, piece_index, piece_count,
                        start_s, end_s, duration_s, audio_rel, silent, old_split, flags,
                        main_language
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(piece_id) DO UPDATE SET
                        audio_rel=excluded.audio_rel,
                        silent=excluded.silent,
                        old_split=excluded.old_split,
                        flags=excluded.flags,
                        piece_index=CASE WHEN pieces.edges_edited=0 THEN excluded.piece_index ELSE pieces.piece_index END,
                        piece_count=CASE WHEN pieces.edges_edited=0 THEN excluded.piece_count ELSE pieces.piece_count END,
                        start_s=CASE WHEN pieces.edges_edited=0 THEN excluded.start_s ELSE pieces.start_s END,
                        end_s=CASE WHEN pieces.edges_edited=0 THEN excluded.end_s ELSE pieces.end_s END,
                        duration_s=CASE WHEN pieces.edges_edited=0 THEN excluded.duration_s ELSE pieces.duration_s END
                """, batch)

            # 3. drafts.jsonl — NEVER touch verified pieces!
            if config.DRAFTS_JSONL.exists():
                draft_batch = []
                with open(config.DRAFTS_JSONL, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        pid = d.get("piece_id")
                        if not pid or pid in deleted_ids:
                            continue
                        lang = d.get("language") or ""
                        txt = d.get("text") or ""
                        rom = d.get("text_roman") or ""
                        draft_batch.append((
                            txt, rom, lang, d.get("draft_by") or "",
                            d.get("decoded_at") or "", 1 if d.get("loop") else 0,
                            1 if d.get("silent") else 0,
                            txt, rom, lang,
                            pid,
                        ))
                conn.executemany("""
                    UPDATE pieces SET
                        draft_text = ?,
                        draft_text_roman = ?,
                        draft_language = ?,
                        draft_by = ?,
                        draft_decoded_at = ?,
                        draft_loop = ?,
                        draft_silent = ?,
                        has_draft = 1,
                        current_text = CASE WHEN human_edited=0 THEN ? ELSE current_text END,
                        current_text_roman = CASE WHEN human_edited=0 THEN ? ELSE current_text_roman END,
                        main_language = CASE WHEN human_edited=0 AND ? != '' THEN ? ELSE main_language END
                    WHERE piece_id = ? AND status = 'unverified'
                """, [
                    (
                        t[0], t[1], t[2], t[3], t[4], t[5], t[6],
                        t[7], t[8], t[9], t[9], t[10],
                    )
                    for t in draft_batch
                ])

            # 4. call_texts.jsonl
            if config.CALL_TEXTS_JSONL.exists():
                ct_batch = []
                with open(config.CALL_TEXTS_JSONL, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        t = json.loads(line)
                        ct_batch.append((
                            t["call_id"],
                            t.get("text_native", ""),
                            t.get("text_romanized", ""),
                            t.get("v4_model", ""),
                            t.get("v4_diarized_text", ""),
                            t.get("v4_english_text", ""),
                            t.get("project", ""),
                        ))
                conn.executemany("""
                    INSERT INTO call_texts (
                        call_id, text_native, text_romanized, v4_model,
                        v4_diarized_text, v4_english_text, project
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(call_id) DO UPDATE SET
                        text_native=excluded.text_native,
                        text_romanized=excluded.text_romanized,
                        v4_model=excluded.v4_model,
                        v4_diarized_text=excluded.v4_diarized_text,
                        v4_english_text=excluded.v4_english_text,
                        project=excluded.project
                """, ct_batch)

            # 5. Recompute per-call stats (pieces_count, pieces_with_text, verified_count, language)
            recompute_all_calls_stats(conn)

            for k, v in sigs.items():
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (f"sig_{k}", v),
                )
            conn.commit()
            return {"synced": True, "seconds": round(time.time() - t0, 2)}
        finally:
            conn.close()


def recompute_all_calls_stats(conn: sqlite3.Connection):
    conn.execute("""
        UPDATE calls SET
            pieces_count = COALESCE((SELECT COUNT(*) FROM pieces p WHERE p.call_id = calls.call_id), 0),
            pieces_with_text = COALESCE((SELECT COUNT(*) FROM pieces p WHERE p.call_id = calls.call_id AND p.has_draft = 1), 0),
            verified_count = COALESCE((SELECT COUNT(*) FROM pieces p WHERE p.call_id = calls.call_id AND p.status = 'verified'), 0)
    """)
    # Fill missing CRM call languages from piece main_language / draft_language
    rows = conn.execute("""
        SELECT p.call_id, p.main_language, COUNT(*) AS cnt
        FROM pieces p
        JOIN calls c ON c.call_id = p.call_id
        WHERE c.source = 'crm' AND p.main_language != ''
        GROUP BY p.call_id, p.main_language
        ORDER BY p.call_id, cnt DESC
    """).fetchall()
    by_call: dict[str, list[str]] = {}
    for r in rows:
        by_call.setdefault(r["call_id"], []).append(r["main_language"])
    if by_call:
        conn.executemany(
            "UPDATE calls SET language = ?, also_languages = ? WHERE call_id = ?",
            [(langs[0], " ".join(langs[1:]), cid) for cid, langs in by_call.items()],
        )


def recompute_call_stats(conn: sqlite3.Connection, call_id: str):
    pieces = conn.execute(
        "SELECT main_language, also_languages, has_draft, status FROM pieces WHERE call_id = ? ORDER BY piece_index",
        (call_id,),
    ).fetchall()
    total = len(pieces)
    with_text = sum(1 for p in pieces if p["has_draft"])
    verified = sum(1 for p in pieces if p["status"] == "verified")
    call = conn.execute("SELECT source, language, also_languages FROM calls WHERE call_id = ?", (call_id,)).fetchone()
    if call and call["source"] == "crm":
        langs = Counter(p["main_language"] for p in pieces if p["main_language"])
        main_lang = langs.most_common(1)[0][0] if langs else call["language"]
        extras = set()
        for l, _ in langs.most_common()[1:]:
            extras.add(l)
        for p in pieces:
            try:
                for l in json.loads(p["also_languages"] or "[]"):
                    if l and l != main_lang:
                        extras.add(l)
            except ValueError:
                pass
        conn.execute(
            "UPDATE calls SET pieces_count=?, pieces_with_text=?, verified_count=?, language=?, also_languages=? WHERE call_id=?",
            (total, with_text, verified, main_lang, " ".join(sorted(extras)), call_id),
        )
    else:
        conn.execute(
            "UPDATE calls SET pieces_count=?, pieces_with_text=?, verified_count=? WHERE call_id=?",
            (total, with_text, verified, call_id),
        )


def maybe_daily_backup(force: bool = False, db_path: Path | None = None) -> Path | None:
    """Create a safe hot SQLite backup once per day (or when forced)."""
    src_path = db_path or config.DB_PATH
    if not src_path.exists():
        return None
    today = dt.date.today().isoformat()
    config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = config.BACKUP_DIR / f"labelling_{today}.db"
    if dst.exists() and not force:
        return None
    if force and dst.exists():
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dst = config.BACKUP_DIR / f"labelling_{stamp}.db"
    src_conn = connect(src_path)
    try:
        tmp = dst.with_suffix(".tmp")
        dst_conn = sqlite3.connect(str(tmp))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
        tmp.replace(dst)
        # Keep up to 30 daily backups to protect disk space
        backups = sorted(config.BACKUP_DIR.glob("labelling_*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[30:]:
            old.unlink(missing_ok=True)
        return dst
    finally:
        src_conn.close()


def parse_also_langs(raw: str | list[str] | None, main_lang: str = "") -> list[str]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip().startswith("[") else raw.replace(",", " ").split()
            items = parsed if isinstance(parsed, list) else []
        except ValueError:
            items = raw.replace(",", " ").split()
    else:
        items = []
    seen = []
    for x in items:
        code = str(x).strip().lower()
        if code and code != main_lang and code not in seen:
            seen.append(code)
    return seen


def piece_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["also_languages"] = parse_also_langs(d.get("also_languages"), d.get("main_language", ""))
    d["silent"] = bool(d.get("silent"))
    d["draft_loop"] = bool(d.get("draft_loop"))
    d["draft_silent"] = bool(d.get("draft_silent"))
    d["has_draft"] = bool(d.get("has_draft"))
    d["human_edited"] = bool(d.get("human_edited"))
    d["edges_edited"] = bool(d.get("edges_edited"))
    return d


def record_history(
    conn: sqlite3.Connection,
    piece_id: str,
    call_id: str,
    action: str,
    old_piece: dict,
    new_piece: dict,
    saved_by: str,
    snapshot: dict | None = None,
) -> int:
    old_text = old_piece.get("current_text", "")
    new_text = new_piece.get("current_text", "")
    old_rom = old_piece.get("current_text_roman", "")
    new_rom = new_piece.get("current_text_roman", "")
    diff_nat = word_diff(old_text, new_text)
    diff_rom = word_diff(old_rom, new_rom)
    cur = conn.execute("""
        INSERT INTO history (
            piece_id, call_id, action,
            old_text, new_text, old_text_roman, new_text_roman,
            old_main_lang, new_main_lang, old_also_langs, new_also_langs,
            old_status, new_status, word_diff_native, word_diff_roman,
            draft_model, saved_by, saved_at, snapshot_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        piece_id, call_id, action,
        old_text, new_text, old_rom, new_rom,
        old_piece.get("main_language", ""), new_piece.get("main_language", ""),
        json.dumps(parse_also_langs(old_piece.get("also_languages"))),
        json.dumps(parse_also_langs(new_piece.get("also_languages"))),
        old_piece.get("status", "unverified"), new_piece.get("status", "unverified"),
        json.dumps(diff_nat, ensure_ascii=False),
        json.dumps(diff_rom, ensure_ascii=False),
        old_piece.get("draft_by", ""),
        saved_by, now_iso(),
        json.dumps(snapshot or {}, ensure_ascii=False),
    ))
    return int(cur.lastrowid)


def get_spelling_guide() -> str:
    if config.SPELLING_GUIDE_PATH.exists():
        return config.SPELLING_GUIDE_PATH.read_text(encoding="utf-8")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.SPELLING_GUIDE_PATH.write_text(config.DEFAULT_SPELLING_GUIDE, encoding="utf-8", newline="\n")
    return config.DEFAULT_SPELLING_GUIDE


def save_spelling_guide(content: str) -> str:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = config.SPELLING_GUIDE_PATH.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    tmp.replace(config.SPELLING_GUIDE_PATH)
    return content
