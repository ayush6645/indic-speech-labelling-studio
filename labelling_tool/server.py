"""FastAPI server for the manual audio labelling & verification tool (default port 8092).

Run:
    python labelling_tool/server.py [--port 8092] [--host 127.0.0.1] [--local-user gyan]
or:
    python -m labelling_tool.server
"""
import argparse
import copy
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from labelling_tool import audio_utils, auth, config, db, exporter, importer, metrics, splitter  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
MIN_PIECE_DURATION_S = 0.4


# -- Pydantic request bodies -----------------------------------------------------------------------------------------

class LoginReq(BaseModel):
    username: str
    password: str


class UserCreateReq(BaseModel):
    username: str
    password: str = ""
    display_name: str = ""
    role: str = "reviewer"
    active: bool | None = None


class SavePieceReq(BaseModel):
    current_text: str | None = None
    current_text_roman: str | None = None
    main_language: str | None = None
    also_languages: list[str] | None = None
    needs_update_script: str | None = None
    action: str = "edit"


class VerifyPieceReq(BaseModel):
    current_text: str | None = None
    current_text_roman: str | None = None
    main_language: str | None = None
    also_languages: list[str] | None = None


class UndoReq(BaseModel):
    history_id: int | None = None
    confirm_unverify: bool = False


class MoveEdgeReq(BaseModel):
    side: str  # "start" or "end"
    time_s: float


class SplitPieceReq(BaseModel):
    time_s: float


class GuideReq(BaseModel):
    content: str


class ImportPathReq(BaseModel):
    path: str


class OverlapCheckReq(BaseModel):
    scope: str = "verified"  # "verified" | "all"
    background: bool = True


class SplitPlanReq(BaseModel):
    test_ratio: float = Field(default=0.15, ge=0.01, le=0.5)
    seed: int = 42
    respect_old_train: bool = True


# -- Dependency helpers ----------------------------------------------------------------------------------------------

def require_user(request: Request) -> dict:
    local_user = getattr(request.app.state, "local_user", config.DEFAULT_LOCAL_USER)
    return auth.get_current_user(request, local_user=local_user)


def require_admin(user: dict = Depends(require_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin role required")
    return user


# -- Lifespan --------------------------------------------------------------------------------------------------------

def create_app(local_user: str = config.DEFAULT_LOCAL_USER, auto_sync: bool = True, verbose: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app_inst: FastAPI):
        app_inst.state.local_user = local_user
        if verbose:
            print("Initializing database...", flush=True)
        db.init_db()
        if auto_sync:
            if verbose:
                print("Checking labelling_data/ for updates...", flush=True)
            res = db.sync_from_labelling_data(force=False)
            if verbose:
                msg = f"Synced manifests in {res['seconds']}s." if res.get("synced") else "Manifests up to date."
                print(msg, flush=True)
        auth.ensure_local_user(local_user)
        bk = db.maybe_daily_backup(force=False)
        if verbose and bk:
            print(f"Daily backup saved: {bk}", flush=True)
        # Rebuild verified/ when missing or when export schema version upgrades
        EXPORTER_VERSION = "2"
        conn = db.connect()
        try:
            v_cnt = conn.execute("SELECT COUNT(*) FROM pieces WHERE status='verified'").fetchone()[0]
            exp_ver_row = conn.execute("SELECT value FROM meta WHERE key='exporter_version'").fetchone()
            exp_ver = exp_ver_row["value"] if exp_ver_row else ""
        finally:
            conn.close()
        if v_cnt > 0 and (not (config.VERIFIED_DIR / "dataset.jsonl").exists() or exp_ver != EXPORTER_VERSION):
            exporter.sync_all_verified()
            with db._WRITE_LOCK:
                conn = db.connect()
                try:
                    conn.execute(
                        "INSERT INTO meta (key, value) VALUES ('exporter_version', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (EXPORTER_VERSION,),
                    )
                    conn.commit()
                finally:
                    conn.close()
        if verbose:
            url = getattr(app_inst.state, "listen_url", f"http://127.0.0.1:{config.DEFAULT_PORT}")
            print(f"\nLabelling Tool is running!\n  Open in browser: {url}\n  Local auto-login user: {local_user} (Ctrl+C to stop)\n", flush=True)
        yield

    app_inst = FastAPI(title="Labelling Tool", lifespan=lifespan)
    app_inst.state.local_user = local_user
    if STATIC_DIR.is_dir():
        app_inst.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    register_routes(app_inst)
    return app_inst


# -- Internal helpers ------------------------------------------------------------------------------------------------

def _call_audio_path(call_row: dict) -> Path:
    return audio_utils.safe_data_path(call_row["audio_rel"])


def _piece_audio_path(piece_row: dict) -> Path:
    rel = piece_row.get("edited_audio_rel") or piece_row.get("audio_rel")
    return audio_utils.safe_data_path(rel)


def _format_piece(row: dict) -> dict:
    d = db.piece_to_dict(row)
    rel = d.get("edited_audio_rel") or d.get("audio_rel") or ""
    p = (config.DATA_DIR / rel)
    try:
        st = p.stat()
        d["audio_v"] = f"{st.st_mtime_ns:x}"
    except OSError:
        d["audio_v"] = "0"
    d["needs_update_script"] = d.get("needs_update_script") or ""
    return d


def _build_calls_where(state: str, lang: str, source: str, status: str, q: str) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if state and state != "all":
        clauses.append("c.state = ?")
        params.append(state)
    if lang and lang != "all":
        if lang == "unknown":
            clauses.append("c.language = ''")
        else:
            clauses.append("(c.language = ? OR EXISTS (SELECT 1 FROM pieces px WHERE px.call_id=c.call_id AND px.main_language=?))")
            params.extend([lang, lang])
    if source and source != "all":
        clauses.append("c.source = ?")
        params.append(source)

    st = (status or "ready").lower()
    if st == "ready":
        clauses.append("c.pieces_with_text > 0")
    elif st == "unverified":
        clauses.append("c.pieces_with_text > 0 AND c.verified_count < c.pieces_count")
    elif st == "partial":
        clauses.append("c.verified_count > 0 AND c.verified_count < c.pieces_count")
    elif st == "verified":
        clauses.append("c.verified_count > 0")
    elif st == "complete":
        clauses.append("c.pieces_count > 0 AND c.verified_count = c.pieces_count")
    elif st == "no_draft":
        clauses.append("c.pieces_count > 0 AND c.pieces_with_text = 0")
    elif st == "not_cut":
        clauses.append("c.pieces_count = 0")
    elif st == "flagged":
        clauses.append("c.flags != ''")
    # st == "all" adds no status clause

    q_clean = (q or "").strip()
    if q_clean:
        like = f"%{q_clean}%"
        clauses.append(
            "(c.call_id LIKE ? OR c.lead_id LIKE ? OR c.crm_call_id LIKE ? OR "
            "EXISTS (SELECT 1 FROM pieces pq WHERE pq.call_id=c.call_id AND pq.piece_id LIKE ?))"
        )
        params.extend([like, like, like, like])

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


def _make_unique_split_id(conn, base_piece_id: str) -> str:
    existing = {r[0] for r in conn.execute("SELECT piece_id FROM pieces UNION SELECT piece_id FROM deleted_pieces")}
    for idx in range(2, 1000):
        cand = f"{base_piece_id}_p{idx}"
        if cand not in existing:
            return cand
    raise RuntimeError("Could not allocate unique split piece_id")


def _snapshot_call_pieces(conn, call_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM pieces WHERE call_id = ? ORDER BY piece_index", (call_id,)).fetchall()
    return [dict(r) for r in rows]


def _restore_call_pieces(conn, call_id: str, snapshot: dict):
    before = snapshot.get("before", [])
    created_ids = snapshot.get("created_ids", [])
    deleted_ids = snapshot.get("deleted_ids", [])
    call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone())
    call_audio = _call_audio_path(call_row)
    wf = audio_utils.compute_waveform(call_id, call_audio)
    peak_amp = wf.get("peak_amp")

    for pid in created_ids:
        conn.execute("DELETE FROM pieces WHERE piece_id = ?", (pid,))
        (config.EDITED_AUDIO_DIR / f"{pid}.flac").unlink(missing_ok=True)
        audio_utils.invalidate_opus_cache(f"piece_{pid}")

    for pid in deleted_ids:
        conn.execute("DELETE FROM deleted_pieces WHERE piece_id = ?", (pid,))

    cols = [
        "piece_id", "call_id", "source", "piece_index", "piece_count", "start_s", "end_s", "duration_s",
        "audio_rel", "silent", "old_split", "flags", "draft_text", "draft_text_roman", "draft_language",
        "draft_by", "draft_decoded_at", "draft_loop", "draft_silent", "has_draft", "status",
        "current_text", "current_text_roman", "main_language", "also_languages", "needs_update_script",
        "human_edited", "edges_edited", "edited_audio_rel", "verified_by", "verified_at", "updated_by", "updated_at",
    ]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols[1:])
    sql = f"INSERT INTO pieces ({', '.join(cols)}) VALUES ({placeholders}) ON CONFLICT(piece_id) DO UPDATE SET {updates}"
    for p in before:
        conn.execute(sql, tuple(p[c] for c in cols))
        audio_utils.invalidate_opus_cache(f"piece_{p['piece_id']}")
        if p.get("edited_audio_rel"):
            dst = config.DATA_DIR / p["edited_audio_rel"]
            audio_utils.recut_piece_audio(
                call_audio, float(p["start_s"]), float(p["end_s"]), dst,
                call_duration_s=float(call_row["duration_s"]), peak_amp=peak_amp,
            )
        else:
            (config.EDITED_AUDIO_DIR / f"{p['piece_id']}.flac").unlink(missing_ok=True)

    # Update call edges_edited flag if no pieces remain edge-edited
    any_edge = conn.execute("SELECT COUNT(*) FROM pieces WHERE call_id=? AND edges_edited=1", (call_id,)).fetchone()[0]
    conn.execute("UPDATE calls SET edges_edited=? WHERE call_id=?", (1 if any_edge else 0, call_id))
    db.recompute_call_stats(conn, call_id)


def _compute_lang_wer(rows: list[dict], draft_filter_prefix: str = "v3") -> tuple[float | None, float | None, int]:
    ed_nat = ref_nat = ed_rom = ref_rom = count = 0
    for r in rows:
        if draft_filter_prefix and not (r.get("draft_by") or "").startswith(draft_filter_prefix):
            continue
        if not r.get("has_draft"):
            continue
        count += 1
        _, en, rn = metrics.compute_wer(r["current_text"], r["draft_text"])
        _, er, rr = metrics.compute_wer(r["current_text_roman"], r["draft_text_roman"])
        ed_nat += en
        ref_nat += rn
        ed_rom += er
        ref_rom += rr
    wer_nat = round((ed_nat / ref_nat) * 100, 1) if ref_nat > 0 else None
    wer_rom = round((ed_rom / ref_rom) * 100, 1) if ref_rom > 0 else None
    return wer_nat, wer_rom, count


# -- Routes ----------------------------------------------------------------------------------------------------------

def register_routes(app: FastAPI):
    @app.get("/")
    def index():
        idx = STATIC_DIR / "index.html"
        if not idx.is_file():
            raise HTTPException(404, "Frontend not built yet")
        return FileResponse(idx, media_type="text/html", headers={"Cache-Control": "no-store"})

    # -- Auth & Users --
    @app.get("/api/me")
    def api_me(request: Request):
        local_user = getattr(request.app.state, "local_user", config.DEFAULT_LOCAL_USER)
        is_local = auth.is_local_request(request)
        try:
            u = auth.get_current_user(request, local_user=local_user)
            return {
                "authenticated": True,
                "is_local": is_local,
                "user": u,
                "default_audio_fmt": "raw" if is_local else "opus",
            }
        except HTTPException:
            return {
                "authenticated": False,
                "is_local": False,
                "user": None,
                "default_audio_fmt": "opus",
            }

    @app.post("/api/login")
    def api_login(req: LoginReq, request: Request, response: Response):
        u = auth.authenticate_user(req.username, req.password)
        if not u:
            raise HTTPException(401, "Invalid username or password")
        token = auth.create_session(u["username"])
        response.set_cookie(
            key=auth.SESSION_COOKIE,
            value=token,
            httponly=True,
            samesite="lax",
            max_age=auth.SESSION_DAYS * 86400,
        )
        is_local = auth.is_local_request(request)
        return {
            "authenticated": True,
            "is_local": is_local,
            "user": u | {"local": is_local},
            "default_audio_fmt": "raw" if is_local else "opus",
        }

    @app.post("/api/logout")
    def api_logout(request: Request, response: Response):
        token = request.cookies.get(auth.SESSION_COOKIE)
        auth.delete_session(token)
        response.delete_cookie(auth.SESSION_COOKIE)
        return {"ok": True}

    @app.get("/api/users")
    def api_list_users(_: dict = Depends(require_admin)):
        return {"users": auth.list_users()}

    @app.post("/api/users")
    def api_save_user(req: UserCreateReq, _: dict = Depends(require_admin)):
        if req.active is not None and not req.password:
            auth.set_user_active(req.username.strip().lower(), req.active)
            return {"ok": True, "users": auth.list_users()}
        try:
            u = auth.create_user(req.username, req.password, display_name=req.display_name, role=req.role)
            if req.active is False:
                auth.set_user_active(u["username"], False)
            return {"user": u, "users": auth.list_users()}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- Filters & Calls --
    @app.get("/api/filters")
    def api_filters(_: dict = Depends(require_user)):
        conn = db.connect()
        try:
            states = [r["state"] for r in conn.execute("SELECT DISTINCT state FROM calls ORDER BY state").fetchall()]
            langs = [r["language"] for r in conn.execute("SELECT DISTINCT language FROM calls WHERE language != '' ORDER BY language").fetchall()]
            sources = [r["source"] for r in conn.execute("SELECT DISTINCT source FROM calls ORDER BY source").fetchall()]
            total_calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            ready_calls = conn.execute("SELECT COUNT(*) FROM calls WHERE pieces_with_text > 0").fetchone()[0]
            verified_calls = conn.execute("SELECT COUNT(*) FROM calls WHERE verified_count > 0").fetchone()[0]
            complete_calls = conn.execute("SELECT COUNT(*) FROM calls WHERE pieces_count > 0 AND verified_count = pieces_count").fetchone()[0]
        finally:
            conn.close()
        return {
            "states": states,
            "languages": langs,
            "lang_meta": {
                k: {"label": config.LANG_LABEL.get(k, k), "color": config.LANG_COLORS.get(k, "#6B6F76"), "full": config.LANG_FULL.get(k, k)}
                for k in (*config.LANGS, "none")
            },
            "sources": sources,
            "counts": {
                "total_calls": total_calls,
                "ready_calls": ready_calls,
                "verified_calls": verified_calls,
                "complete_calls": complete_calls,
            },
        }

    @app.get("/api/calls")
    def api_calls(
        state: str = "all",
        lang: str = "all",
        source: str = "all",
        status: str = "ready",
        q: str = "",
        sort: str = "resume",
        offset: int = 0,
        limit: int = 120,
        _: dict = Depends(require_user),
    ):
        limit = min(max(limit, 1), 500)
        offset = max(offset, 0)
        where_sql, params = _build_calls_where(state, lang, source, status, q)
        if sort == "duration":
            order_sql = "ORDER BY c.duration_s DESC, c.call_id ASC"
        elif sort == "id":
            order_sql = "ORDER BY c.call_id ASC"
        else:
            # Put calls with verified progress (but incomplete) first, then ready unstarted, then completed
            order_sql = (
                "ORDER BY "
                "CASE WHEN c.pieces_count > 0 AND c.verified_count = c.pieces_count THEN 2 "
                "     WHEN c.verified_count > 0 THEN 0 "
                "     ELSE 1 END ASC, "
                "c.source ASC, c.state ASC, c.call_id ASC"
            )

        conn = db.connect()
        try:
            agg = conn.execute(f"""
                SELECT
                    COUNT(*) AS total_calls,
                    COALESCE(SUM(CASE WHEN c.pieces_count > 0 THEN COALESCE(ps.ver_s, 0) ELSE 0 END), 0) AS verified_s,
                    COALESCE(SUM(CASE WHEN c.pieces_count > 0 THEN COALESCE(ps.tot_s, 0) ELSE c.duration_s END), 0) AS total_s,
                    COALESCE(SUM(c.verified_count), 0) AS verified_pieces,
                    COALESCE(SUM(c.pieces_count), 0) AS total_pieces
                FROM calls c
                LEFT JOIN (
                    SELECT call_id,
                           SUM(duration_s) AS tot_s,
                           SUM(CASE WHEN status='verified' THEN duration_s ELSE 0 END) AS ver_s
                    FROM pieces
                    GROUP BY call_id
                ) ps ON ps.call_id = c.call_id
                {where_sql}
            """, params).fetchone()

            rows = conn.execute(f"""
                SELECT c.*, COALESCE(sp.split, '') AS assigned_split
                FROM calls c
                LEFT JOIN splits sp ON sp.call_id = c.call_id
                {where_sql}
                {order_sql}
                LIMIT ? OFFSET ?
            """, [*params, limit, offset]).fetchall()

            lang_ver_rows = conn.execute("""
                SELECT
                    CASE
                        WHEN p.main_language != '' THEN p.main_language
                        WHEN c.language != '' THEN c.language
                        ELSE 'unknown'
                    END AS lang,
                    COALESCE(SUM(p.duration_s), 0) AS ver_s,
                    COUNT(*) AS ver_pieces
                FROM pieces p
                JOIN calls c ON c.call_id = p.call_id
                WHERE p.status = 'verified'
                GROUP BY lang
            """).fetchall()
        finally:
            conn.close()

        ver_by_lang_map = {
            r["lang"]: (float(r["ver_s"] or 0.0), int(r["ver_pieces"] or 0))
            for r in lang_ver_rows
        }
        verified_by_lang = []
        for lg in config.LANGS:
            vs, vp = ver_by_lang_map.get(lg, (0.0, 0))
            verified_by_lang.append({
                "language": lg,
                "label": config.LANG_LABEL.get(lg, lg.capitalize()),
                "color": config.LANG_COLORS.get(lg, "#6B6F76"),
                "verified_seconds": round(vs, 1),
                "verified_hours": round(vs / 3600, 2),
                "verified_pieces": vp,
            })
        for lg, (vs, vp) in sorted(ver_by_lang_map.items()):
            if lg not in config.LANGS and vp > 0:
                verified_by_lang.append({
                    "language": lg,
                    "label": config.LANG_LABEL.get(lg, lg.capitalize()),
                    "color": config.LANG_COLORS.get(lg, "#6B6F76"),
                    "verified_seconds": round(vs, 1),
                    "verified_hours": round(vs / 3600, 2),
                    "verified_pieces": vp,
                })

        return {
            "total": agg["total_calls"],
            "offset": offset,
            "limit": limit,
            "filtered_verified_hours": round(float(agg["verified_s"] or 0) / 3600, 2),
            "filtered_total_hours": round(float(agg["total_s"] or 0) / 3600, 2),
            "filtered_verified_pieces": int(agg["verified_pieces"] or 0),
            "filtered_total_pieces": int(agg["total_pieces"] or 0),
            "verified_by_lang": verified_by_lang,
            "calls": [dict(r) for r in rows],
        }

    @app.get("/api/call/{call_id}")
    def api_call_detail(call_id: str, _: dict = Depends(require_user)):
        conn = db.connect()
        try:
            c_row = conn.execute("""
                SELECT c.*, COALESCE(sp.split, '') AS assigned_split
                FROM calls c
                LEFT JOIN splits sp ON sp.call_id = c.call_id
                WHERE c.call_id = ?
            """, (call_id,)).fetchone()
            if not c_row:
                raise HTTPException(404, f"Call not found: {call_id}")
            pieces = conn.execute(
                "SELECT * FROM pieces WHERE call_id = ? ORDER BY piece_index",
                (call_id,),
            ).fetchall()
            ct_row = conn.execute("SELECT * FROM call_texts WHERE call_id = ?", (call_id,)).fetchone()
            overlaps = conn.execute(
                "SELECT * FROM audio_overlaps WHERE old_call_id = ? OR crm_call_id = ?",
                (call_id, call_id),
            ).fetchall()
        finally:
            conn.close()

        call_dict = dict(c_row)
        call_dict["complete"] = bool(call_dict["pieces_count"] > 0 and call_dict["verified_count"] == call_dict["pieces_count"])
        return {
            "call": call_dict,
            "pieces": [_format_piece(dict(p)) for p in pieces],
            "call_text": dict(ct_row) if ct_row else None,
            "overlaps": [dict(o) for o in overlaps],
        }

    @app.get("/api/waveform/{call_id}")
    def api_waveform(call_id: str, _: dict = Depends(require_user)):
        conn = db.connect()
        try:
            row = conn.execute("SELECT call_id, audio_rel FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            raise HTTPException(404, "Call not found")
        audio_path = _call_audio_path(dict(row))
        try:
            wf = audio_utils.compute_waveform(call_id, audio_path)
            return wf
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc

    # -- Audio streaming (local raw FLAC vs remote 24 kbps Opus) --
    @app.get("/api/audio/piece/{piece_id}")
    def api_piece_audio(piece_id: str, request: Request, fmt: str = "auto", _: dict = Depends(require_user)):
        conn = db.connect()
        try:
            row = conn.execute("SELECT piece_id, audio_rel, edited_audio_rel FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            raise HTTPException(404, "Piece not found")
        src_path = _piece_audio_path(dict(row))
        use_opus = (fmt == "opus") or (fmt == "auto" and not auth.is_local_request(request))
        if use_opus:
            opus_path = audio_utils.ensure_opus(src_path, f"piece_{piece_id}")
            return audio_utils.range_file_response(request, opus_path, "audio/ogg")
        return audio_utils.range_file_response(request, src_path)

    @app.get("/api/audio/call/{call_id}")
    def api_call_audio(call_id: str, request: Request, fmt: str = "auto", _: dict = Depends(require_user)):
        conn = db.connect()
        try:
            row = conn.execute("SELECT call_id, audio_rel FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            raise HTTPException(404, "Call not found")
        src_path = _call_audio_path(dict(row))
        use_opus = (fmt == "opus") or (fmt == "auto" and not auth.is_local_request(request))
        if use_opus:
            opus_path = audio_utils.ensure_opus(src_path, f"call_{call_id}")
            return audio_utils.range_file_response(request, opus_path, "audio/ogg")
        return audio_utils.range_file_response(request, src_path)

    # -- Save / Verify / Unverify / History / Undo --
    @app.post("/api/pieces/{piece_id}/save")
    def api_save_piece(piece_id: str, req: SavePieceReq, user: dict = Depends(require_user)):
        username = user["username"]
        with db._WRITE_LOCK:
            conn = db.connect()
            try:
                row = conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone()
                if not row:
                    raise HTTPException(404, "Piece not found")
                old = dict(row)
                new_text = old["current_text"] if req.current_text is None else req.current_text
                new_rom = old["current_text_roman"] if req.current_text_roman is None else req.current_text_roman
                new_main = (old["main_language"] if req.main_language is None else req.main_language).strip().lower()
                if new_main and new_main not in (*config.LANGS, "none"):
                    raise HTTPException(400, f"Invalid main_language: {new_main}")
                if req.also_languages is None:
                    new_also = db.parse_also_langs(old["also_languages"], new_main)
                else:
                    new_also = db.parse_also_langs(req.also_languages, new_main)
                new_needs = old["needs_update_script"] if req.needs_update_script is None else req.needs_update_script

                old_also = db.parse_also_langs(old["also_languages"], old["main_language"])
                changed_content = (
                    new_text != old["current_text"]
                    or new_rom != old["current_text_roman"]
                    or new_main != old["main_language"]
                    or new_also != old_also
                )
                if not changed_content and new_needs == old["needs_update_script"]:
                    return {"saved": False, "piece": _format_piece(old)}

                now = db.now_iso()
                human_edited = 1 if (
                    old["human_edited"]
                    or new_text != old["draft_text"]
                    or new_rom != old["draft_text_roman"]
                    or new_main != old["draft_language"]
                    or bool(new_also)
                ) else 0

                conn.execute("""
                    UPDATE pieces SET
                        current_text = ?,
                        current_text_roman = ?,
                        main_language = ?,
                        also_languages = ?,
                        needs_update_script = ?,
                        human_edited = ?,
                        updated_by = ?,
                        updated_at = ?
                    WHERE piece_id = ?
                """, (
                    new_text, new_rom, new_main, json.dumps(new_also),
                    new_needs, human_edited, username, now, piece_id,
                ))
                updated = dict(conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone())
                if changed_content:
                    db.record_history(conn, piece_id, old["call_id"], req.action or "edit", old, updated, username)
                db.recompute_call_stats(conn, old["call_id"])
                conn.commit()
                call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id = ?", (old["call_id"],)).fetchone())
            finally:
                conn.close()

        if updated["status"] == "verified":
            exporter.sync_piece(piece_id)
        return {"saved": True, "piece": _format_piece(updated), "call": call_row}

    @app.post("/api/pieces/{piece_id}/verify")
    def api_verify_piece(piece_id: str, req: VerifyPieceReq | None = None, user: dict = Depends(require_user)):
        username = user["username"]
        req = req or VerifyPieceReq()
        with db._WRITE_LOCK:
            conn = db.connect()
            try:
                row = conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone()
                if not row:
                    raise HTTPException(404, "Piece not found")
                old = dict(row)
                new_text = old["current_text"] if req.current_text is None else req.current_text
                new_rom = old["current_text_roman"] if req.current_text_roman is None else req.current_text_roman
                new_main = (old["main_language"] if req.main_language is None else req.main_language).strip().lower()
                if not new_main:
                    raise HTTPException(400, "Choose a main language before verifying")
                if new_main not in (*config.LANGS, "none"):
                    raise HTTPException(400, f"Invalid main_language: {new_main}")
                if req.also_languages is None:
                    new_also = db.parse_also_langs(old["also_languages"], new_main)
                else:
                    new_also = db.parse_also_langs(req.also_languages, new_main)

                now = db.now_iso()
                human_edited = 1 if (
                    old["human_edited"]
                    or new_text != old["draft_text"]
                    or new_rom != old["draft_text_roman"]
                    or new_main != old["draft_language"]
                    or bool(new_also)
                ) else 0

                conn.execute("""
                    UPDATE pieces SET
                        current_text = ?,
                        current_text_roman = ?,
                        main_language = ?,
                        also_languages = ?,
                        needs_update_script = '',
                        human_edited = ?,
                        status = 'verified',
                        verified_by = ?,
                        verified_at = ?,
                        updated_by = ?,
                        updated_at = ?
                    WHERE piece_id = ?
                """, (
                    new_text, new_rom, new_main, json.dumps(new_also),
                    human_edited, username, now, username, now, piece_id,
                ))
                updated = dict(conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone())
                db.record_history(conn, piece_id, old["call_id"], "verify", old, updated, username)
                db.recompute_call_stats(conn, old["call_id"])
                conn.commit()
                call_pieces = [
                    dict(p) for p in conn.execute(
                        "SELECT piece_id, piece_index, status FROM pieces WHERE call_id = ? ORDER BY piece_index",
                        (old["call_id"],),
                    ).fetchall()
                ]
                call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id = ?", (old["call_id"],)).fetchone())
            finally:
                conn.close()

        exporter.sync_piece(piece_id)
        # Find next unverified piece after current index, wrapping earlier in same call if needed
        cur_idx = updated["piece_index"]
        next_pid = None
        for p in call_pieces:
            if p["piece_index"] > cur_idx and p["status"] == "unverified":
                next_pid = p["piece_id"]
                break
        if next_pid is None:
            for p in call_pieces:
                if p["status"] == "unverified":
                    next_pid = p["piece_id"]
                    break
        call_row["complete"] = bool(call_row["pieces_count"] > 0 and call_row["verified_count"] == call_row["pieces_count"])
        return {
            "piece": _format_piece(updated),
            "call": call_row,
            "next_unverified_piece_id": next_pid,
        }

    @app.post("/api/pieces/{piece_id}/unverify")
    def api_unverify_piece(piece_id: str, user: dict = Depends(require_user)):
        username = user["username"]
        with db._WRITE_LOCK:
            conn = db.connect()
            try:
                row = conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone()
                if not row:
                    raise HTTPException(404, "Piece not found")
                old = dict(row)
                if old["status"] == "unverified":
                    call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id = ?", (old["call_id"],)).fetchone())
                    return {"piece": _format_piece(old), "call": call_row}
                now = db.now_iso()
                conn.execute("""
                    UPDATE pieces SET
                        status = 'unverified',
                        verified_by = '',
                        verified_at = '',
                        updated_by = ?,
                        updated_at = ?
                    WHERE piece_id = ?
                """, (username, now, piece_id))
                updated = dict(conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone())
                db.record_history(
                    conn, piece_id, old["call_id"], "unverify", old, updated, username,
                    snapshot={"old_verified_by": old.get("verified_by", ""), "old_verified_at": old.get("verified_at", "")},
                )
                db.recompute_call_stats(conn, old["call_id"])
                conn.commit()
                call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id = ?", (old["call_id"],)).fetchone())
            finally:
                conn.close()

        exporter.sync_piece(piece_id)
        call_row["complete"] = bool(call_row["pieces_count"] > 0 and call_row["verified_count"] == call_row["pieces_count"])
        return {"piece": _format_piece(updated), "call": call_row}

    @app.get("/api/pieces/{piece_id}/history")
    def api_piece_history(piece_id: str, _: dict = Depends(require_user)):
        conn = db.connect()
        try:
            row = conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone()
            if not row:
                raise HTTPException(404, "Piece not found")
            p = dict(row)
            hist_rows = conn.execute("""
                SELECT * FROM history
                WHERE piece_id = ? OR (call_id = ? AND action LIKE 'edge_%')
                ORDER BY id DESC LIMIT 50
            """, (piece_id, p["call_id"])).fetchall()
        finally:
            conn.close()

        revs = []
        for h in hist_rows:
            d = dict(h)
            for k in ("word_diff_native", "word_diff_roman", "old_also_langs", "new_also_langs"):
                try:
                    d[k] = json.loads(d[k] or "[]")
                except ValueError:
                    d[k] = []
            d["undone"] = bool(d["undone"])
            revs.append(d)

        return {
            "piece_id": piece_id,
            "draft": {
                "draft_by": p["draft_by"],
                "draft_decoded_at": p["draft_decoded_at"],
                "draft_language": p["draft_language"],
                "draft_text": p["draft_text"],
                "draft_text_roman": p["draft_text_roman"],
                "diff_from_draft_native": metrics.word_diff(p["draft_text"], p["current_text"]),
                "diff_from_draft_roman": metrics.word_diff(p["draft_text_roman"], p["current_text_roman"]),
            },
            "revisions": revs,
        }

    @app.post("/api/pieces/{piece_id}/undo")
    def api_undo_piece(piece_id: str, req: UndoReq | None = None, user: dict = Depends(require_user)):
        req = req or UndoReq()
        with db._WRITE_LOCK:
            conn = db.connect()
            try:
                p_row = conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone()
                if not p_row:
                    raise HTTPException(404, "Piece not found")
                p = dict(p_row)
                call_id = p["call_id"]

                if req.history_id is not None:
                    h_row = conn.execute(
                        "SELECT * FROM history WHERE id = ? AND undone = 0 AND (piece_id = ? OR (call_id = ? AND action LIKE 'edge_%'))",
                        (req.history_id, piece_id, call_id),
                    ).fetchone()
                else:
                    h_row = conn.execute(
                        "SELECT * FROM history WHERE undone = 0 AND (piece_id = ? OR (call_id = ? AND action LIKE 'edge_%')) ORDER BY id DESC LIMIT 1",
                        (piece_id, call_id),
                    ).fetchone()

                if not h_row:
                    return {"undone": False, "message": "Nothing to undo"}
                h = dict(h_row)

                # If undoing a verification -> must confirm first ("Mark as unverified?")
                if h["old_status"] == "unverified" and h["new_status"] == "verified" and not req.confirm_unverify:
                    return {
                        "undone": False,
                        "needs_confirm": True,
                        "message": "Mark as unverified?",
                        "history_id": h["id"],
                    }

                if h["action"].startswith("edge_"):
                    snap = json.loads(h["snapshot_json"] or "{}")
                    _restore_call_pieces(conn, call_id, snap)
                    conn.execute("UPDATE history SET undone = 1 WHERE id = ?", (h["id"],))
                    conn.commit()
                    pieces_after = [
                        _format_piece(dict(r))
                        for r in conn.execute("SELECT * FROM pieces WHERE call_id = ? ORDER BY piece_index", (call_id,)).fetchall()
                    ]
                    call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone())
                    return {
                        "undone": True,
                        "edge_restored": True,
                        "pieces": pieces_after,
                        "call": call_row,
                        "undone_action": h["action"],
                    }

                now = db.now_iso()
                was_verified = p["status"] == "verified"
                snap = {}
                try:
                    snap = json.loads(h.get("snapshot_json") or "{}")
                except ValueError:
                    snap = {}
                restored_v_by = snap.get("old_verified_by") or h["saved_by"] if h["old_status"] == "verified" else ""
                restored_v_at = snap.get("old_verified_at") or h["saved_at"] if h["old_status"] == "verified" else ""
                old_also_list = db.parse_also_langs(h["old_also_langs"], h["old_main_lang"])
                re_human = 1 if (
                    h["old_text"] != p["draft_text"]
                    or h["old_text_roman"] != p["draft_text_roman"]
                    or h["old_main_lang"] != p["draft_language"]
                    or bool(old_also_list)
                ) else 0
                conn.execute("""
                    UPDATE pieces SET
                        current_text = ?,
                        current_text_roman = ?,
                        main_language = ?,
                        also_languages = ?,
                        status = ?,
                        human_edited = ?,
                        verified_by = ?,
                        verified_at = ?,
                        updated_by = ?,
                        updated_at = ?
                    WHERE piece_id = ?
                """, (
                    h["old_text"], h["old_text_roman"], h["old_main_lang"], h["old_also_langs"],
                    h["old_status"], re_human,
                    restored_v_by, restored_v_at,
                    user["username"], now, h["piece_id"],
                ))
                conn.execute("UPDATE history SET undone = 1 WHERE id = ?", (h["id"],))
                db.recompute_call_stats(conn, call_id)
                conn.commit()
                updated = dict(conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (h["piece_id"],)).fetchone())
                call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone())
            finally:
                conn.close()

        if was_verified or updated["status"] == "verified":
            exporter.sync_piece(updated["piece_id"])
        return {
            "undone": True,
            "edge_restored": False,
            "piece": _format_piece(updated),
            "call": call_row,
            "undone_action": h["action"],
        }

    # -- Edge fix (continuous move / split / merge) --
    @app.post("/api/pieces/{piece_id}/move-edge")
    def api_move_edge(piece_id: str, req: MoveEdgeReq, user: dict = Depends(require_user)):
        if req.side not in ("start", "end"):
            raise HTTPException(400, "side must be 'start' or 'end'")
        with db._WRITE_LOCK:
            conn = db.connect()
            try:
                p_row = conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (piece_id,)).fetchone()
                if not p_row:
                    raise HTTPException(404, "Piece not found")
                call_id = p_row["call_id"]
                call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone())
                pieces = _snapshot_call_pieces(conn, call_id)
                idx_pos = next(i for i, x in enumerate(pieces) if x["piece_id"] == piece_id)
                cur = pieces[idx_pos]
                if cur["status"] == "verified":
                    raise HTTPException(409, "Current piece is verified — unverify it before changing edges")

                snap_before = copy.deepcopy(pieces)
                max_t = float(call_row["duration_s"]) or float(pieces[-1]["end_s"])
                new_t = round(max(0.0, min(float(req.time_s), max_t)), 3)
                affected: list[dict] = []

                if req.side == "start":
                    prev_p = pieces[idx_pos - 1] if idx_pos > 0 else None
                    if prev_p and prev_p["status"] == "verified":
                        raise HTTPException(409, f"Previous piece s{prev_p['piece_index']:03d} is verified — unverify it first")
                    lo = (prev_p["start_s"] + MIN_PIECE_DURATION_S) if prev_p else 0.0
                    hi = cur["end_s"] - MIN_PIECE_DURATION_S
                    if hi < lo:
                        raise HTTPException(400, "Piece is too short to move edge")
                    new_t = round(max(lo, min(new_t, hi)), 3)
                    cur["start_s"] = new_t
                    cur["duration_s"] = round(cur["end_s"] - cur["start_s"], 3)
                    affected.append(cur)
                    if prev_p:
                        prev_p["end_s"] = new_t
                        prev_p["duration_s"] = round(prev_p["end_s"] - prev_p["start_s"], 3)
                        affected.append(prev_p)
                else:
                    next_p = pieces[idx_pos + 1] if idx_pos + 1 < len(pieces) else None
                    if next_p and next_p["status"] == "verified":
                        raise HTTPException(409, f"Next piece s{next_p['piece_index']:03d} is verified — unverify it first")
                    lo = cur["start_s"] + MIN_PIECE_DURATION_S
                    hi = (next_p["end_s"] - MIN_PIECE_DURATION_S) if next_p else max_t
                    if hi < lo:
                        raise HTTPException(400, "Piece is too short to move edge")
                    new_t = round(max(lo, min(new_t, hi)), 3)
                    cur["end_s"] = new_t
                    cur["duration_s"] = round(cur["end_s"] - cur["start_s"], 3)
                    affected.append(cur)
                    if next_p:
                        next_p["start_s"] = new_t
                        next_p["duration_s"] = round(next_p["end_s"] - next_p["start_s"], 3)
                        affected.append(next_p)

                call_audio = _call_audio_path(call_row)
                wf = audio_utils.compute_waveform(call_id, call_audio)
                peak_amp = wf.get("peak_amp")
                now = db.now_iso()
                for ap_row in affected:
                    rel_ed = f"edited_audio/{ap_row['piece_id']}.flac"
                    audio_utils.recut_piece_audio(
                        call_audio, float(ap_row["start_s"]), float(ap_row["end_s"]),
                        config.DATA_DIR / rel_ed,
                        call_duration_s=float(call_row["duration_s"]), peak_amp=peak_amp,
                    )
                    audio_utils.invalidate_opus_cache(f"piece_{ap_row['piece_id']}")
                    conn.execute("""
                        UPDATE pieces SET
                            start_s=?, end_s=?, duration_s=?, edges_edited=1, edited_audio_rel=?,
                            updated_by=?, updated_at=?
                        WHERE piece_id=?
                    """, (
                        ap_row["start_s"], ap_row["end_s"], ap_row["duration_s"], rel_ed,
                        user["username"], now, ap_row["piece_id"],
                    ))

                conn.execute("UPDATE calls SET edges_edited=1 WHERE call_id=?", (call_id,))
                updated_cur = dict(conn.execute("SELECT * FROM pieces WHERE piece_id=?", (piece_id,)).fetchone())
                db.record_history(
                    conn, piece_id, call_id, "edge_move",
                    snap_before[idx_pos], updated_cur, user["username"],
                    snapshot={"before": snap_before, "created_ids": [], "deleted_ids": []},
                )
                db.recompute_call_stats(conn, call_id)
                conn.commit()
                final_pieces = [
                    _format_piece(dict(r))
                    for r in conn.execute("SELECT * FROM pieces WHERE call_id=? ORDER BY piece_index", (call_id,)).fetchall()
                ]
                call_out = dict(conn.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone())
            finally:
                conn.close()
        return {"pieces": final_pieces, "call": call_out}

    @app.post("/api/pieces/{piece_id}/split")
    def api_split_piece(piece_id: str, req: SplitPieceReq, user: dict = Depends(require_user)):
        with db._WRITE_LOCK:
            conn = db.connect()
            try:
                p_row = conn.execute("SELECT * FROM pieces WHERE piece_id=?", (piece_id,)).fetchone()
                if not p_row:
                    raise HTTPException(404, "Piece not found")
                call_id = p_row["call_id"]
                call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone())
                pieces = _snapshot_call_pieces(conn, call_id)
                idx_pos = next(i for i, x in enumerate(pieces) if x["piece_id"] == piece_id)
                cur = pieces[idx_pos]
                if cur["status"] == "verified":
                    raise HTTPException(409, "Unverify this piece before splitting it")

                t = round(float(req.time_s), 3)
                if t <= cur["start_s"] + MIN_PIECE_DURATION_S or t >= cur["end_s"] - MIN_PIECE_DURATION_S:
                    raise HTTPException(
                        400,
                        f"Split time {t:.2f}s must leave at least {MIN_PIECE_DURATION_S}s on both halves "
                        f"({cur['start_s'] + MIN_PIECE_DURATION_S:.2f}s .. {cur['end_s'] - MIN_PIECE_DURATION_S:.2f}s)",
                    )

                snap_before = copy.deepcopy(pieces)
                new_pid = _make_unique_split_id(conn, piece_id)
                new_p = copy.deepcopy(cur)
                new_p.update({
                    "piece_id": new_pid,
                    "start_s": t,
                    "end_s": cur["end_s"],
                    "duration_s": round(cur["end_s"] - t, 3),
                    "draft_text": "",
                    "draft_text_roman": "",
                    "has_draft": 0,
                    "status": "unverified",
                    "current_text": "",
                    "current_text_roman": "",
                    "also_languages": "[]",
                    "needs_update_script": "",
                    "human_edited": 0,
                    "edges_edited": 1,
                    "edited_audio_rel": f"edited_audio/{new_pid}.flac",
                    "verified_by": "",
                    "verified_at": "",
                    "updated_by": user["username"],
                    "updated_at": db.now_iso(),
                })
                cur["end_s"] = t
                cur["duration_s"] = round(t - cur["start_s"], 3)
                cur["edges_edited"] = 1
                cur["edited_audio_rel"] = f"edited_audio/{piece_id}.flac"
                cur["updated_by"] = user["username"]
                cur["updated_at"] = db.now_iso()

                call_audio = _call_audio_path(call_row)
                wf = audio_utils.compute_waveform(call_id, call_audio)
                peak_amp = wf.get("peak_amp")
                for half in (cur, new_p):
                    audio_utils.recut_piece_audio(
                        call_audio, float(half["start_s"]), float(half["end_s"]),
                        config.DATA_DIR / half["edited_audio_rel"],
                        call_duration_s=float(call_row["duration_s"]), peak_amp=peak_amp,
                    )
                    audio_utils.invalidate_opus_cache(f"piece_{half['piece_id']}")

                new_list = pieces[:idx_pos] + [cur, new_p] + pieces[idx_pos + 1:]
                total_cnt = len(new_list)
                for n_idx, item in enumerate(new_list, 1):
                    item["piece_index"] = n_idx
                    item["piece_count"] = total_cnt

                conn.execute(
                    "UPDATE pieces SET end_s=?, duration_s=?, edges_edited=1, edited_audio_rel=?, updated_by=?, updated_at=? WHERE piece_id=?",
                    (cur["end_s"], cur["duration_s"], cur["edited_audio_rel"], cur["updated_by"], cur["updated_at"], piece_id),
                )
                cols = [
                    "piece_id", "call_id", "source", "piece_index", "piece_count", "start_s", "end_s", "duration_s",
                    "audio_rel", "silent", "old_split", "flags", "draft_text", "draft_text_roman", "draft_language",
                    "draft_by", "draft_decoded_at", "draft_loop", "draft_silent", "has_draft", "status",
                    "current_text", "current_text_roman", "main_language", "also_languages", "needs_update_script",
                    "human_edited", "edges_edited", "edited_audio_rel", "verified_by", "verified_at", "updated_by", "updated_at",
                ]
                conn.execute(
                    f"INSERT INTO pieces ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                    tuple(new_p[c] for c in cols),
                )
                for item in new_list:
                    conn.execute(
                        "UPDATE pieces SET piece_index=?, piece_count=? WHERE piece_id=?",
                        (item["piece_index"], item["piece_count"], item["piece_id"]),
                    )
                conn.execute("UPDATE calls SET edges_edited=1 WHERE call_id=?", (call_id,))
                updated_cur = dict(conn.execute("SELECT * FROM pieces WHERE piece_id=?", (piece_id,)).fetchone())
                db.record_history(
                    conn, piece_id, call_id, "edge_split",
                    snap_before[idx_pos], updated_cur, user["username"],
                    snapshot={"before": snap_before, "created_ids": [new_pid], "deleted_ids": []},
                )
                db.recompute_call_stats(conn, call_id)
                conn.commit()
                final_pieces = [
                    _format_piece(dict(r))
                    for r in conn.execute("SELECT * FROM pieces WHERE call_id=? ORDER BY piece_index", (call_id,)).fetchall()
                ]
                call_out = dict(conn.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone())
            finally:
                conn.close()
        return {"pieces": final_pieces, "call": call_out, "created_piece_id": new_pid}

    @app.post("/api/pieces/{piece_id}/merge")
    def api_merge_piece(piece_id: str, user: dict = Depends(require_user)):
        with db._WRITE_LOCK:
            conn = db.connect()
            try:
                p_row = conn.execute("SELECT * FROM pieces WHERE piece_id=?", (piece_id,)).fetchone()
                if not p_row:
                    raise HTTPException(404, "Piece not found")
                call_id = p_row["call_id"]
                call_row = dict(conn.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone())
                pieces = _snapshot_call_pieces(conn, call_id)
                idx_pos = next(i for i, x in enumerate(pieces) if x["piece_id"] == piece_id)
                if idx_pos + 1 >= len(pieces):
                    raise HTTPException(400, "No next piece to merge with")
                cur = pieces[idx_pos]
                nxt = pieces[idx_pos + 1]
                if cur["status"] == "verified" or nxt["status"] == "verified":
                    raise HTTPException(409, "Unverify both pieces before merging them")

                snap_before = copy.deepcopy(pieces)
                cur["end_s"] = nxt["end_s"]
                cur["duration_s"] = round(nxt["end_s"] - cur["start_s"], 3)
                t1, t2 = (cur["current_text"] or "").strip(), (nxt["current_text"] or "").strip()
                r1, r2 = (cur["current_text_roman"] or "").strip(), (nxt["current_text_roman"] or "").strip()
                cur["current_text"] = " ".join(x for x in (t1, t2) if x)
                cur["current_text_roman"] = " ".join(x for x in (r1, r2) if x)
                also = db.parse_also_langs(
                    db.parse_also_langs(cur["also_languages"]) + db.parse_also_langs(nxt["also_languages"]),
                    cur["main_language"],
                )
                cur["also_languages"] = json.dumps(also)
                cur["edges_edited"] = 1
                cur["edited_audio_rel"] = f"edited_audio/{piece_id}.flac"
                cur["human_edited"] = 1
                cur["updated_by"] = user["username"]
                cur["updated_at"] = db.now_iso()

                call_audio = _call_audio_path(call_row)
                wf = audio_utils.compute_waveform(call_id, call_audio)
                audio_utils.recut_piece_audio(
                    call_audio, float(cur["start_s"]), float(cur["end_s"]),
                    config.DATA_DIR / cur["edited_audio_rel"],
                    call_duration_s=float(call_row["duration_s"]), peak_amp=wf.get("peak_amp"),
                )
                audio_utils.invalidate_opus_cache(f"piece_{piece_id}")
                audio_utils.invalidate_opus_cache(f"piece_{nxt['piece_id']}")

                conn.execute("DELETE FROM pieces WHERE piece_id = ?", (nxt["piece_id"],))
                conn.execute(
                    "INSERT INTO deleted_pieces (piece_id, call_id, deleted_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(piece_id) DO UPDATE SET deleted_at=excluded.deleted_at",
                    (nxt["piece_id"], call_id, db.now_iso()),
                )
                conn.execute("""
                    UPDATE pieces SET
                        end_s=?, duration_s=?, current_text=?, current_text_roman=?, also_languages=?,
                        edges_edited=1, edited_audio_rel=?, human_edited=1, updated_by=?, updated_at=?
                    WHERE piece_id=?
                """, (
                    cur["end_s"], cur["duration_s"], cur["current_text"], cur["current_text_roman"],
                    cur["also_languages"], cur["edited_audio_rel"], cur["updated_by"], cur["updated_at"], piece_id,
                ))

                remaining = [cur if x["piece_id"] == piece_id else x for x in pieces if x["piece_id"] != nxt["piece_id"]]
                total_cnt = len(remaining)
                for n_idx, item in enumerate(remaining, 1):
                    conn.execute("UPDATE pieces SET piece_index=?, piece_count=? WHERE piece_id=?", (n_idx, total_cnt, item["piece_id"]))

                conn.execute("UPDATE calls SET edges_edited=1 WHERE call_id=?", (call_id,))
                updated_cur = dict(conn.execute("SELECT * FROM pieces WHERE piece_id=?", (piece_id,)).fetchone())
                db.record_history(
                    conn, piece_id, call_id, "edge_merge",
                    snap_before[idx_pos], updated_cur, user["username"],
                    snapshot={"before": snap_before, "created_ids": [], "deleted_ids": [nxt["piece_id"]]},
                )
                db.recompute_call_stats(conn, call_id)
                conn.commit()
                final_pieces = [
                    _format_piece(dict(r))
                    for r in conn.execute("SELECT * FROM pieces WHERE call_id=? ORDER BY piece_index", (call_id,)).fetchall()
                ]
                call_out = dict(conn.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone())
            finally:
                conn.close()
        return {"pieces": final_pieces, "call": call_out, "deleted_piece_id": nxt["piece_id"]}

    # -- Progress & Model Error Rate (WER) --
    @app.get("/api/progress")
    def api_progress(_: dict = Depends(require_user)):
        conn = db.connect()
        try:
            ver_rows = [
                dict(r) for r in conn.execute("""
                    SELECT p.piece_id, p.duration_s, p.main_language, p.current_text, p.current_text_roman,
                           p.draft_text, p.draft_text_roman, p.draft_by, p.has_draft, c.state, c.language AS call_lang
                    FROM pieces p
                    JOIN calls c ON c.call_id = p.call_id
                    WHERE p.status = 'verified'
                """).fetchall()
            ]
            totals_by_lang = {
                r["lang"]: {"pieces": r["cnt"], "s": float(r["dur"] or 0)}
                for r in conn.execute("""
                    SELECT COALESCE(NULLIF(main_language, ''), 'unknown') AS lang,
                           COUNT(*) AS cnt, COALESCE(SUM(duration_s), 0) AS dur
                    FROM pieces
                    GROUP BY lang
                """).fetchall()
            }
            totals_by_state = conn.execute("""
                SELECT c.state,
                       COUNT(p.piece_id) AS total_pieces,
                       COALESCE(SUM(p.duration_s), 0) AS total_s,
                       SUM(CASE WHEN p.status='verified' THEN 1 ELSE 0 END) AS ver_pieces,
                       COALESCE(SUM(CASE WHEN p.status='verified' THEN p.duration_s ELSE 0 END), 0) AS ver_s
                FROM calls c
                LEFT JOIN pieces p ON p.call_id = c.call_id
                GROUP BY c.state
                ORDER BY c.state
            """).fetchall()
            summary_row = conn.execute("""
                SELECT
                    COUNT(*) AS total_pieces,
                    COALESCE(SUM(duration_s), 0) AS total_s,
                    SUM(CASE WHEN status='verified' THEN 1 ELSE 0 END) AS ver_pieces,
                    COALESCE(SUM(CASE WHEN status='verified' THEN duration_s ELSE 0 END), 0) AS ver_s
                FROM pieces
            """).fetchone()
            call_cnts = conn.execute("""
                SELECT
                    COUNT(*) AS total_calls,
                    SUM(CASE WHEN verified_count > 0 THEN 1 ELSE 0 END) AS ver_calls,
                    SUM(CASE WHEN pieces_count > 0 AND verified_count = pieces_count THEN 1 ELSE 0 END) AS complete_calls
                FROM calls
            """).fetchone()
        finally:
            conn.close()

        by_lang_ver: dict[str, list[dict]] = {}
        for r in ver_rows:
            lg = r["main_language"] or r["call_lang"] or "unknown"
            by_lang_ver.setdefault(lg, []).append(r)

        lang_keys = list(config.LANGS) + sorted(
            (set(totals_by_lang.keys()) | set(by_lang_ver.keys())) - set(config.LANGS)
        )
        # Preserve unique order
        seen_lg = []
        for lg in lang_keys:
            if lg not in seen_lg:
                seen_lg.append(lg)

        lang_table = []
        for lg in seen_lg:
            v_list = by_lang_ver.get(lg, [])
            tot = totals_by_lang.get(lg, {"pieces": 0, "s": 0.0})
            if not v_list and not tot["pieces"]:
                continue
            v3_nat, v3_rom, v3_cnt = _compute_lang_wer(v_list, draft_filter_prefix="v3")
            all_nat, all_rom, all_cnt = _compute_lang_wer(v_list, draft_filter_prefix="")
            ver_s = sum(float(x["duration_s"]) for x in v_list)
            lang_table.append({
                "language": lg,
                "label": config.LANG_LABEL.get(lg, lg.capitalize()),
                "color": config.LANG_COLORS.get(lg, "#6B6F76"),
                "verified_hours": round(ver_s / 3600, 2),
                "verified_pieces": len(v_list),
                "total_hours": round(tot["s"] / 3600, 2),
                "total_pieces": tot["pieces"],
                "v3_eval_pieces": v3_cnt,
                "v3_wer_native": v3_nat,
                "v3_wer_roman": v3_rom,
                "all_eval_pieces": all_cnt,
                "all_wer_native": all_nat,
                "all_wer_roman": all_rom,
            })

        return {
            "summary": {
                "verified_pieces": int(summary_row["ver_pieces"] or 0),
                "total_pieces": int(summary_row["total_pieces"] or 0),
                "verified_hours": round(float(summary_row["ver_s"] or 0) / 3600, 2),
                "total_hours": round(float(summary_row["total_s"] or 0) / 3600, 2),
                "verified_calls": int(call_cnts["ver_calls"] or 0),
                "complete_calls": int(call_cnts["complete_calls"] or 0),
                "total_calls": int(call_cnts["total_calls"] or 0),
            },
            "by_lang": lang_table,
            "by_state": [
                {
                    "state": r["state"],
                    "verified_hours": round(float(r["ver_s"] or 0) / 3600, 2),
                    "total_hours": round(float(r["total_s"] or 0) / 3600, 2),
                    "verified_pieces": int(r["ver_pieces"] or 0),
                    "total_pieces": int(r["total_pieces"] or 0),
                }
                for r in totals_by_state
            ],
        }

    # -- Spelling guide (editable in UI, persisted to spelling_guide.md) --
    @app.get("/api/guide")
    def api_get_guide(_: dict = Depends(require_user)):
        return {"content": db.get_spelling_guide()}

    @app.put("/api/guide")
    def api_put_guide(req: GuideReq, _: dict = Depends(require_user)):
        return {"content": db.save_spelling_guide(req.content)}

    # -- Dynamic Session Import & Sync (A5) --
    @app.post("/api/import/upload")
    async def api_import_upload(file: UploadFile = File(...), _: dict = Depends(require_user)):
        content = await file.read()
        if not content:
            raise HTTPException(400, "Empty file uploaded")
        try:
            res = importer.import_uploaded_bytes(file.filename or "session_results.zip", content)
            return res
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/import/path")
    def api_import_path(req: ImportPathReq, _: dict = Depends(require_user)):
        try:
            return importer.import_local_path(req.path)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/sync")
    def api_force_sync(_: dict = Depends(require_user)):
        res = db.sync_from_labelling_data(force=True)
        return res

    # -- Audio Overlap Check & Lead-Grouped Train/Test Split (Feature 12 + A7) --
    @app.get("/api/split/status")
    def api_split_status(_: dict = Depends(require_user)):
        return splitter.get_overlap_status()

    @app.post("/api/split/check-overlaps")
    def api_split_check_overlaps(req: OverlapCheckReq | None = None, _: dict = Depends(require_user)):
        req = req or OverlapCheckReq()
        if req.scope not in ("verified", "all"):
            raise HTTPException(400, "scope must be 'verified' or 'all'")
        if req.background:
            started = splitter.start_overlap_check_background(scope=req.scope)
            return {"started": started, "status": splitter.get_overlap_status()}
        res = splitter.run_overlap_check(scope=req.scope)
        return {"started": False, "result": res, "status": splitter.get_overlap_status()}

    @app.post("/api/split/preview")
    def api_split_preview(req: SplitPlanReq | None = None, _: dict = Depends(require_user)):
        req = req or SplitPlanReq()
        return splitter.preview_split(
            test_ratio=req.test_ratio,
            seed=req.seed,
            respect_old_train=req.respect_old_train,
        )

    @app.post("/api/split/apply")
    def api_split_apply(req: SplitPlanReq | None = None, _: dict = Depends(require_user)):
        req = req or SplitPlanReq()
        try:
            return splitter.apply_split(
                test_ratio=req.test_ratio,
                seed=req.seed,
                respect_old_train=req.respect_old_train,
            )
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- Hot SQLite backup --
    @app.post("/api/backup")
    def api_backup(_: dict = Depends(require_user)):
        dst = db.maybe_daily_backup(force=True)
        return {"backup_path": str(dst) if dst else None}


app = create_app()


def main():
    import uvicorn
    import webbrowser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="Use 127.0.0.1 (ngrok connects locally and triggers login)")
    ap.add_argument("--local-user", default=config.DEFAULT_LOCAL_USER)
    ap.add_argument("--open", action="store_true", help="Open browser automatically on startup")
    args = ap.parse_args()
    browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browse_host}:{args.port}"
    application = create_app(local_user=args.local_user, auto_sync=True, verbose=True)
    application.state.listen_url = url
    if args.open:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(application, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
