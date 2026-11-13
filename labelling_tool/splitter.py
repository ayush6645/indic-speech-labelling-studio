"""Audio duplicate checker (old .m4a/.mp3 vs CRM calls) and lead-grouped stratified train/test splitter."""
import argparse
import hashlib
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labelling_tool import config, db, exporter  # noqa: E402

FRAME_HZ = 8            # 125 ms frames
SAMPLE_SR = 2000        # fast decode rate for envelope fingerprinting
MAX_FINGERPRINT_S = 60  # first 60 s of audio is plenty to uniquely identify a recording
DEFAULT_CORR_THRESHOLD = 0.88
DEFAULT_DUR_TOL_S = 1.5

_SCAN_LOCK = threading.Lock()
_SCAN_STATE = {
    "running": False,
    "scope": "",
    "done": 0,
    "total": 0,
    "matches_found": 0,
    "started_at": "",
    "finished_at": "",
    "error": "",
}


def compute_fingerprint(audio_path: Path) -> tuple[float, np.ndarray]:
    """Decode first 60s at 2kHz and return (duration_estimate_or_sample_s, float32 log-RMS envelope at 8Hz)."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
        "-t", str(MAX_FINGERPRINT_S),
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", str(SAMPLE_SR), "-f", "f32le", "-",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="replace").strip()[-200:]
        raise RuntimeError(f"ffmpeg fingerprint failed for {audio_path.name}: {err or res.returncode}")
    samples = np.frombuffer(res.stdout, np.float32)
    step = SAMPLE_SR // FRAME_HZ  # 250 samples
    n_frames = len(samples) // step
    if n_frames == 0:
        return 0.0, np.zeros(0, dtype=np.float32)
    blocks = samples[: n_frames * step].reshape(n_frames, step)
    rms = np.sqrt(np.mean(np.square(blocks, dtype=np.float64), axis=1))
    env = np.log1p(rms * 50.0).astype(np.float32)
    return round(len(samples) / SAMPLE_SR, 3), env


def match_envelopes(a: np.ndarray, b: np.ndarray, max_lag_frames: int = 12, min_overlap: int = 64) -> float:
    """Best normalized Pearson correlation between two 8Hz envelopes allowing small shift (±1.5s)."""
    la, lb = len(a), len(b)
    if min(la, lb) < min_overlap:
        return 0.0
    best = 0.0
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            sa = a[lag:]
            sb = b[: len(sa)]
        else:
            sb = b[-lag:]
            sa = a[: len(sb)]
        n = min(len(sa), len(sb))
        if n < min_overlap:
            continue
        xa = sa[:n]
        xb = sb[:n]
        xa_c = xa - xa.mean()
        xb_c = xb - xb.mean()
        denom = float(np.sqrt(np.dot(xa_c, xa_c) * np.dot(xb_c, xb_c)))
        if denom <= 1e-6:
            continue
        r = float(np.dot(xa_c, xb_c) / denom)
        if r > best:
            best = r
    return round(best, 4)


def run_overlap_check(
    scope: str = "verified",
    dur_tol_s: float = DEFAULT_DUR_TOL_S,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    db_path: Path | None = None,
    workers: int = 4,
) -> dict:
    """Compare old calls vs CRM calls by duration window + audio envelope cross-correlation.
    scope='verified' checks only calls overlapping duration with verified calls (fast);
    scope='all' checks every old vs CRM duration-matched pair.
    """
    global _SCAN_STATE
    with _SCAN_LOCK:
        _SCAN_STATE.update({
            "running": True,
            "scope": scope,
            "done": 0,
            "total": 0,
            "matches_found": 0,
            "started_at": db.now_iso(),
            "finished_at": "",
            "error": "",
        })

    try:
        conn = db.connect(db_path)
        try:
            old_calls = [dict(r) for r in conn.execute(
                "SELECT call_id, duration_s, audio_rel, md5, verified_count FROM calls WHERE source='old'"
            ).fetchall()]
            crm_calls = [dict(r) for r in conn.execute(
                "SELECT call_id, duration_s, audio_rel, md5, verified_count FROM calls WHERE source='crm'"
            ).fetchall()]
            cached_rows = conn.execute("SELECT call_id, vec_blob FROM audio_fingerprints").fetchall()
            fp_cache: dict[str, np.ndarray] = {
                r["call_id"]: np.frombuffer(r["vec_blob"], dtype=np.float32).copy()
                for r in cached_rows
            }
        finally:
            conn.close()

        # Filter to verified calls if scope == 'verified'
        if scope == "verified":
            ver_old = [c for c in old_calls if c["verified_count"] > 0]
            ver_crm = [c for c in crm_calls if c["verified_count"] > 0]
            cand_pairs: list[tuple[dict, dict]] = []
            for oc in ver_old:
                for cc in crm_calls:
                    if abs(oc["duration_s"] - cc["duration_s"]) <= dur_tol_s:
                        cand_pairs.append((oc, cc))
            seen_pairs = {(oc["call_id"], cc["call_id"]) for oc, cc in cand_pairs}
            for cc in ver_crm:
                for oc in old_calls:
                    if (oc["call_id"], cc["call_id"]) not in seen_pairs and abs(oc["duration_s"] - cc["duration_s"]) <= dur_tol_s:
                        cand_pairs.append((oc, cc))
                        seen_pairs.add((oc["call_id"], cc["call_id"]))
        else:
            # Bucket CRM calls by rounded 0.5s duration for fast lookup
            crm_by_dur: dict[int, list[dict]] = defaultdict(list)
            for cc in crm_calls:
                crm_by_dur[int(round(cc["duration_s"]))].append(cc)
            cand_pairs = []
            span = int(dur_tol_s) + 1
            for oc in old_calls:
                b0 = int(round(oc["duration_s"]))
                for b in range(b0 - span, b0 + span + 1):
                    for cc in crm_by_dur.get(b, ()):
                        if abs(oc["duration_s"] - cc["duration_s"]) <= dur_tol_s:
                            cand_pairs.append((oc, cc))

        # Exact MD5 matches bypass fingerprinting
        matches: list[tuple[str, str, float, float]] = []
        remaining_pairs: list[tuple[dict, dict]] = []
        for oc, cc in cand_pairs:
            dd = round(abs(oc["duration_s"] - cc["duration_s"]), 3)
            if oc["md5"] and cc["md5"] and oc["md5"] == cc["md5"]:
                matches.append((oc["call_id"], cc["call_id"], 1.0, dd))
            else:
                remaining_pairs.append((oc, cc))

        needed_calls: dict[str, dict] = {}
        for oc, cc in remaining_pairs:
            if oc["call_id"] not in fp_cache:
                needed_calls[oc["call_id"]] = oc
            if cc["call_id"] not in fp_cache:
                needed_calls[cc["call_id"]] = cc

        with _SCAN_LOCK:
            _SCAN_STATE["total"] = len(needed_calls) + len(remaining_pairs)

        # Compute missing fingerprints in worker threads
        new_fps: list[tuple[str, float, bytes]] = []
        if needed_calls:
            def _work(c: dict):
                p = (config.DATA_DIR / c["audio_rel"]).resolve()
                if not p.is_file():
                    return c["call_id"], c["duration_s"], np.zeros(0, dtype=np.float32)
                _, vec = compute_fingerprint(p)
                return c["call_id"], c["duration_s"], vec

            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                futs = [ex.submit(_work, c) for c in needed_calls.values()]
                for fut in as_completed(futs):
                    try:
                        cid, dur, vec = fut.result()
                        fp_cache[cid] = vec
                        new_fps.append((cid, float(dur), vec.tobytes()))
                    except Exception:
                        pass
                    with _SCAN_LOCK:
                        _SCAN_STATE["done"] += 1

        # Score remaining pairs
        for oc, cc in remaining_pairs:
            va = fp_cache.get(oc["call_id"])
            vb = fp_cache.get(cc["call_id"])
            if va is not None and vb is not None and len(va) >= 64 and len(vb) >= 64:
                score = match_envelopes(va, vb)
                if score >= corr_threshold:
                    dd = round(abs(oc["duration_s"] - cc["duration_s"]), 3)
                    matches.append((oc["call_id"], cc["call_id"], score, dd))
            with _SCAN_LOCK:
                _SCAN_STATE["done"] += 1
                _SCAN_STATE["matches_found"] = len(matches)

        now = db.now_iso()
        with db._WRITE_LOCK:
            conn = db.connect(db_path)
            try:
                if new_fps:
                    conn.executemany(
                        "INSERT INTO audio_fingerprints (call_id, duration_s, vec_blob) VALUES (?, ?, ?) "
                        "ON CONFLICT(call_id) DO UPDATE SET duration_s=excluded.duration_s, vec_blob=excluded.vec_blob",
                        new_fps,
                    )
                if matches:
                    conn.executemany(
                        "INSERT INTO audio_overlaps (old_call_id, crm_call_id, score, dur_diff_s, confirmed, detected_at) "
                        "VALUES (?, ?, ?, ?, 1, ?) "
                        "ON CONFLICT(old_call_id, crm_call_id) DO UPDATE SET score=excluded.score, "
                        "dur_diff_s=excluded.dur_diff_s, detected_at=excluded.detected_at",
                        [(o, c, s, d, now) for o, c, s, d in matches],
                    )
                conn.commit()
            finally:
                conn.close()

        with _SCAN_LOCK:
            _SCAN_STATE.update({
                "running": False,
                "finished_at": db.now_iso(),
                "matches_found": len(matches),
            })
        return {
            "scope": scope,
            "candidate_pairs": len(cand_pairs),
            "fingerprints_computed": len(new_fps),
            "matches_found": len(matches),
            "matches": [{"old_call_id": o, "crm_call_id": c, "score": s, "dur_diff_s": d} for o, c, s, d in matches],
        }
    except Exception as exc:
        with _SCAN_LOCK:
            _SCAN_STATE.update({"running": False, "error": str(exc), "finished_at": db.now_iso()})
        raise


def start_overlap_check_background(**kwargs) -> bool:
    with _SCAN_LOCK:
        if _SCAN_STATE["running"]:
            return False
    t = threading.Thread(target=run_overlap_check, kwargs=kwargs, daemon=True)
    t.start()
    return True


def get_overlap_status(db_path: Path | None = None) -> dict:
    conn = db.connect(db_path)
    try:
        overlaps = [
            dict(r) for r in conn.execute(
                "SELECT * FROM audio_overlaps ORDER BY score DESC, old_call_id"
            ).fetchall()
        ]
        fp_count = conn.execute("SELECT COUNT(*) FROM audio_fingerprints").fetchone()[0]
        ver_calls = conn.execute("SELECT COUNT(DISTINCT call_id) FROM pieces WHERE status='verified'").fetchone()[0]
        split_counts = {
            r["split"]: r["cnt"]
            for r in conn.execute("SELECT split, COUNT(*) AS cnt FROM splits GROUP BY split").fetchall()
        }
    finally:
        conn.close()
    with _SCAN_LOCK:
        scan = dict(_SCAN_STATE)
    return {
        "scan": scan,
        "fingerprints_cached": fp_count,
        "verified_calls": ver_calls,
        "overlaps_count": len(overlaps),
        "overlaps": overlaps,
        "current_splits": split_counts,
    }


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def preview_split(
    test_ratio: float = 0.15,
    seed: int = 42,
    respect_old_train: bool = True,
    db_path: Path | None = None,
) -> dict:
    """Compute a leakage-free lead-grouped stratified train/test split over verified calls."""
    test_ratio = min(max(float(test_ratio), 0.01), 0.5)
    conn = db.connect(db_path)
    try:
        all_calls = {
            r["call_id"]: dict(r)
            for r in conn.execute(
                "SELECT call_id, source, state, language, lead_id, old_split, verified_count, duration_s FROM calls"
            ).fetchall()
        }
        overlaps = [
            dict(r) for r in conn.execute("SELECT * FROM audio_overlaps WHERE confirmed = 1").fetchall()
        ]
        ver_pieces = [
            dict(r) for r in conn.execute(
                "SELECT piece_id, call_id, duration_s, main_language, old_split FROM pieces WHERE status='verified'"
            ).fetchall()
        ]
    finally:
        conn.close()

    if not ver_pieces:
        return {
            "total_verified_pieces": 0,
            "total_verified_hours": 0.0,
            "assignments": {},
            "by_lang": {},
            "overlaps": overlaps,
            "leakage_check": {"shared_leads_across_splits": 0, "shared_overlaps_across_splits": 0},
        }

    ver_dur_by_call: dict[str, float] = defaultdict(float)
    ver_cnt_by_call: dict[str, int] = defaultdict(int)
    ver_lang_dur_by_call: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    piece_old_splits_by_call: dict[str, set[str]] = defaultdict(set)
    for p in ver_pieces:
        cid = p["call_id"]
        ver_dur_by_call[cid] += float(p["duration_s"])
        ver_cnt_by_call[cid] += 1
        lang = p["main_language"] or all_calls.get(cid, {}).get("language", "") or "unknown"
        ver_lang_dur_by_call[cid][lang] += float(p["duration_s"])
        if p.get("old_split"):
            piece_old_splits_by_call[cid].add(p["old_split"])

    uf = _UnionFind()
    for cid in all_calls:
        uf.find(cid)

    by_lead: dict[str, list[str]] = defaultdict(list)
    for cid, c in all_calls.items():
        lid = (c.get("lead_id") or "").strip()
        if lid:
            by_lead[lid].append(cid)
    for cids in by_lead.values():
        for other in cids[1:]:
            uf.union(cids[0], other)

    for ov in overlaps:
        if ov["old_call_id"] in all_calls and ov["crm_call_id"] in all_calls:
            uf.union(ov["old_call_id"], ov["crm_call_id"])

    comps: dict[str, list[str]] = defaultdict(list)
    for cid in ver_cnt_by_call:
        comps[uf.find(cid)].append(cid)

    # Expand each active component to check ALL calls connected to it (even unverified old_test/old_train)
    all_members: dict[str, list[str]] = defaultdict(list)
    active_roots = set(comps.keys())
    for cid in all_calls:
        root = uf.find(cid)
        if root in active_roots:
            all_members[root].append(cid)

    comp_info: dict[str, dict] = {}
    by_lang_assignable: dict[str, list[str]] = defaultdict(list)
    lang_totals: dict[str, float] = defaultdict(float)
    lang_pinned_test: dict[str, float] = defaultdict(float)

    for root in sorted(active_roots):
        members = all_members[root]
        old_splits = set()
        for cid in members:
            os_val = (all_calls[cid].get("old_split") or "").strip().lower()
            if os_val:
                old_splits.add(os_val)
            old_splits |= piece_old_splits_by_call.get(cid, set())

        lang_scores: dict[str, float] = defaultdict(float)
        dur_sum = 0.0
        for cid in comps[root]:
            dur_sum += ver_dur_by_call[cid]
            for lg, d in ver_lang_dur_by_call[cid].items():
                lang_scores[lg] += d
        primary_lang = max(lang_scores.items(), key=lambda kv: kv[1])[0] if lang_scores else "unknown"

        if "test" in old_splits:
            split = "test"
            reason = "old_test"
            lang_pinned_test[primary_lang] += dur_sum
            lang_totals[primary_lang] += dur_sum
        elif old_splits == {"heldback"}:
            split = "heldback"
            reason = "old_heldback"
        elif respect_old_train and "train" in old_splits:
            split = "train"
            reason = "old_train"
            lang_totals[primary_lang] += dur_sum
        else:
            split = "train"
            reason = "assignable"
            by_lang_assignable[primary_lang].append(root)
            lang_totals[primary_lang] += dur_sum

        comp_info[root] = {
            "root": root,
            "verified_calls": comps[root],
            "all_calls": members,
            "primary_lang": primary_lang,
            "verified_dur_s": dur_sum,
            "split": split,
            "reason": reason,
        }

    rng = random.Random(seed)
    for lang, roots in by_lang_assignable.items():
        # Deterministic shuffle per language + seed
        ordered = list(roots)
        ordered.sort(key=lambda r: hashlib.md5(f"{seed}:{lang}:{r}".encode()).hexdigest())
        rng.shuffle(ordered)
        target_s = test_ratio * lang_totals[lang]
        cur_test_s = lang_pinned_test[lang]
        for root in ordered:
            d = comp_info[root]["verified_dur_s"]
            # Pick into test if adding improves closeness to target
            if cur_test_s < target_s and (cur_test_s + d - target_s) <= (target_s - cur_test_s) + 0.25 * target_s:
                comp_info[root]["split"] = "test"
                cur_test_s += d

    assignments: dict[str, tuple[str, str]] = {}
    for root, info in comp_info.items():
        for cid in info["verified_calls"]:
            assignments[cid] = (info["split"], root)

    # Aggregate stats & verify zero leakage
    by_lang_stats: dict[str, dict] = {}
    lead_splits: dict[str, set[str]] = defaultdict(set)
    for cid, (sp, _) in assignments.items():
        lid = (all_calls[cid].get("lead_id") or "").strip()
        if lid:
            lead_splits[lid].add(sp)

    for p in ver_pieces:
        cid = p["call_id"]
        sp, _ = assignments[cid]
        lg = p["main_language"] or all_calls[cid].get("language", "") or "unknown"
        st = by_lang_stats.setdefault(lg, {
            "language": lg,
            "train_s": 0.0, "test_s": 0.0, "heldback_s": 0.0,
            "train_pieces": 0, "test_pieces": 0, "heldback_pieces": 0,
            "train_calls": set(), "test_calls": set(),
        })
        if sp == "test":
            st["test_s"] += float(p["duration_s"])
            st["test_pieces"] += 1
            st["test_calls"].add(cid)
        elif sp == "heldback":
            st["heldback_s"] += float(p["duration_s"])
            st["heldback_pieces"] += 1
        else:
            st["train_s"] += float(p["duration_s"])
            st["train_pieces"] += 1
            st["train_calls"].add(cid)

    out_lang = {}
    for lg, st in sorted(by_lang_stats.items()):
        tt = st["train_s"] + st["test_s"]
        out_lang[lg] = {
            "language": lg,
            "train_hours": round(st["train_s"] / 3600, 3),
            "test_hours": round(st["test_s"] / 3600, 3),
            "heldback_hours": round(st["heldback_s"] / 3600, 3),
            "train_pieces": st["train_pieces"],
            "test_pieces": st["test_pieces"],
            "heldback_pieces": st["heldback_pieces"],
            "train_calls": len(st["train_calls"]),
            "test_calls": len(st["test_calls"]),
            "test_ratio_actual": round(st["test_s"] / tt, 3) if tt > 0 else 0.0,
        }

    shared_leads = sum(1 for s in lead_splits.values() if len(s - {"heldback"}) > 1)
    shared_overlaps = 0
    for ov in overlaps:
        a = assignments.get(ov["old_call_id"], (None,))[0]
        b = assignments.get(ov["crm_call_id"], (None,))[0]
        if a and b and a != b:
            shared_overlaps += 1

    return {
        "total_verified_pieces": len(ver_pieces),
        "total_verified_hours": round(sum(float(p["duration_s"]) for p in ver_pieces) / 3600, 3),
        "components_count": len(comp_info),
        "pinned_old_test_components": sum(1 for i in comp_info.values() if i["reason"] == "old_test"),
        "pinned_old_train_components": sum(1 for i in comp_info.values() if i["reason"] == "old_train"),
        "assignments": {cid: {"split": sp, "component_id": root} for cid, (sp, root) in assignments.items()},
        "by_lang": out_lang,
        "overlaps": overlaps,
        "leakage_check": {
            "shared_leads_across_splits": shared_leads,
            "shared_overlaps_across_splits": shared_overlaps,
        },
    }


def apply_split(
    test_ratio: float = 0.15,
    seed: int = 42,
    respect_old_train: bool = True,
    db_path: Path | None = None,
) -> dict:
    """Persist computed splits in SQLite and immediately regenerate verified/ dataset & JSONs."""
    plan = preview_split(test_ratio=test_ratio, seed=seed, respect_old_train=respect_old_train, db_path=db_path)
    if plan["leakage_check"]["shared_leads_across_splits"] or plan["leakage_check"]["shared_overlaps_across_splits"]:
        raise RuntimeError(f"Split leakage check failed: {plan['leakage_check']}")

    now = db.now_iso()
    rows = [(cid, info["split"], info["component_id"], now) for cid, info in plan["assignments"].items()]
    with db._WRITE_LOCK:
        conn = db.connect(db_path)
        try:
            conn.execute("DELETE FROM splits")
            if rows:
                conn.executemany(
                    "INSERT INTO splits (call_id, split, component_id, updated_at) VALUES (?, ?, ?, ?)",
                    rows,
                )
            conn.commit()
        finally:
            conn.close()
    exporter.sync_all_verified(db_path=db_path)
    plan["applied"] = True
    return plan


def main():
    db.init_db()
    db.sync_from_labelling_data()
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ov = sub.add_parser("check-overlaps", help="Check audio duplicates between old and CRM calls")
    p_ov.add_argument("--scope", choices=("verified", "all"), default="all")
    p_ov.add_argument("--dur-tol", type=float, default=DEFAULT_DUR_TOL_S)
    p_ov.add_argument("--threshold", type=float, default=DEFAULT_CORR_THRESHOLD)

    p_pr = sub.add_parser("preview", help="Preview lead-grouped stratified train/test split")
    p_pr.add_argument("--test-ratio", type=float, default=0.15)
    p_pr.add_argument("--seed", type=int, default=42)

    p_ap = sub.add_parser("apply", help="Apply train/test split to DB and verified/")
    p_ap.add_argument("--test-ratio", type=float, default=0.15)
    p_ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    if args.cmd == "check-overlaps":
        res = run_overlap_check(scope=args.scope, dur_tol_s=args.dur_tol, corr_threshold=args.threshold)
        print(f"Checked {res['candidate_pairs']} pairs -> {res['matches_found']} matches")
        for m in res["matches"]:
            print(f"  {m['old_call_id']} <-> {m['crm_call_id']} score={m['score']} dur_diff={m['dur_diff_s']}s")
    elif args.cmd == "preview":
        res = preview_split(test_ratio=args.test_ratio, seed=args.seed)
        print(f"Verified pieces: {res['total_verified_pieces']} ({res['total_verified_hours']} h), leakage: {res['leakage_check']}")
        for lg, s in res["by_lang"].items():
            print(f"  {lg:10} train={s['train_hours']}h ({s['train_pieces']}) | test={s['test_hours']}h ({s['test_pieces']}) ratio={s['test_ratio_actual']}")
    elif args.cmd == "apply":
        res = apply_split(test_ratio=args.test_ratio, seed=args.seed)
        print(f"Applied splits to {len(res['assignments'])} verified calls and rebuilt labelling_data/verified/.")


if __name__ == "__main__":
    main()
