"""Move every call and piece that has no verified text to D:\\GG, then free the space on C.

    python labelling_tool/move_unverified.py --copy      copy + check (md5) each file to D:\\GG, write metadata (resumable)
    python labelling_tool/move_unverified.py --delete    delete every C copy (all hard links) of the files checked on D

A file counts as unverified when its call has no verified piece. Every hard link of the same file anywhere in the
project folder is found by inode, so deleting really frees the space. --delete only removes files listed as checked
in D:\\GG\\moved_files.jsonl.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "labelling_data"
DEST = Path("D:/GG")
LOG = DEST / "moved_files.jsonl"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".aac", ".amr", ".webm"}
csv.field_size_limit(10 ** 9)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def plan():
    db = sqlite3.connect(f"file:{DATA / 'labelling.db'}?mode=ro", uri=True)
    keep = {r[0] for r in db.execute("select distinct call_id from pieces where status='verified'")}
    files = []                                            # (source path under labelling_data, dest rel)
    for c in csv.DictReader(open(DATA / "calls.csv", encoding="utf-8")):
        if c["call_id"] not in keep:
            files.append((DATA / c["audio"], c["audio"]))
    for cid, rel in db.execute("select call_id, audio_rel from pieces"):
        if cid not in keep and rel:
            files.append((DATA / rel, rel))
    return keep, files


def write_metadata(keep):
    meta = DEST / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{DATA / 'labelling.db'}?mode=ro", uri=True)
    out = sqlite3.connect(meta / "labelling.db")
    src.backup(out)                                       # full copy of the tool's database (safe while it runs)
    out.close()
    src.row_factory = sqlite3.Row
    with open(meta / "pieces.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for p in src.execute("select * from pieces order by source, call_id, piece_index"):
            if p["call_id"] in keep:
                continue
            row = dict(p)
            row["call_audio"] = ""
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    calls = [c for c in csv.DictReader(open(DATA / "calls.csv", encoding="utf-8")) if c["call_id"] not in keep]
    with open(meta / "calls.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(calls[0]))
        w.writeheader()
        w.writerows(calls)
    for name in ("call_texts.jsonl", "drafts.jsonl", "pieces.csv", "README.md", "spelling_guide.md"):
        if (DATA / name).exists():
            shutil.copy2(DATA / name, meta / name)
    (DEST / "README.md").write_text(
        "# Unverified CRM ASR data (moved off C on " + time.strftime("%Y-%m-%d") + ")\n\n"
        "- audio/calls/: whole calls with no verified piece; audio/pieces/: their pieces (same paths as in "
        "labelling_data/).\n- metadata/pieces.jsonl: every piece of these calls (call_id, start_s/end_s inside the "
        "call, draft/current native + roman text, status).\n- metadata/calls.csv: one row per call.\n"
        "- metadata/labelling.db: full copy of the labelling tool's database at move time.\n"
        "- moved_files.jsonl: each file, its md5, and the C paths it was removed from.\n", encoding="utf-8")


def copy():
    keep, files = plan()
    DEST.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["rel"] for l in open(LOG, encoding="utf-8")} if LOG.exists() else set()
    todo = [(s, r) for s, r in files if r not in done]
    print(f"{len(files):,} files to move, {len(done):,} already copied, {len(todo):,} left", flush=True)
    t0, nbytes = time.time(), 0
    with open(LOG, "a", encoding="utf-8", newline="\n") as log:
        for i, (src, rel) in enumerate(todo, 1):
            if not src.exists():
                log.write(json.dumps({"rel": rel, "missing": True}) + "\n")
                continue
            dst = DEST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            a, b = md5(src), md5(dst)
            if a != b:
                raise SystemExit(f"md5 differs after copy: {rel}")
            nbytes += dst.stat().st_size
            log.write(json.dumps({"rel": rel, "md5": a, "size": dst.stat().st_size, "ino": os.stat(src).st_ino}) + "\n")
            if i % 2000 == 0:
                log.flush()
                el = time.time() - t0
                print(f"  {i:,}/{len(todo):,} files, {nbytes / 1e9:.1f} GB, {nbytes / 1e6 / el:.0f} MB/s", flush=True)
    write_metadata(keep)
    print("copy finished and checked; metadata written to D:\\GG\\metadata", flush=True)


def delete():
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8")]
    ok = {r["ino"]: r for r in rows if not r.get("missing")}
    print(f"{len(ok):,} checked files on D; finding every hard link on C ...", flush=True)
    links = defaultdict(list)
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for n in fn:
            if Path(n).suffix.lower() in AUDIO_EXT:
                p = os.path.join(dp, n)
                try:
                    ino = os.stat(p).st_ino
                except OSError:
                    continue
                if ino in ok:
                    links[ino].append(p)
    freed, removed, kept_elsewhere = 0, 0, 0
    for ino, r in ok.items():
        d = DEST / r["rel"]
        if not d.exists() or d.stat().st_size != r["size"]:
            raise SystemExit(f"D copy missing or wrong size, stopping: {r['rel']}")
        paths = links.get(ino, [])
        for p in paths:
            os.remove(p)
            removed += 1
        if paths:
            freed += r["size"]
    hf = DATA / "hf_dataset"
    if hf.exists():
        shutil.rmtree(hf)
    print(f"removed {removed:,} links of {len(ok):,} files, {freed / 1e9:.1f} GB freed on C; hf_dataset/ removed",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    if a.copy:
        copy()
    if a.delete:
        delete()
