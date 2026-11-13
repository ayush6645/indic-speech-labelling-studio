"""Build labelling_data/: ONE folder with every call, piece and transcript, old and new, for the labelling tool.

What goes in:
  CRM calls (new)  Bangalore, Kolkata, Maharashtra, Hyderabad; only calls of 30 s or more (the rule of the Kaggle run).
                   Audio: audio_dataset_collector/audios + /segments. Text: the v3 drafts from the Kaggle results zips.
  Old calls        data/raw/audio (+ 59 calls whose audio is only in 'V4 fine tuning dataset/'). The 735 calls cut
                   into the old training clips come with those clips and their labels (data/final/dataset.jsonl).
                   Every old call also brings its whole-call text (data/final/calls/*.json and, for the Karnataka
                   calls, the transcripts of 'V4 fine tuning dataset/').

Audio files are HARD LINKS: the same file appears under labelling_data/audio/ and in its old place, using no extra disk
space. Deleting one of the two names leaves the other. The old folders and tools keep working unchanged.

Run it again whenever new transcripts arrive: put the Kaggle results zip (sessionN_results.zip) into
labelling_data/incoming/ and run. It links only what is missing, rewrites the lists, and never touches
labelling_data/verified/ (the labelling tool's output).

    python labelling_tool/build_data.py            build or update
    python labelling_tool/build_data.py --check    count only, change nothing
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "labelling_data"
V4 = ROOT / "V4 fine tuning dataset"
MIN_CALL_SECONDS = 30.0
CODE = {"bengali": "bn", "english": "en", "hindi": "hi", "kannada": "kn", "marathi": "mr", "telugu": "te"}
csv.field_size_limit(10 ** 9)

CALL_FIELDS = ["call_id", "source", "state", "language", "also_languages", "lead_id", "crm_call_id", "duration_s",
               "pieces", "pieces_with_text", "audio", "md5", "old_split", "has_call_text", "flags"]
PIECE_FIELDS = ["piece_id", "call_id", "source", "index", "count", "start_s", "end_s", "duration_s", "audio",
                "silent", "old_split", "flags"]


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def code(lang):
    lang = (lang or "").strip().lower()
    return CODE.get(lang, lang)


def rel(p):
    return Path(p).as_posix()


# -- the sources -----------------------------------------------------------------------------------------------------

def crm_calls_and_pieces():
    calls, pieces = [], []
    wanted = set()
    for c in rows(ROOT / "kaggle_upload/calls_kaggle.csv"):
        if float(c["source_duration_s"]) < MIN_CALL_SECONDS:
            continue
        cid = Path(c["filename"]).stem                         # <lead>_<call>_<state>
        wanted.add(c["source_path"])
        calls.append({"call_id": cid, "source": "crm", "state": c["state"], "lead_id": c["lead_id"],
                      "crm_call_id": c["call_id"], "duration_s": round(float(c["source_duration_s"]), 3),
                      "md5": c["source_md5"], "_src": ROOT / c["source_path"],
                      "audio": rel(Path("audio/calls/crm") / c["state"] / Path(c["source_path"]).name)})
    for p in rows(ROOT / "kaggle_upload/segments_manifest_kaggle.csv"):
        if p["source_path"] not in wanted:
            continue
        pieces.append({"piece_id": p["segment_id"], "call_id": Path(p["source_path"]).stem, "source": "crm",
                       "index": int(p["segment_index"]), "count": int(p["segment_count"]),
                       "start_s": float(p["start_s"]), "end_s": float(p["end_s"]),
                       "duration_s": float(p["duration_s"]), "silent": p["silent"], "old_split": "", "flags": "",
                       "_src": ROOT / p["segment_path"],
                       "audio": rel(Path("audio/pieces/crm") / p["state"] / Path(p["segment_path"]).name)})
    return calls, pieces


def old_calls_pieces_texts():
    fmap = {Path(r["path"]).stem: r for r in rows(ROOT / "data/manifests/file_map.csv")}
    cjson = {Path(p).stem: p for p in glob.glob(str(ROOT / "data/final/calls/*/*.json"))}
    v4 = {r["canonical_call_id"]: r for r in rows(V4 / "audio_manifest.csv")} if (V4 / "audio_manifest.csv").exists() else {}
    calls, texts = [], []
    for cid in sorted(set(fmap) | set(v4)):
        f, v, j = fmap.get(cid), v4.get(cid), None
        if cid in cjson:
            with open(cjson[cid], encoding="utf-8") as fh:
                j = json.load(fh)
        if f:
            src, lang, state = ROOT / f["path"], f["language"], f["region"]
            dur, md5, also = f["duration_s"], f["md5"], f["mixed_with"]
        else:                                                   # audio only in the V4 folder
            src, lang, state = V4 / v["relative_audio_path"], v["language"], "karnataka"
            dur, md5, also = v["duration_seconds"], v["md5"], ""
        flags = []
        name = (v or {}).get("relative_audio_path", "")
        if "x1.25" in name:
            flags.append("file name says x1.25 (maybe sped up)")
        words = len(((j or {}).get("text_romanized") or "").split())
        minutes = float(dur or 0) / 60
        if j and minutes and words / minutes < 50:   # normal speech here is ~150 words a minute
            flags.append(f"whole-call text covers little of the audio ({words / minutes:.0f} words/min)")
        calls.append({"call_id": cid, "source": "old", "state": state, "language": code(lang),
                      "also_languages": " ".join(code(x) for x in (also or "").replace(";", ",").split(",") if x.strip()),
                      "lead_id": (v or {}).get("lead_id", ""), "crm_call_id": (v or {}).get("call_id", ""),
                      "duration_s": round(float(dur or 0), 3), "md5": md5, "old_split": (j or {}).get("split", ""),
                      "flags": "; ".join(flags), "_src": src,
                      "audio": rel(Path("audio/calls/old") / code(lang) / (cid + src.suffix))})
        t = {"call_id": cid}
        if j and (j.get("text_native") or j.get("text_romanized")):
            t.update(text_native=j.get("text_native", ""), text_romanized=j.get("text_romanized", ""))
        if v:
            t.update(v4_model=v.get("model", ""), v4_diarized_text=v.get("diarized_text", ""),
                     v4_english_text=v.get("english_text", ""), project=v.get("project", ""))
        if len(t) > 1:
            texts.append(t)
    excluded = {r["segment_id"]: r.get("reason", "excluded") for r in jsonl(ROOT / "data/final/_not_for_training/excluded.jsonl")}
    pieces = []
    for s in rows(ROOT / "data/manifests/segments.csv"):
        lang = code(s["language"])
        src = ROOT / s["audio_filepath"]
        pieces.append({"piece_id": s["segment_id"], "call_id": s["parent_call_id"], "source": "old",
                       "index": int(s["segment_index"]), "count": 0, "start_s": float(s["start_time_s"]),
                       "end_s": float(s["end_time_s"]), "duration_s": float(s["duration_s"]), "silent": "",
                       "old_split": s["split"], "flags": f"left out of training: {excluded[s['segment_id']]}"
                       if s["segment_id"] in excluded else "", "_src": src,
                       "audio": rel(Path("audio/pieces/old") / lang / src.name)})
    per_call = Counter(p["call_id"] for p in pieces)
    for p in pieces:
        p["count"] = per_call[p["call_id"]]
    drafts = []
    for r in jsonl(ROOT / "data/final/dataset.jsonl"):
        drafts.append({"piece_id": r["segment_id"], "text": r.get("text_native", ""),
                       "text_roman": r.get("text_romanized", ""), "language": code(r.get("language")),
                       "silent": False, "loop": False, "draft_by": "old label", "decoded_at": ""})
    return calls, pieces, texts, drafts


def crm_drafts(piece_ids):
    """v3 drafts: every results zip / transcripts jsonl in incoming/, plus the laptop's own run; one row per piece."""
    sources = sorted(glob.glob(str(OUT / "incoming/*.zip"))) + sorted(glob.glob(str(OUT / "incoming/*.jsonl")))
    laptop = ROOT / "audio_dataset_collector/segments/transcripts.jsonl"
    if laptop.exists():
        sources.append(str(laptop))
    best, read = {}, []
    for src in sources:
        if src.endswith(".zip"):
            with zipfile.ZipFile(src) as z:
                names = [n for n in z.namelist() if n.endswith("segments/transcripts.jsonl")]
                lines = [json.loads(l) for n in names for l in z.open(n) if l.strip()]
        else:
            lines = jsonl(src)
        read.append((Path(src).name, len(lines)))
        for r in lines:
            sid = r.get("segment_id")
            if sid not in piece_ids:
                continue
            old = best.get(sid)
            if old is None or ("text_roman" in r and "text_roman" not in old) or \
                    ("text_roman" in r) == ("text_roman" in old) and r.get("decoded_at", "") > old.get("decoded_at", ""):
                best[sid] = r
    drafts = [{"piece_id": sid, "text": r.get("text", ""), "text_roman": r.get("text_roman", ""),
               "language": r.get("language") or "", "silent": bool(r.get("silent")), "loop": bool(r.get("loop")),
               "draft_by": "v3 (crm-indic-asr 1.0.0)", "decoded_at": r.get("decoded_at", "")}
              for sid, r in best.items()]
    return drafts, read


# -- writing ---------------------------------------------------------------------------------------------------------

def link(src, dst):
    """Hard-link src to dst. -> 'new', 'kept' or 'missing'."""
    if not src.exists():
        return "missing"
    if dst.exists():
        if os.stat(dst).st_ino == os.stat(src).st_ino:
            return "kept"
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)
    return "new"


def write_csv(path, fields, items):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)
    os.replace(tmp, path)


def write_jsonl(path, items):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="count only, change nothing")
    args = ap.parse_args()
    t0 = time.time()
    (OUT / "incoming").mkdir(parents=True, exist_ok=True)

    crm_c, crm_p = crm_calls_and_pieces()
    old_c, old_p, texts, old_d = old_calls_pieces_texts()
    gone = [x for x in crm_c + old_c + crm_p + old_p if not Path(x["_src"]).exists()]
    if gone:   # e.g. old clips deleted when they were left out of training: listed in the README, not in the lists
        print("audio missing on disk, left out:", [x.get("piece_id") or x["call_id"] for x in gone])
    gone_ids = {id(x) for x in gone}
    crm_c, old_c, crm_p, old_p = ([x for x in xs if id(x) not in gone_ids] for xs in (crm_c, old_c, crm_p, old_p))
    calls, pieces = crm_c + old_c, crm_p + old_p
    piece_ids = {p["piece_id"] for p in pieces}
    v3_d, read = crm_drafts({p["piece_id"] for p in crm_p})
    drafts = v3_d + [d for d in old_d if d["piece_id"] in piece_ids]
    have = {d["piece_id"]: d for d in drafts}

    # per call: pieces, pieces with text, language of the CRM calls from their drafts
    by_call = defaultdict(list)
    for p in pieces:
        by_call[p["call_id"]].append(p)
    has_call_text = {t["call_id"] for t in texts}
    for c in calls:
        ps = by_call.get(c["call_id"], [])
        c["pieces"] = len(ps)
        c["pieces_with_text"] = sum(p["piece_id"] in have for p in ps)
        c["has_call_text"] = "yes" if c["call_id"] in has_call_text else ""
        if c["source"] == "crm":
            langs = Counter(have[p["piece_id"]]["language"] for p in ps if p["piece_id"] in have and have[p["piece_id"]]["language"])
            c["language"] = langs.most_common(1)[0][0] if langs else ""
            c["also_languages"] = " ".join(l for l, _ in langs.most_common()[1:])
            if not c["pieces_with_text"]:
                c["flags"] = "no draft yet"
        elif not ps:
            c["flags"] = "; ".join(x for x in (c["flags"], "not cut into pieces yet") if x)

    # summary
    summary = defaultdict(lambda: Counter())
    for p in pieces:
        key =p["source"] + (" " + p["call_id"].rsplit("_", 1)[-1] if p["source"] == "crm" else "")
        summary[key]["pieces"] += 1
        summary[key]["hours"] += p["duration_s"] / 3600
        if p["piece_id"] in have:
            summary[key]["pieces_with_text"] += 1
            summary[key]["hours_with_text"] += p["duration_s"] / 3600
    for c in calls:
        key = c["source"] + (" " + c["state"] if c["source"] == "crm" else "")
        summary[key]["calls"] += 1
        summary[key]["call_hours"] += c["duration_s"] / 3600
        if c["source"] == "old" and not c["pieces"]:
            summary[key]["calls_not_cut"] += 1
            summary[key]["hours_not_cut"] += c["duration_s"] / 3600

    print("draft sources read:", read)
    for k in sorted(summary):
        s = summary[k]
        print(f"{k:18} calls {s['calls']:6,} ({s['call_hours']:6.1f} h) | pieces {s['pieces']:6,} ({s['hours']:6.1f} h) | "
              f"with text {s['pieces_with_text']:6,} ({s['hours_with_text']:6.1f} h)"
              + (f" | calls not cut yet {s['calls_not_cut']:,} ({s['hours_not_cut']:.1f} h)" if s["calls_not_cut"] else ""))
    missing_audio = [x.get("piece_id") or x["call_id"] for x in gone]
    if args.check:
        print("check only: nothing written")
        return

    done = Counter()
    for i, x in enumerate(calls + pieces, 1):
        done[link(Path(x["_src"]), OUT / x["audio"])] += 1
        if i % 10000 == 0:
            print(f"  linked {i:,} of {len(calls) + len(pieces):,} files", flush=True)
    (OUT / "verified").mkdir(exist_ok=True)
    write_csv(OUT / "calls.csv", CALL_FIELDS, sorted(calls, key=lambda c: (c["source"], c["state"], c["call_id"])))
    write_csv(OUT / "pieces.csv", PIECE_FIELDS, sorted(pieces, key=lambda p: (p["source"], p["call_id"], p["index"])))
    write_jsonl(OUT / "drafts.jsonl", sorted(drafts, key=lambda d: d["piece_id"]))
    write_jsonl(OUT / "call_texts.jsonl", texts)
    write_readme(summary, read, done, missing_audio)
    print(f"audio links: {dict(done)}; lists written; {time.time() - t0:.0f} s")


def write_readme(summary, read, done, missing):
    lines = [
        "# labelling_data: every call, piece and transcript in one folder",
        "",
        f"Built by `labelling_tool/build_data.py` on {time.strftime('%Y-%m-%d %H:%M')}. Run it again after adding a "
        "Kaggle results zip to `incoming/`. It never changes `verified/`.",
        "",
        "| What | Calls | Call audio | Pieces | Piece audio | Pieces with text |",
        "|---|---|---|---|---|---|",
    ]
    for k in sorted(summary):
        s = summary[k]
        lines.append(f"| {k} | {s['calls']:,} | {s['call_hours']:.1f} h | {s['pieces']:,} | {s['hours']:.1f} h | "
                     f"{s['pieces_with_text']:,} ({s['hours_with_text']:.1f} h) |")
    lines += [
        "",
        "- `calls.csv`: one row per call (old and CRM).",
        "  - `flags` says what is still missing, for example *no draft yet* or *not cut into pieces yet*.",
        "  - CRM calls shorter than 30 s are left out.",
        "- `pieces.csv`: one row per piece, with its start and end inside its call.",
        "  - `old_split` is the old train/test split, for reference only: the new split is made by lead after labelling.",
        "- `drafts.jsonl`: the text to check for each piece: `text` (native) and `text_roman`.",
        "  - `draft_by` is *v3 (crm-indic-asr 1.0.0)* for the CRM pieces and *old label* for the old training clips.",
        "- `call_texts.jsonl`: the whole-call text of the old calls (`text_native`, `text_romanized`).",
        "  - The Karnataka calls also carry the V4 folder's `v4_diarized_text` and `v4_english_text`.",
        "  - Shown as reference while checking pieces; they can't be split into pieces exactly.",
        "- `audio/`: hard links to the audio files in `audio_dataset_collector/`, `data/` and `V4 fine tuning dataset/`.",
        "  - They are the same files, so they use no extra disk space. Deleting one name leaves the other.",
        "- `incoming/`: Kaggle results zips. Every `segments/transcripts.jsonl` inside them is read.",
        "- `verified/`: written only by the labelling tool.",
        "",
        f"Draft sources read: {', '.join(f'{n} ({c:,} rows)' for n, c in read)}.",
        f"Audio links this run: {dict(done)}.",
        f"Left out because their audio is not on disk: {', '.join(missing) or 'none'}.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
