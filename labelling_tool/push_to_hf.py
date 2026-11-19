"""Push the CRM ASR dataset (old + new calls, every piece's text and its status) to a PRIVATE Hugging Face dataset.

The dataset holds each call's audio ONCE plus every piece's start and end inside its call (pieces are cut again when
training), so it is about 11 GB instead of 30+. Text comes from the labelling tool's database, so verified work is
included as soon as it is verified. Every piece says where its text came from:
    label_status  verified | v3_draft | old_label | none

    python labelling_tool/push_to_hf.py --stage            build labelling_data/hf_dataset/ (hard links, no extra disk)
    python labelling_tool/push_to_hf.py --upload           stage, then upload (resumable: run again after a break)

The token is read from .env: HF_WRITE_TOKEN if set, else HF_TOKEN. It needs write access to the repo's owner.
"""
import argparse
import csv
import json
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "labelling_data"
STAGE = DATA / "hf_dataset"
REPO = "brickfoliosolutions/crm-asr-dataset-v4"
csv.field_size_limit(10 ** 9)


def env_token():
    if os.environ.get("HF_WRITE_TOKEN"):
        return os.environ["HF_WRITE_TOKEN"]
    vals = {}
    for line in open(ROOT / ".env", encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals.get("HF_WRITE_TOKEN") or vals.get("HF_TOKEN")


def link(src, dst):
    if dst.exists():
        if os.stat(dst).st_ino == os.stat(src).st_ino:
            return 0
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)
    return 1


def stage():
    db = sqlite3.connect(f"file:{DATA / 'labelling.db'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    calls = list(csv.DictReader(open(DATA / "calls.csv", encoding="utf-8")))
    new_links = 0
    for c in calls:                                   # audio/calls/<crm|old>/<state or language>/<file>
        rel = Path(c["audio"]).relative_to("audio")
        new_links += link(DATA / c["audio"], STAGE / "audio" / rel)
        c["audio"] = (Path("audio") / rel).as_posix()
    (STAGE / "metadata").mkdir(parents=True, exist_ok=True)
    with open(STAGE / "metadata/calls.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(calls[0]))
        w.writeheader()
        w.writerows(calls)
    audio_of = {c["call_id"]: c["audio"] for c in calls}

    status = Counter()
    hours = Counter()
    with open(STAGE / "metadata/pieces.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for p in db.execute("select * from pieces order by source, call_id, piece_index"):
            if p["status"] == "verified":
                st = "verified"
            elif p["has_draft"]:
                st = "v3_draft" if (p["draft_by"] or "").startswith("v3") else "old_label"
            else:
                st = "none"
            status[st] += 1
            hours[st] += (p["duration_s"] or 0) / 3600
            f.write(json.dumps({
                "piece_id": p["piece_id"], "call_id": p["call_id"], "source": p["source"],
                "call_audio": audio_of.get(p["call_id"], ""), "index": p["piece_index"], "count": p["piece_count"],
                "start_s": p["start_s"], "end_s": p["end_s"], "duration_s": p["duration_s"],
                "label_status": st,
                "text": p["current_text"] or "", "text_roman": p["current_text_roman"] or "",
                "language": p["main_language"] or p["draft_language"] or "",
                "also_languages": p["also_languages"] or "",
                "silent": bool(p["silent"] in (1, "1", "True", "true") or p["draft_silent"]),
                "draft_text": p["draft_text"] or "", "draft_text_roman": p["draft_text_roman"] or "",
                "draft_by": p["draft_by"] or "", "human_edited": bool(p["human_edited"]),
                "verified_by": p["verified_by"] or "", "verified_at": p["verified_at"] or "",
                "edges_edited": bool(p["edges_edited"]), "old_split": p["old_split"] or "", "flags": p["flags"] or "",
            }, ensure_ascii=False) + "\n")
    with open(STAGE / "metadata/call_texts.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for line in open(DATA / "call_texts.jsonl", encoding="utf-8"):
            f.write(line)

    by_src = Counter(c["source"] for c in calls)
    size = sum((STAGE / c["audio"]).stat().st_size for c in calls)
    write_card(calls, status, hours, by_src, size)
    print(f"staged {len(calls):,} calls ({size / 1e9:.1f} GB audio, {new_links:,} new links) | pieces by status: "
          + ", ".join(f"{k} {v:,} ({hours[k]:.1f} h)" for k, v in status.most_common()))
    return size


def write_card(calls, status, hours, by_src, size):
    states = Counter((c["source"], c["state"]) for c in calls)
    rows = "\n".join(f"| {s} | {st} | {n:,} |" for (s, st), n in sorted(states.items()))
    st_rows = "\n".join(f"| `{k}` | {v:,} | {hours[k]:.1f} h |" for k, v in status.most_common())
    card = f"""---
license: other
pretty_name: CRM call ASR dataset (v4)
language: [hi, en, kn, te, mr, bn]
task_categories: [automatic-speech-recognition]
---
# CRM call ASR dataset (v4), PRIVATE

Customer calls of Brickfolio's CRM with a native-script and a romanized transcript for every piece. **Contains customer
voices, names and phone numbers: keep this repository private.**

Updated {time.strftime('%Y-%m-%d %H:%M')} from the labelling tool (`labelling_tool/push_to_hf.py`).
{len(calls):,} calls ({by_src['crm']:,} new CRM calls, {by_src['old']:,} old calls), {size / 1e9:.1f} GB of audio.

## Files
- `audio/calls/<crm|old>/<state or language>/...`: each call's audio file, once, exactly as downloaded.
- `metadata/pieces.jsonl`: one row per piece.
  - `call_audio`, `start_s`, `end_s`: cut the piece from its call (decode to 16 kHz mono, slice by time).
  - `text`, `text_roman`, `language`, `also_languages`: the current text.
  - `label_status`: where that text comes from (see below).
  - `draft_text`, `draft_text_roman`, `draft_by`: the model's or the old label's original text, never edited.
  - `verified_by`, `verified_at`, `human_edited`: who verified it, when, and whether they changed it.
- `metadata/calls.csv`: one row per call (state, language, lead, length, md5, flags).
- `metadata/call_texts.jsonl`: whole-call texts of the old calls (no times; reference only).

## How far each piece's text can be trusted
| label_status | pieces | audio |
|---|---|---|
{st_rows}

- `verified`: checked by ear and corrected in the labelling tool. **Train and test on these.**
- `old_label`: the v1–v3 training labels (Gemini / hand-made, audited 2026-10-31).
- `v3_draft`: the v3 model's own output, not checked. About 12% of its words were wrong in the first checked Hindi
  pieces. Don't use these as test data.
- `none`: no text yet.

## Calls
| source | state / region | calls |
|---|---|---|
{rows}

CRM calls shorter than 30 s are not included. Split train/test **by lead** (all calls of one lead on one side) only
after labelling.
"""
    (STAGE / "README.md").write_text(card, encoding="utf-8", newline="\n")


def upload(repo):
    from huggingface_hub import HfApi
    api = HfApi(token=env_token())
    who = api.whoami()["name"]
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo, repo_type="dataset")
    assert info.private, f"{repo} is not private: stopping"
    print(f"uploading as {who} to private dataset {repo} (resumable; run again if it stops)")
    from huggingface_hub import CommitOperationAdd
    remote = set(api.list_repo_files(repo, repo_type="dataset"))
    audio = sorted(p for p in STAGE.rglob("*") if p.is_file() and p.relative_to(STAGE).parts[0] == "audio")
    todo = [p for p in audio if p.relative_to(STAGE).as_posix() not in remote]
    print(f"audio files: {len(audio):,}, already uploaded {len(audio) - len(todo):,}, to upload {len(todo):,}")
    BATCH = 500
    t0 = time.time()
    done_bytes = 0
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        api.create_commit(repo, repo_type="dataset",
                          operations=[CommitOperationAdd(p.relative_to(STAGE).as_posix(), str(p)) for p in part],
                          commit_message=f"audio batch {i // BATCH + 1}")
        done_bytes += sum(p.stat().st_size for p in part)
        rate = done_bytes / max(time.time() - t0, 1)
        left = sum(p.stat().st_size for p in todo[i + BATCH:])
        print(f"  {min(i + BATCH, len(todo)):,} of {len(todo):,} files | {done_bytes / 1e9:.2f} GB at "
              f"{rate * 8 / 1e6:.1f} Mbit/s | about {left / max(rate, 1) / 3600:.1f} h left", flush=True)
    meta = [p for p in STAGE.rglob("*") if p.is_file() and p.relative_to(STAGE).parts[0] != "audio"
            and ".cache" not in p.parts]
    api.create_commit(repo, repo_type="dataset",
                      operations=[CommitOperationAdd(p.relative_to(STAGE).as_posix(), str(p)) for p in meta],
                      commit_message="metadata and dataset card")
    print("upload finished: audio + metadata + README")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    if not (a.stage or a.upload):
        ap.error("use --stage or --upload")
    stage()
    if a.upload:
        upload(a.repo)
