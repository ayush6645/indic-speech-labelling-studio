"""Comprehensive unit & API integration tests for labelling_tool (uses isolated temp audio & DB)."""
import csv
import gzip
import io
import json
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labelling_tool import audio_utils, auth, config, db, exporter, importer, metrics, server, splitter  # noqa: E402


def _make_speech_like_signal(duration_s: float = 12.0, sr: int = 16000, seed: int = 1) -> np.ndarray:
    """Create a speech-like amplitude-modulated multi-tone signal so RMS envelopes are uniquely identifiable."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    env = np.zeros_like(t)
    # Random syllabic bursts & pauses
    pos = 0
    while pos < len(t):
        dur = int(rng.integers(int(0.15 * sr), int(0.8 * sr)))
        amp = float(rng.choice([0.02, 0.25, 0.55, 0.7]))
        env[pos : pos + dur] = amp
        pos += dur
    carrier = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 510 * t)
        + 0.2 * np.sin(2 * np.pi * 890 * t)
    ).astype(np.float32)
    return np.clip(env * carrier, -0.9, 0.9)


@pytest.fixture()
def lab_env(tmp_path, monkeypatch):
    """Build an isolated labelling_data/ workspace with real audio files & CSV/JSONL manifests."""
    data_dir = tmp_path / "labelling_data"
    audio_dir = data_dir / "audio"
    incoming_dir = data_dir / "incoming"
    verified_dir = data_dir / "verified"
    edited_dir = data_dir / "edited_audio"
    wf_dir = data_dir / ".waveform_cache"
    opus_dir = data_dir / ".opus_cache"
    backup_dir = data_dir / "backups"

    for d in (
        audio_dir / "calls" / "crm" / "bangalore",
        audio_dir / "calls" / "crm" / "kolkata",
        audio_dir / "calls" / "old" / "hi",
        audio_dir / "pieces" / "crm" / "bangalore",
        audio_dir / "pieces" / "crm" / "kolkata",
        audio_dir / "pieces" / "old" / "hi",
        incoming_dir,
        verified_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "AUDIO_DIR", audio_dir)
    monkeypatch.setattr(config, "INCOMING_DIR", incoming_dir)
    monkeypatch.setattr(config, "VERIFIED_DIR", verified_dir)
    monkeypatch.setattr(config, "EDITED_AUDIO_DIR", edited_dir)
    monkeypatch.setattr(config, "WAVEFORM_CACHE_DIR", wf_dir)
    monkeypatch.setattr(config, "OPUS_CACHE_DIR", opus_dir)
    monkeypatch.setattr(config, "BACKUP_DIR", backup_dir)
    monkeypatch.setattr(config, "DB_PATH", data_dir / "labelling.db")
    monkeypatch.setattr(config, "SPELLING_GUIDE_PATH", data_dir / "spelling_guide.md")
    monkeypatch.setattr(config, "CALLS_CSV", data_dir / "calls.csv")
    monkeypatch.setattr(config, "PIECES_CSV", data_dir / "pieces.csv")
    monkeypatch.setattr(config, "DRAFTS_JSONL", data_dir / "drafts.jsonl")
    monkeypatch.setattr(config, "CALL_TEXTS_JSONL", data_dir / "call_texts.jsonl")

    sr = 16000
    # Call 1 (CRM bangalore, 12s, 3 pieces of 4s each)
    sig1 = _make_speech_like_signal(12.0, sr, seed=101)
    c1_wav = tmp_path / "c1.wav"
    sf.write(str(c1_wav), sig1, sr)
    c1_mp3 = audio_dir / "calls" / "crm" / "bangalore" / "1001_501_bangalore.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(c1_wav), "-b:a", "64k", str(c1_mp3)], check=True)
    for idx, (a, b) in enumerate([(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)], 1):
        sf.write(
            str(audio_dir / "pieces" / "crm" / "bangalore" / f"1001_501_bangalore_s{idx:03d}.flac"),
            sig1[int(a * sr) : int(b * sr)],
            sr,
            format="FLAC",
            subtype="PCM_16",
        )

    # Call 2 (Old hindi, identical underlying signal to Call 4 CRM overlap, encoded differently as .m4a!)
    sig_dup = _make_speech_like_signal(15.0, sr, seed=202)
    dup_wav = tmp_path / "dup.wav"
    sf.write(str(dup_wav), sig_dup, sr)
    c2_m4a = audio_dir / "calls" / "old" / "hi" / "hi_ka_0001.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(dup_wav), "-c:a", "aac", "-b:a", "48k", str(c2_m4a)], check=True)
    c2_piece_wav = audio_dir / "pieces" / "old" / "hi" / "hi_ka_0001_s001.wav"
    sf.write(str(c2_piece_wav), sig_dup[: 7 * sr], sr)

    # Call 3 (CRM bangalore, DIFFERENT signal, same lead_id='1001' as Call 1!)
    sig3 = _make_speech_like_signal(10.0, sr, seed=303)
    c3_mp3 = audio_dir / "calls" / "crm" / "bangalore" / "1001_502_bangalore.mp3"
    sf.write(str(tmp_path / "c3.wav"), sig3, sr)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmp_path / "c3.wav"), "-b:a", "64k", str(c3_mp3)], check=True)
    sf.write(
        str(audio_dir / "pieces" / "crm" / "bangalore" / "1001_502_bangalore_s001.flac"),
        sig3,
        sr,
        format="FLAC",
        subtype="PCM_16",
    )

    # Call 4 (CRM kolkata, duplicate audio of Call 2 old with 0.2s leading silence -> different MD5!)
    padded = np.concatenate([np.zeros(int(0.2 * sr), dtype=np.float32), sig_dup])[: int(15.0 * sr)]
    sf.write(str(tmp_path / "c4.wav"), padded, sr)
    c4_mp3 = audio_dir / "calls" / "crm" / "kolkata" / "2002_601_kolkata.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmp_path / "c4.wav"), "-b:a", "64k", str(c4_mp3)], check=True)
    sf.write(
        str(audio_dir / "pieces" / "crm" / "kolkata" / "2002_601_kolkata_s001.flac"),
        padded[: 7 * sr],
        sr,
        format="FLAC",
        subtype="PCM_16",
    )

    # More independent CRM leads so stratified split has enough groups
    extra_calls = []
    extra_pieces = []
    extra_drafts = []
    for k in range(5):
        lead = str(3000 + k)
        cid = f"{lead}_70{k}_bangalore"
        s_x = _make_speech_like_signal(8.0, sr, seed=400 + k)
        sf.write(str(tmp_path / f"ex{k}.wav"), s_x, sr)
        mp3_x = audio_dir / "calls" / "crm" / "bangalore" / f"{cid}.mp3"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmp_path / f"ex{k}.wav"), str(mp3_x)], check=True)
        sf.write(
            str(audio_dir / "pieces" / "crm" / "bangalore" / f"{cid}_s001.flac"),
            s_x,
            sr,
            format="FLAC",
            subtype="PCM_16",
        )
        extra_calls.append({
            "call_id": cid, "source": "crm", "state": "bangalore", "language": "hi", "also_languages": "",
            "lead_id": lead, "crm_call_id": f"70{k}", "duration_s": "8.0", "pieces": "1",
            "pieces_with_text": "1", "audio": f"audio/calls/crm/bangalore/{cid}.mp3",
            "md5": f"md5ex{k}", "old_split": "", "has_call_text": "", "flags": "",
        })
        extra_pieces.append({
            "piece_id": f"{cid}_s001", "call_id": cid, "source": "crm", "index": "1", "count": "1",
            "start_s": "0.0", "end_s": "8.0", "duration_s": "8.0",
            "audio": f"audio/pieces/crm/bangalore/{cid}_s001.flac", "silent": "0", "old_split": "", "flags": "",
        })
        extra_drafts.append({
            "piece_id": f"{cid}_s001", "text": "नमस्ते सर जी", "text_roman": "namaste sir ji",
            "language": "hi", "silent": False, "loop": False, "draft_by": "v3 (crm-indic-asr 1.0.0)", "decoded_at": "2026-11-11",
        })

    calls = [
        {
            "call_id": "1001_501_bangalore", "source": "crm", "state": "bangalore", "language": "hi", "also_languages": "en",
            "lead_id": "1001", "crm_call_id": "501", "duration_s": "12.0", "pieces": "3", "pieces_with_text": "2",
            "audio": "audio/calls/crm/bangalore/1001_501_bangalore.mp3", "md5": "m1", "old_split": "", "has_call_text": "", "flags": "",
        },
        {
            "call_id": "hi_ka_0001", "source": "old", "state": "karnataka", "language": "hi", "also_languages": "",
            "lead_id": "", "crm_call_id": "", "duration_s": "15.0", "pieces": "1", "pieces_with_text": "1",
            "audio": "audio/calls/old/hi/hi_ka_0001.m4a", "md5": "m2_old", "old_split": "test", "has_call_text": "yes", "flags": "",
        },
        {
            "call_id": "1001_502_bangalore", "source": "crm", "state": "bangalore", "language": "hi", "also_languages": "",
            "lead_id": "1001", "crm_call_id": "502", "duration_s": "10.0", "pieces": "1", "pieces_with_text": "1",
            "audio": "audio/calls/crm/bangalore/1001_502_bangalore.mp3", "md5": "m3", "old_split": "", "has_call_text": "", "flags": "",
        },
        {
            "call_id": "2002_601_kolkata", "source": "crm", "state": "kolkata", "language": "bn", "also_languages": "",
            "lead_id": "2002", "crm_call_id": "601", "duration_s": "15.0", "pieces": "1", "pieces_with_text": "1",
            "audio": "audio/calls/crm/kolkata/2002_601_kolkata.mp3", "md5": "m4_crm_different", "old_split": "", "has_call_text": "", "flags": "",
        },
        *extra_calls,
    ]
    pieces = [
        {
            "piece_id": "1001_501_bangalore_s001", "call_id": "1001_501_bangalore", "source": "crm",
            "index": "1", "count": "3", "start_s": "0.0", "end_s": "4.0", "duration_s": "4.0",
            "audio": "audio/pieces/crm/bangalore/1001_501_bangalore_s001.flac", "silent": "0", "old_split": "", "flags": "",
        },
        {
            "piece_id": "1001_501_bangalore_s002", "call_id": "1001_501_bangalore", "source": "crm",
            "index": "2", "count": "3", "start_s": "4.0", "end_s": "8.0", "duration_s": "4.0",
            "audio": "audio/pieces/crm/bangalore/1001_501_bangalore_s002.flac", "silent": "0", "old_split": "", "flags": "",
        },
        {
            "piece_id": "1001_501_bangalore_s003", "call_id": "1001_501_bangalore", "source": "crm",
            "index": "3", "count": "3", "start_s": "8.0", "end_s": "12.0", "duration_s": "4.0",
            "audio": "audio/pieces/crm/bangalore/1001_501_bangalore_s003.flac", "silent": "0", "old_split": "", "flags": "",
        },
        {
            "piece_id": "hi_ka_0001_s001", "call_id": "hi_ka_0001", "source": "old",
            "index": "1", "count": "1", "start_s": "0.0", "end_s": "7.0", "duration_s": "7.0",
            "audio": "audio/pieces/old/hi/hi_ka_0001_s001.wav", "silent": "", "old_split": "test", "flags": "",
        },
        {
            "piece_id": "1001_502_bangalore_s001", "call_id": "1001_502_bangalore", "source": "crm",
            "index": "1", "count": "1", "start_s": "0.0", "end_s": "10.0", "duration_s": "10.0",
            "audio": "audio/pieces/crm/bangalore/1001_502_bangalore_s001.flac", "silent": "0", "old_split": "", "flags": "",
        },
        {
            "piece_id": "2002_601_kolkata_s001", "call_id": "2002_601_kolkata", "source": "crm",
            "index": "1", "count": "1", "start_s": "0.0", "end_s": "7.0", "duration_s": "7.0",
            "audio": "audio/pieces/crm/kolkata/2002_601_kolkata_s001.flac", "silent": "0", "old_split": "", "flags": "",
        },
        *extra_pieces,
    ]
    drafts = [
        {
            "piece_id": "1001_501_bangalore_s001", "text": "हैलो सर गुड मॉर्निंग", "text_roman": "hello sir good morning",
            "language": "hi", "silent": False, "loop": False, "draft_by": "v3 (crm-indic-asr 1.0.0)", "decoded_at": "2026-11-11",
        },
        {
            "piece_id": "1001_501_bangalore_s002", "text": "तीन बीएचके प्रोजेक्ट है", "text_roman": "3 bhk project hai",
            "language": "hi", "silent": False, "loop": False, "draft_by": "v3 (crm-indic-asr 1.0.0)", "decoded_at": "2026-11-11",
        },
        {
            "piece_id": "hi_ka_0001_s001", "text": "ठीक है सर", "text_roman": "thik hai sir",
            "language": "hi", "silent": False, "loop": False, "draft_by": "old label", "decoded_at": "",
        },
        {
            "piece_id": "1001_502_bangalore_s001", "text": "बिल्कुल ठीक", "text_roman": "bilkul thik",
            "language": "hi", "silent": False, "loop": False, "draft_by": "v3 (crm-indic-asr 1.0.0)", "decoded_at": "2026-11-11",
        },
        {
            "piece_id": "2002_601_kolkata_s001", "text": "হ্যালো স্যার", "text_roman": "hello sir",
            "language": "bn", "silent": False, "loop": False, "draft_by": "v3 (crm-indic-asr 1.0.0)", "decoded_at": "2026-11-11",
        },
        *extra_drafts,
    ]
    call_texts = [
        {
            "call_id": "hi_ka_0001",
            "text_native": "ठीक है सर पूरा कॉल यहाँ है",
            "text_romanized": "thik hai sir pura call yahan hai",
            "v4_model": "v4",
            "v4_diarized_text": "Agent: ठीक है सर",
            "v4_english_text": "Okay sir",
            "project": "TestProject",
        }
    ]

    with open(config.CALLS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(calls[0].keys()))
        w.writeheader()
        w.writerows(calls)
    with open(config.PIECES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(pieces[0].keys()))
        w.writeheader()
        w.writerows(pieces)
    with open(config.DRAFTS_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(config.CALL_TEXTS_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for ct in call_texts:
            f.write(json.dumps(ct, ensure_ascii=False) + "\n")

    app = server.create_app(local_user="gyan", auto_sync=True)
    return {"app": app, "data_dir": data_dir, "tmp_path": tmp_path}


def test_metrics_norm_diff_and_wer():
    assert metrics.norm("हैलो, सर! Good—morning।") == "हैलो सर good morning"
    # Keeps Indic vowel signs & virama intact
    assert metrics.norm("ব্রিকফোলিও।") == "ব্রিকফোলিও"
    diffs = metrics.word_diff("hello sir good morning", "hello madam good morning today")
    assert len(diffs) == 2
    assert diffs[0]["op"] == "replace" and diffs[0]["old"] == "sir" and diffs[0]["new"] == "madam"
    assert diffs[1]["op"] == "insert" and diffs[1]["new"] == "today"
    wer, edits, ref_n = metrics.compute_wer("एक दो तीन चार", "एक दो पांच चार")
    assert edits == 1 and ref_n == 4 and wer == pytest.approx(0.25)


def _local_client(app):
    return TestClient(app, client=("127.0.0.1", 54321), headers={"host": "127.0.0.1:8092"})


def test_auth_strict_local_vs_remote_headers(lab_env):
    with _local_client(lab_env["app"]) as client:
        # True localhost auto-logins as local admin
        me = client.get("/api/me").json()
        assert me["authenticated"] is True
        assert me["is_local"] is True
        assert me["default_audio_fmt"] == "raw"

        # Ngrok / proxy headers MUST force 401 on protected endpoints even if TCP came from 127.0.0.1!
        for spoofs in [
            {"host": "demo123.ngrok-free.app"},
            {"host": "127.0.0.1:8092", "x-forwarded-for": "8.8.8.8"},
            {"host": "127.0.0.1:8092", "ngrok-trace-id": "trace-xyz"},
        ]:
            assert client.get("/api/calls", headers=spoofs).status_code == 401
            assert client.get("/api/audio/piece/1001_501_bangalore_s001", headers=spoofs).status_code == 401

        # Admin creates a remote reviewer
        res = client.post("/api/users", json={
            "username": "alice", "password": "secret123", "display_name": "Alice R", "role": "reviewer",
        })
        assert res.status_code == 200

    # Even if a remote IP spoofs Host: 127.0.0.1, client IP check blocks it!
    with TestClient(lab_env["app"], client=("203.0.113.5", 54321), headers={"host": "127.0.0.1:8092"}) as ext:
        assert ext.get("/api/calls").status_code == 401

    # New unauthenticated remote client (ngrok host)
    with TestClient(lab_env["app"], client=("127.0.0.1", 54321), headers={"host": "demo.ngrok-free.app"}) as remote:
        assert remote.get("/api/me").json()["authenticated"] is False
        assert remote.get("/api/calls").status_code == 401
        assert remote.post("/api/login", json={"username": "alice", "password": "wrong"}).status_code == 401

        login = remote.post("/api/login", json={"username": "alice", "password": "secret123"})
        assert login.status_code == 200
        assert login.json()["default_audio_fmt"] == "opus"

        # Authenticated reviewer works, sees opus default, cannot access admin users list
        assert remote.get("/api/calls").status_code == 200
        assert remote.get("/api/users").status_code == 403

        # Remote auto audio format streams compact Opus
        aud_r = remote.get("/api/audio/piece/1001_501_bangalore_s001")
        assert aud_r.status_code == 200
        assert aud_r.headers["content-type"].startswith("audio/ogg")
        assert aud_r.headers["accept-ranges"] == "bytes"
        opus_bytes = len(aud_r.content)

    # Disabling user revokes active cookie immediately
    with _local_client(lab_env["app"]) as admin_c:
        admin_c.post("/api/users", json={"username": "alice", "active": False})
        # Raw FLAC is larger than 24kbps Opus
        raw_r = admin_c.get("/api/audio/piece/1001_501_bangalore_s001")
        assert raw_r.headers["content-type"] == "audio/flac"
        assert len(raw_r.content) > opus_bytes

        # HTTP 206 Range test
        rng_r = admin_c.get("/api/audio/piece/1001_501_bangalore_s001", headers={"range": "bytes=0-255"})
        assert rng_r.status_code == 206
        assert rng_r.headers["content-range"].startswith("bytes 0-255/")
        assert len(rng_r.content) == 256


def test_save_verify_unverify_undo_export_and_progress(lab_env):
    with _local_client(lab_env["app"]) as client:
        # Edit Native -> Roman marked needs_update_script='text_roman'
        s_res = client.post("/api/pieces/1001_501_bangalore_s001/save", json={
            "current_text": "हैलो सर गुड ईवनिंग",
            "needs_update_script": "text_roman",
        })
        assert s_res.status_code == 200
        assert s_res.json()["piece"]["needs_update_script"] == "text_roman"

        # Undo text edit
        u_res = client.post("/api/pieces/1001_501_bangalore_s001/undo", json={})
        assert u_res.json()["undone"] is True
        assert u_res.json()["piece"]["current_text"] == "हैलो सर गुड मॉर्निंग"

        # Verify piece with modified Roman word + extra language 'en' (1 word out of 4 changed -> 25% WER roman!)
        v_res = client.post("/api/pieces/1001_501_bangalore_s001/verify", json={
            "current_text": "हैलो सर गुड मॉर्निंग",
            "current_text_roman": "hello sir good evening",
            "main_language": "hi",
            "also_languages": ["en", "hi"],  # 'hi' duplicate/main should strip automatically
        })
        assert v_res.status_code == 200
        vj = v_res.json()
        assert vj["piece"]["status"] == "verified"
        assert vj["piece"]["also_languages"] == ["en"]
        assert vj["next_unverified_piece_id"] == "1001_501_bangalore_s002"

        # Verified folder must immediately contain hard-linked audio, segment json, call json, dataset.jsonl
        vdir = config.VERIFIED_DIR
        ds = [json.loads(l) for l in (vdir / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(ds) == 1
        rec = ds[0]
        assert rec["segment_id"] == "1001_501_bangalore_s001"
        assert rec["language"] == "hindi"
        assert rec["also_languages"] == ["english"]
        assert rec["text_romanized"] == "hello sir good evening"
        assert rec["draft_text_native"] == "हैलो सर गुड मॉर्निंग"
        assert rec["draft_text_romanized"] == "hello sir good morning"
        assert rec["draft_language"] == "hindi"
        assert rec["human_edited"] is True
        assert rec["source"]["parent_call_id"] == "1001_501_bangalore"
        assert (vdir / rec["audio_filepath"]).is_file()
        seg_json_path = vdir / "segments" / "hindi" / "1001_501_bangalore_s001.json"
        assert seg_json_path.is_file()
        seg_json = json.loads(seg_json_path.read_text(encoding="utf-8"))
        assert seg_json["draft_text_romanized"] == "hello sir good morning"
        call_export = json.loads((vdir / "calls" / "hindi" / "1001_501_bangalore.json").read_text(encoding="utf-8"))
        assert call_export["complete"] is False
        assert call_export["verified_count"] == 1
        assert call_export["segments"][0]["draft_text_romanized"] == "hello sir good morning"
        assert call_export["segments"][0]["text_romanized"] == "hello sir good evening"

        # Verify remaining 2 pieces -> call becomes complete = True
        client.post("/api/pieces/1001_501_bangalore_s002/verify", json={})
        client.post("/api/pieces/1001_501_bangalore_s003/verify", json={
            "current_text": "ठीक है", "current_text_roman": "thik hai", "main_language": "hi",
        })
        call_export2 = json.loads((vdir / "calls" / "hindi" / "1001_501_bangalore.json").read_text(encoding="utf-8"))
        assert call_export2["complete"] is True
        assert call_export2["verified_count"] == 3

        # Progress & WER endpoint reports exact v3 WER on verified v3 pieces
        prog = client.get("/api/progress").json()
        hi_row = next(r for r in prog["by_lang"] if r["language"] == "hi")
        assert hi_row["v3_eval_pieces"] == 2  # s001 & s002 had v3 drafts; s003 had no draft
        assert hi_row["v3_wer_native"] == 0.0
        # s001 changed 1/4 words + s002 changed 0/4 words = 1/8 = 12.5%
        assert hi_row["v3_wer_roman"] == pytest.approx(12.5)

        # Changing verified piece language moves files cleanly & removes old lang dir copy
        client.post("/api/pieces/1001_501_bangalore_s001/save", json={"main_language": "bn"})
        assert not (vdir / "segments" / "hindi" / "1001_501_bangalore_s001.json").exists()
        assert (vdir / "segments" / "bengali" / "1001_501_bangalore_s001.json").is_file()

        # /api/calls reports per-language verified_by_lang across all languages cleanly
        calls_res = client.get("/api/calls?status=all").json()
        vbl = {x["language"]: x for x in calls_res["verified_by_lang"]}
        assert vbl["hi"]["verified_pieces"] == 2
        assert vbl["bn"]["verified_pieces"] == 1
        assert vbl["bn"]["verified_seconds"] == pytest.approx(4.0, abs=0.2)

        # Undo verification prompts once before removing work
        undo_ask = client.post("/api/pieces/1001_501_bangalore_s001/undo", json={})
        # Note: latest action was save(lang=bn), undo that first, then next undo hits verify -> needs_confirm
        assert undo_ask.json()["undone"] is True
        undo_ask2 = client.post("/api/pieces/1001_501_bangalore_s001/undo", json={})
        assert undo_ask2.json()["needs_confirm"] is True
        assert undo_ask2.json()["message"] == "Mark as unverified?"

        undo_done = client.post("/api/pieces/1001_501_bangalore_s001/undo", json={
            "history_id": undo_ask2.json()["history_id"], "confirm_unverify": True,
        })
        assert undo_done.json()["undone"] is True
        assert undo_done.json()["piece"]["status"] == "unverified"
        assert not (vdir / "segments" / "hindi" / "1001_501_bangalore_s001.json").exists()

        # History endpoint lists draft + diffs
        hist = client.get("/api/pieces/1001_501_bangalore_s001/history").json()
        assert hist["draft"]["draft_by"] == "v3 (crm-indic-asr 1.0.0)"
        assert len(hist["revisions"]) >= 2

        # Daily hot backup creates readable SQLite DB
        bk = client.post("/api/backup").json()["backup_path"]
        assert Path(bk).is_file()
        c_bk = sqlite3.connect(bk)
        assert c_bk.execute("SELECT COUNT(*) FROM pieces").fetchone()[0] > 0
        c_bk.close()


def test_edge_fix_continuous_split_merge_undo_and_sync(lab_env):
    with _local_client(lab_env["app"]) as client:
        # 1. Verify s001 cannot collide with s002 when s001 is unverified: move s001 end +1.0s -> [0..5], [5..8]
        mv = client.post("/api/pieces/1001_501_bangalore_s001/move-edge", json={"side": "end", "time_s": 5.0})
        assert mv.status_code == 200
        ps = mv.json()["pieces"]
        assert ps[0]["end_s"] == pytest.approx(5.0) and ps[1]["start_s"] == pytest.approx(5.0)
        assert ps[0]["edited_audio_rel"] == "edited_audio/1001_501_bangalore_s001.flac"
        assert (config.DATA_DIR / ps[0]["edited_audio_rel"]).is_file()
        # Original audio duration must match 5.0s cleanly
        info = sf.info(str(config.DATA_DIR / ps[0]["edited_audio_rel"]))
        assert info.duration == pytest.approx(5.0, abs=0.05)

        # 2. Split s001 at 2.5s -> keeps text in s001, creates s001_p2, preserves s002 & s003 IDs!
        sp = client.post("/api/pieces/1001_501_bangalore_s001/split", json={"time_s": 2.5})
        assert sp.status_code == 200
        ps = sp.json()["pieces"]
        assert len(ps) == 4
        assert [p["piece_id"] for p in ps] == [
            "1001_501_bangalore_s001",
            "1001_501_bangalore_s001_p2",
            "1001_501_bangalore_s002",
            "1001_501_bangalore_s003",
        ]
        assert ps[0]["current_text"] == "हैलो सर गुड मॉर्निंग" and ps[1]["current_text"] == ""
        assert [p["piece_index"] for p in ps] == [1, 2, 3, 4]

        # 3. Undo split cleanly restores 3 pieces & removes _p2
        un_sp = client.post("/api/pieces/1001_501_bangalore_s001/undo", json={}).json()
        assert un_sp["edge_restored"] is True
        assert len(un_sp["pieces"]) == 3

        # 4. Merge s001 with s002 -> s002 deleted and recorded in deleted_pieces
        mg = client.post("/api/pieces/1001_501_bangalore_s001/merge").json()
        assert len(mg["pieces"]) == 2
        assert mg["pieces"][0]["end_s"] == pytest.approx(8.0)
        assert "तीन बीएचके" in mg["pieces"][0]["current_text"]

        # Syncing CSV again MUST NOT resurrect deleted piece s002 or clobber edge edits!
        sync_res = db.sync_from_labelling_data(force=True)
        assert sync_res["synced"] is True
        detail = client.get("/api/call/1001_501_bangalore").json()
        assert len(detail["pieces"]) == 2

        # Undo merge resurrects s002 properly
        un_mg = client.post("/api/pieces/1001_501_bangalore_s001/undo", json={}).json()
        assert un_mg["edge_restored"] is True
        assert len(un_mg["pieces"]) == 3

        # 5. Edge conflict protection if neighbor verified
        client.post("/api/pieces/1001_501_bangalore_s002/verify", json={})
        conflict = client.post("/api/pieces/1001_501_bangalore_s001/move-edge", json={"side": "end", "time_s": 5.5})
        assert conflict.status_code == 409


def test_dynamic_session_import_preserves_verified(lab_env):
    with _local_client(lab_env["app"]) as client:
        # Mark s001 verified first with human label
        client.post("/api/pieces/1001_501_bangalore_s001/verify", json={
            "current_text": "मेरा वेरिफाइड टेक्स्ट",
            "current_text_roman": "mera verified text",
            "main_language": "hi",
        })

        # Create a mock Kaggle session zip updating BOTH s001 (verified) and s003 (unverified without draft!)
        buf = io.BytesIO()
        new_rows = [
            {
                "segment_id": "1001_501_bangalore_s001", "text": "गलत ड्राफ्ट", "text_roman": "galat draft",
                "language": "en", "decoded_at": "2026-11-12T10:00:00",
            },
            {
                "segment_id": "1001_501_bangalore_s003", "text": "नया सेशन ड्राफ्ट", "text_roman": "naya session draft",
                "language": "hi", "decoded_at": "2026-11-12T10:00:00",
            },
        ]
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr(
                "segments/transcripts.jsonl",
                "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows) + "\n",
            )
        # Note: build_data.main() reads full disk manifests unless mocked; here we simulate staging + drafts merge cleanly
        # Check upload validation & staging of .zip + direct sync
        # Append new drafts directly for build_data-free unit test OR invoke importer with run_build=False after staging jsonl!
        # First test invalid zip fails 400
        bad = client.post("/api/import/upload", files={"file": ("bad.zip", b"not a zip", "application/zip")})
        assert bad.status_code == 400

        # Also verify .jsonl.gz decompression into incoming/ works cleanly
        gz_bytes = gzip.compress(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows).encode("utf-8")
        )
        res = importer.import_uploaded_bytes("session99.jsonl.gz", gz_bytes, run_build=False)
        assert "session99.jsonl" in res["staged_files"]

        # Simulate build_data writing drafts.jsonl with those new session drafts
        existing = [json.loads(l) for l in config.DRAFTS_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
        existing = [d for d in existing if d["piece_id"] != "1001_501_bangalore_s001"]
        for r in new_rows:
            existing.append({
                "piece_id": r["segment_id"], "text": r["text"], "text_roman": r["text_roman"],
                "language": r["language"], "silent": False, "loop": False, "draft_by": "v4", "decoded_at": r["decoded_at"],
            })
        config.DRAFTS_JSONL.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in existing) + "\n", encoding="utf-8")
        db.sync_from_labelling_data(force=True)

        call_d = client.get("/api/call/1001_501_bangalore").json()
        by_id = {p["piece_id"]: p for p in call_d["pieces"]}
        # Verified piece MUST remain completely untouched!
        assert by_id["1001_501_bangalore_s001"]["status"] == "verified"
        assert by_id["1001_501_bangalore_s001"]["current_text"] == "मेरा वेरिफाइड टेक्स्ट"
        assert by_id["1001_501_bangalore_s001"]["main_language"] == "hi"
        # Unverified empty s003 dynamically gets the new draft!
        assert by_id["1001_501_bangalore_s003"]["has_draft"] is True
        assert by_id["1001_501_bangalore_s003"]["current_text"] == "नया सेशन ड्राफ्ट"


def test_audio_overlap_and_lead_grouped_split(lab_env):
    with _local_client(lab_env["app"]) as client:
        # Verify ALL pieces across all calls in our synthetic dataset
        calls = client.get("/api/calls?status=all&limit=100").json()["calls"]
        for c in calls:
            detail = client.get(f"/api/call/{c['call_id']}").json()
            for p in detail["pieces"]:
                res = client.post(f"/api/pieces/{p['piece_id']}/verify", json={
                    "main_language": p["main_language"] or c["language"] or "hi",
                })
                assert res.status_code == 200

        # Run audio overlap check (m4a vs shifted mp3 with different MD5!)
        ov = client.post("/api/split/check-overlaps", json={"scope": "all", "background": False}).json()
        matches = ov["result"]["matches"]
        assert any(m["old_call_id"] == "hi_ka_0001" and m["crm_call_id"] == "2002_601_kolkata" for m in matches)

        # Apply lead + overlap grouped split
        sp = client.post("/api/split/apply", json={"test_ratio": 0.25, "seed": 42}).json()
        assert sp["applied"] is True
        assert sp["leakage_check"] == {"shared_leads_across_splits": 0, "shared_overlaps_across_splits": 0}

        assign = sp["assignments"]
        # Old test call + overlapping CRM call MUST both be on 'test'
        assert assign["hi_ka_0001"]["split"] == "test"
        assert assign["2002_601_kolkata"]["split"] == "test"
        # Same lead_id ('1001') calls MUST be on the exact same side!
        assert assign["1001_501_bangalore"]["split"] == assign["1001_502_bangalore"]["split"]

        # dataset.jsonl carries the applied splits
        ds = [json.loads(l) for l in (config.VERIFIED_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        by_seg = {r["segment_id"]: r for r in ds}
        assert by_seg["2002_601_kolkata_s001"]["split"] == "test"


def test_spelling_guide_persistence(lab_env):
    with _local_client(lab_env["app"]) as client:
        g1 = client.get("/api/guide").json()["content"]
        assert "Spelling Guide" in g1
        new_md = g1 + "\n- `nahin` -> always `nahi`\n"
        g2 = client.put("/api/guide", json={"content": new_md}).json()["content"]
        assert "`nahi`" in g2
        assert "`nahi`" in config.SPELLING_GUIDE_PATH.read_text(encoding="utf-8")
