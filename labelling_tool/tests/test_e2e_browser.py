"""End-to-end Playwright Chromium tests for the Labelling Tool UI."""
import csv
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import uvicorn
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labelling_tool import config, server  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _make_signal(dur_s: float = 8.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, dur_s, int(dur_s * sr), endpoint=False, dtype=np.float32)
    return (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


@pytest.fixture()
def live_server(tmp_path, monkeypatch):
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
        audio_dir / "calls" / "old" / "hi",
        audio_dir / "pieces" / "crm" / "bangalore",
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
    sig = _make_signal(8.0, sr)
    w_tmp = tmp_path / "c1.wav"
    sf.write(str(w_tmp), sig, sr)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(w_tmp),
         str(audio_dir / "calls" / "crm" / "bangalore" / "101_1_bangalore.mp3")],
        check=True,
    )
    for i, (a, b) in enumerate([(0.0, 4.0), (4.0, 8.0)], 1):
        sf.write(
            str(audio_dir / "pieces" / "crm" / "bangalore" / f"101_1_bangalore_s{i:03d}.flac"),
            sig[int(a * sr) : int(b * sr)],
            sr,
            format="FLAC",
            subtype="PCM_16",
        )

    sf.write(str(audio_dir / "calls" / "old" / "hi" / "hi_old_001.wav"), sig[: 4 * sr], sr)
    sf.write(str(audio_dir / "pieces" / "old" / "hi" / "hi_old_001_s001.wav"), sig[: 4 * sr], sr)

    calls = [
        {
            "call_id": "101_1_bangalore", "source": "crm", "state": "bangalore", "language": "hi", "also_languages": "",
            "lead_id": "101", "crm_call_id": "1", "duration_s": "8.0", "pieces": "2", "pieces_with_text": "2",
            "audio": "audio/calls/crm/bangalore/101_1_bangalore.mp3", "md5": "a1", "old_split": "", "has_call_text": "", "flags": "",
        },
        {
            "call_id": "hi_old_001", "source": "old", "state": "karnataka", "language": "hi", "also_languages": "",
            "lead_id": "", "crm_call_id": "", "duration_s": "4.0", "pieces": "1", "pieces_with_text": "1",
            "audio": "audio/calls/old/hi/hi_old_001.wav", "md5": "a2", "old_split": "train", "has_call_text": "yes", "flags": "",
        },
    ]
    pieces = [
        {
            "piece_id": "101_1_bangalore_s001", "call_id": "101_1_bangalore", "source": "crm", "index": "1", "count": "2",
            "start_s": "0.0", "end_s": "4.0", "duration_s": "4.0",
            "audio": "audio/pieces/crm/bangalore/101_1_bangalore_s001.flac", "silent": "0", "old_split": "", "flags": "",
        },
        {
            "piece_id": "101_1_bangalore_s002", "call_id": "101_1_bangalore", "source": "crm", "index": "2", "count": "2",
            "start_s": "4.0", "end_s": "8.0", "duration_s": "4.0",
            "audio": "audio/pieces/crm/bangalore/101_1_bangalore_s002.flac", "silent": "0", "old_split": "", "flags": "",
        },
        {
            "piece_id": "hi_old_001_s001", "call_id": "hi_old_001", "source": "old", "index": "1", "count": "1",
            "start_s": "0.0", "end_s": "4.0", "duration_s": "4.0",
            "audio": "audio/pieces/old/hi/hi_old_001_s001.wav", "silent": "", "old_split": "train", "flags": "",
        },
    ]
    drafts = [
        {
            "piece_id": "101_1_bangalore_s001", "text": "हैलो सर", "text_roman": "hello sir",
            "language": "hi", "silent": False, "loop": False, "draft_by": "v3 (crm-indic-asr 1.0.0)", "decoded_at": "2026-11-11",
        },
        {
            "piece_id": "101_1_bangalore_s002", "text": "गुड मॉर्निंग", "text_roman": "good morning",
            "language": "hi", "silent": False, "loop": False, "draft_by": "v3 (crm-indic-asr 1.0.0)", "decoded_at": "2026-11-11",
        },
        {
            "piece_id": "hi_old_001_s001", "text": "ठीक है", "text_roman": "thik hai",
            "language": "hi", "silent": False, "loop": False, "draft_by": "old label", "decoded_at": "",
        },
    ]
    call_texts = [
        {
            "call_id": "hi_old_001", "text_native": "ठीक है पूरा कॉल", "text_romanized": "thik hai pura call",
            "v4_model": "v4", "v4_diarized_text": "Agent: ठीक है", "v4_english_text": "Okay", "project": "Demo",
        }
    ]

    with open(config.CALLS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(calls[0].keys())); w.writeheader(); w.writerows(calls)
    with open(config.PIECES_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(pieces[0].keys())); w.writeheader(); w.writerows(pieces)
    with open(config.DRAFTS_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for d in drafts: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(config.CALL_TEXTS_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for ct in call_texts: f.write(json.dumps(ct, ensure_ascii=False) + "\n")

    app = server.create_app(local_user="gyan", auto_sync=True)
    port = _free_port()
    uv_cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv_srv = uvicorn.Server(uv_cfg)
    thread = threading.Thread(target=uv_srv.run, daemon=True)
    thread.start()
    for _ in range(60):
        if uv_srv.started:
            break
        time.sleep(0.05)

    yield f"http://127.0.0.1:{port}", data_dir
    uv_srv.should_exit = True
    thread.join(timeout=5)


def test_full_browser_workflow(live_server):
    url, data_dir = live_server
    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 860})
        page = context.new_page()
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Auto-login locally & initial workspace
        page.goto(url, wait_until="networkidle")
        page.wait_for_selector("#app-shell:not(.hidden)")
        page.wait_for_selector("#call-view:not(.hidden)")
        assert page.locator("#login-screen").is_hidden()
        assert page.locator(".call-item").count() == 2

        # 2. Waveform rendered cleanly & audio metadata loads
        wf_box = page.locator("#waveform-canvas").bounding_box()
        assert wf_box and wf_box["width"] > 200 and wf_box["height"] == pytest.approx(68, abs=2)
        page.wait_for_function("document.getElementById('audio-el').readyState >= 1", timeout=8000)
        aud_dur = page.evaluate("document.getElementById('audio-el').duration")
        assert aud_dur == pytest.approx(4.0, abs=0.2)

        # 2b. Speed presets & step buttons work, sync playbackRate, and persist across reload
        page.click(".speed-preset-btn[data-speed='1.25']")
        assert page.locator("#speed-val").inner_text() == "1.25×"
        assert "active" in (page.locator(".speed-preset-btn[data-speed='1.25']").get_attribute("class") or "")
        assert page.evaluate("document.getElementById('audio-el').playbackRate") == pytest.approx(1.25, abs=0.01)
        page.click("#btn-speed-up")
        assert page.locator("#speed-val").inner_text() == "1.35×"
        page.click("#btn-speed-down")
        assert page.locator("#speed-val").inner_text() == "1.25×"
        page.click(".speed-preset-btn[data-speed='1.5']")
        assert page.locator("#speed-val").inner_text() == "1.50×"
        assert page.evaluate("document.getElementById('audio-el').playbackRate") == pytest.approx(1.5, abs=0.01)
        page.reload(wait_until="networkidle")
        page.wait_for_selector("#call-view:not(.hidden)")
        assert page.locator("#speed-val").inner_text() == "1.50×"
        assert "active" in (page.locator(".speed-preset-btn[data-speed='1.5']").get_attribute("class") or "")
        page.click(".speed-preset-btn[data-speed='1']")
        assert page.locator("#speed-val").inner_text() == "1.00×"

        # 2c. Sticky header + waveform + player controls remain pinned while scrolling pieces
        page.click("#btn-toggle-edge")
        page.click("#btn-toggle-hist")
        page.wait_for_selector("#hist-panel .word-diff-line")
        page.set_viewport_size({"width": 1280, "height": 620})
        page.wait_for_function(
            "document.getElementById('call-area').scrollHeight > document.getElementById('call-area').clientHeight",
            timeout=5000,
        )
        sticky_before = page.locator("#call-sticky-top").bounding_box()
        page.evaluate("document.getElementById('call-area').scrollTop = 180")
        page.wait_for_selector("#call-sticky-top.is-stuck")
        sticky_after = page.locator("#call-sticky-top").bounding_box()
        assert sticky_before and sticky_after
        assert sticky_after["y"] == pytest.approx(sticky_before["y"], abs=2)
        # Open s002 while keeping hist open (so s002 is tall), scroll down past top of s001, then re-open s001 -> s001 must not hide under sticky bar
        page.click("[data-open-piece='101_1_bangalore_s002']")
        page.wait_for_selector("[data-piece-id='101_1_bangalore_s002']")
        page.evaluate("document.getElementById('call-area').scrollTop = 200")
        page.click("[data-open-piece='101_1_bangalore_s001']")
        page.wait_for_selector("[data-piece-id='101_1_bangalore_s001']")
        sticky_now = page.locator("#call-sticky-top").bounding_box()
        open_box = page.locator("[data-piece-id='101_1_bangalore_s001']").bounding_box()
        assert sticky_now and open_box
        assert open_box["y"] >= sticky_now["y"] + sticky_now["height"] - 1
        # Reset edge/hist toggles and viewport back to default for downstream assertions
        page.set_viewport_size({"width": 1280, "height": 860})
        if page.locator("#edge-panel").is_visible():
            page.click("#btn-toggle-edge")
        if page.locator("#hist-panel").is_visible():
            page.click("#btn-toggle-hist")

        # 3. Empty state & Clear filters
        page.fill("#filter-search", "no_such_call_xyz")
        page.wait_for_selector("#btn-clear-filters:not(.hidden)")
        assert "No calls match" in page.locator("#empty-state-msg").inner_text()
        page.click("#btn-clear-filters")
        page.wait_for_selector(".call-item")

        # 4. Two boxes + Amber "Needs update" border + autosave after 2s + Confirm ok
        ta_nat = page.locator("#edit-native")
        ta_rom = page.locator("#edit-roman")
        ta_nat.fill("हैलो सर जी")
        assert "needs-update" in (ta_rom.get_attribute("class") or "")
        page.wait_for_selector("#chip-needs-roman:not(.hidden)")
        # Wait > 2.2s for 2s autosave
        page.wait_for_function("document.getElementById('save-note')?.textContent === 'Saved'", timeout=6000)
        page.click("#chip-needs-roman")
        page.wait_for_function("!document.getElementById('edit-roman').classList.contains('needs-update')")

        # 5. Also contains language pill
        page.click("[data-toggle-also='en']")
        page.wait_for_selector("[data-toggle-also='en'].selected")

        # 6. Verify piece -> updates green badge, opens s002, updates top per-language verified hours legend, writes verified/dataset.jsonl
        page.click("#btn-piece-verify")
        page.wait_for_selector("[data-piece-id='101_1_bangalore_s002']")
        page.wait_for_selector("#lang-hours-bar [data-lang-hour='hi'].has-data")
        assert "<0.01h" in page.locator("#lang-hours-bar [data-lang-hour='hi']").inner_text()
        assert "0.00h" in page.locator("#lang-hours-bar [data-lang-hour='en']").inner_text()
        ds_file = data_dir / "verified" / "dataset.jsonl"
        assert ds_file.is_file()
        assert "हैलो सर जी" in ds_file.read_text(encoding="utf-8")

        # 7. Edge fix on s002: start buttons are safely disabled because adjacent s001 is verified; end buttons enabled!
        page.click("#btn-toggle-edge")
        assert page.locator("[data-move-side='start'][data-move-delta='0.5']").is_disabled()
        assert page.locator("[data-move-side='end'][data-move-delta='-0.5']").is_enabled()

        # 8. Open s001 & Unverify with "Mark as unverified?" dialog
        page.click("[data-open-piece='101_1_bangalore_s001']")
        page.wait_for_selector("#btn-piece-unverify")
        page.click("#btn-piece-unverify")
        page.wait_for_selector("#confirm-overlay:not(.hidden)")
        assert "Mark as unverified?" in page.locator("#confirm-title").inner_text()
        page.click("#btn-confirm-ok")
        page.wait_for_function(
            "document.getElementById('piece-status-badge')?.textContent.trim() === 'Unverified'",
            timeout=6000,
        )

        # Now edge move on unverified s001 end +0.5s works cleanly!
        if page.locator("#edge-panel").is_hidden():
            page.click("#btn-toggle-edge")
        page.click("[data-move-side='end'][data-move-delta='0.5']")
        page.wait_for_function(
            "document.querySelector('.piece-open-meta .meta-text')?.textContent.includes('4.5 s')",
            timeout=6000,
        )

        # Verify s001 again so progress/WER modal has data
        page.click("#btn-piece-verify")
        page.wait_for_selector("[data-piece-id='101_1_bangalore_s002']")

        # 9. Spelling Guide side drawer (? shortcut) NEVER covers text boxes + persists edits
        page.locator("body").click(position={"x": 10, "y": 10})
        page.keyboard.press("?")
        page.wait_for_selector("#guide-drawer:not(.hidden)")
        page.wait_for_selector("#guide-view h1")
        rom_box = page.locator("#edit-roman").bounding_box()
        guide_box = page.locator("#guide-drawer").bounding_box()
        assert rom_box and guide_box
        assert guide_box["x"] >= rom_box["x"] + rom_box["width"] - 1

        page.click("#btn-guide-edit")
        cur_guide = page.locator("#guide-editor").input_value()
        page.fill("#guide-editor", cur_guide + "\n- `playwright_rule` test\n")
        page.click("#btn-guide-save")
        page.wait_for_selector("#guide-view:not(.hidden)")
        assert "playwright_rule" in (data_dir / "spelling_guide.md").read_text(encoding="utf-8")
        page.click("#btn-guide-close")

        # 10. Reference card on old call
        page.locator(".call-item", has_text="hi_old_001").click()
        page.wait_for_selector("#reference-card:not(.hidden)")
        page.click("#btn-toggle-ref")
        page.wait_for_selector("#reference-body:not(.hidden)")
        assert "Agent:" in page.locator("#reference-body").inner_text()

        # 11. Progress & WER modal + Dark theme toggle + Mobile viewport
        page.click("#btn-progress")
        page.wait_for_selector("#progress-content table")
        assert "v3 WER (Native)" in page.locator("#progress-content").inner_text()
        page.click("#progress-overlay .modal-close")

        page.click("#btn-theme")
        assert page.locator("html").get_attribute("data-theme") == "dark"

        page.set_viewport_size({"width": 390, "height": 844})
        assert page.locator("#btn-drawer-toggle").is_visible()
        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
        assert not overflow

        # 12. Remote / ngrok visitor sees Login screen & Opus enabled after sign-in
        page.click("#btn-user")
        page.wait_for_selector("#add-user-form")
        page.fill("#new-u-name", "ravi")
        page.fill("#new-u-pass", "ravi123")
        page.click("#add-user-form button[type='submit']")
        page.wait_for_selector("[data-toggle-user='ravi']")

        remote_ctx = browser.new_context(viewport={"width": 1280, "height": 860})

        def _add_proxy_hdr(route):
            hdrs = {**route.request.headers, "x-forwarded-host": "demo.ngrok-free.app"}
            route.continue_(headers=hdrs)

        remote_ctx.route(f"{url}/**", _add_proxy_hdr)
        rpage = remote_ctx.new_page()
        rpage.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        rpage.on("pageerror", lambda err: page_errors.append(str(err)))
        rpage.goto(url, wait_until="networkidle")
        rpage.wait_for_selector("#login-screen:not(.hidden)")
        rpage.fill("#login-username", "ravi")
        rpage.fill("#login-password", "ravi123")
        rpage.click("#login-form button[type='submit']")
        rpage.wait_for_selector("#app-shell:not(.hidden)")
        assert "btn-active" in (rpage.locator("#btn-opus").get_attribute("class") or "")
        # Remote Opus audio loads & plays cleanly in Chromium
        rpage.wait_for_function("document.getElementById('audio-el').readyState >= 1", timeout=8000)
        opus_src = rpage.evaluate("document.getElementById('audio-el').src")
        assert "fmt=opus" in opus_src

        remote_ctx.close()
        context.close()
        browser.close()

    assert console_errors == [], f"Console errors: {console_errors}"
    assert page_errors == [], f"Page errors: {page_errors}"
