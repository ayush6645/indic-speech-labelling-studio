# Indic Speech Labelling Studio 🎙️

> **High-efficiency web audio verification and transcription platform tailored for multilingual Indic speech recognition datasets (Hindi, Marathi, Bengali, Telugu, Kannada, English).**

---

## Overview

**Indic Speech Labelling Studio** is a standalone, full-stack audio annotation, verification, and dataset curation platform designed specifically for real-world, conversational Indic speech recognition (ASR) pipelines. 

Built to handle noisy, code-switched multi-dialect telephony and customer conversations, it bridges the gap between automated ASR model drafts and high-quality human-verified ground truth.

---

## Key Features

- **Multilingual Indic Support:** Built-in support for Hindi (`hi`), Marathi (`mr`), Bengali (`bn`), Telugu (`te`), Kannada (`kn`), and English (`en`), including mixed-code detection.
- **Dual-Script Verification:** Side-by-side editing of both **Native Script** (e.g., Devanagari) and **Romanized / Hinglish Script** with instant synchronization.
- **Dual-Version Persistence:** Preserves both the original AI model draft (`draft_text_native`, `draft_text_romanized`, `draft_by`) and the verified human transcription, enabling continuous model evaluation and error analysis.
- **Pinned Audio Player & Waveform:** Sticky waveform header with interactive scrubbing, piece/call toggle, audio volume boost, and loop piece playback.
- **Granular Speed Controls:** Real-time playback speed adjustment from `0.50×` to `2.00×` with pitch preservation and quick-select presets (`0.75×`, `1.00×`, `1.25×`, `1.50×`, `2.00×`).
- **Live Verified Hours Tracking:** Dynamic top-bar metrics tracking cumulative and per-language verified audio duration (`hi 0.28h`, `en 0.09h`, etc.).
- **Inline Diff Viewer:** Highlights modifications (`.diff-chg`) and deletions (`.diff-del`) between AI drafts and human corrections.
- **Offline & Low-Bandwidth Optimized:** Supports raw audio streaming and on-the-fly Opus encoding (`.opus_cache/`) for fast remote labelling sessions.
- **Automated Dataset Export:** Real-time exports to verified JSONL, call JSONs, and segment manifests for downstream fine-tuning.

---

## Architecture & Project Structure

```
indic-speech-labelling-studio/
├── labelling_tool/             # Web annotation platform
│   ├── server.py               # FastAPI backend with SQLite WAL
│   ├── db.py                   # Thread-safe SQLite database manager
│   ├── exporter.py             # Verified dataset exporter
│   ├── audio_utils.py          # Audio slicing, Opus conversion & waveforms
│   ├── static/
│   │   ├── index.html          # Annotation studio UI
│   │   ├── app.js              # State management & keyboard shortcuts
│   │   └── styles.css          # Modern dark-theme UI styling
│   └── tests/                  # Automated pytest suite
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- `ffmpeg` (recommended for audio decoding and opus streaming)

### 2. Installation
```bash
# Clone repository
git clone https://github.com/ayush6645/indic-speech-labelling-studio.git
cd indic-speech-labelling-studio

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pydantic soundfile pytest
```

### 3. Launch the Studio
```bash
python labelling_tool/server.py
```
Open your browser at `http://127.0.0.1:8092` to start labelling.

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Space` | Play / Pause audio |
| `Alt + Left / Right` | Skip back / forward 2 seconds |
| `Alt + Up / Down` | Previous / Next piece |
| `Ctrl + Enter` | Save and mark verified |
| `Alt + 1 .. 6` | Switch language (`hi`, `en`, `kn`, `te`, `mr`, `bn`) |
| `Tab` | Switch between Native and Romanized textareas |

---

## Testing

Run the automated test suite:
```bash
python -m pytest labelling_tool/tests/ -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
