"""Configuration, paths, language metadata, and default spelling guide for the labelling tool."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "labelling_data"
AUDIO_DIR = DATA_DIR / "audio"
INCOMING_DIR = DATA_DIR / "incoming"
VERIFIED_DIR = DATA_DIR / "verified"
EDITED_AUDIO_DIR = DATA_DIR / "edited_audio"
WAVEFORM_CACHE_DIR = DATA_DIR / ".waveform_cache"
OPUS_CACHE_DIR = DATA_DIR / ".opus_cache"
BACKUP_DIR = DATA_DIR / "backups"
DB_PATH = DATA_DIR / "labelling.db"
SPELLING_GUIDE_PATH = DATA_DIR / "spelling_guide.md"

CALLS_CSV = DATA_DIR / "calls.csv"
PIECES_CSV = DATA_DIR / "pieces.csv"
DRAFTS_JSONL = DATA_DIR / "drafts.jsonl"
CALL_TEXTS_JSONL = DATA_DIR / "call_texts.jsonl"

DEFAULT_PORT = 8092
DEFAULT_LOCAL_USER = "gyan"

LANGS = ("hi", "en", "kn", "te", "mr", "bn")
LANG_FULL = {
    "hi": "hindi",
    "en": "english",
    "kn": "kannada",
    "te": "telugu",
    "mr": "marathi",
    "bn": "bengali",
}
LANG_LABEL = {
    "hi": "Hindi",
    "en": "English",
    "kn": "Kannada",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "none": "No speech",
}
LANG_COLORS = {
    "hi": "#7C5CFF",
    "en": "#2F6FEB",
    "kn": "#C2410C",
    "te": "#0E9F6E",
    "mr": "#B45309",
    "bn": "#BE185D",
}

DEFAULT_SPELLING_GUIDE = """# Spelling Guide

Goal: keep new labels consistent with the cleaned dataset so the model learns one clear style.

## Core Rules
1. **Proper native and romanized completely:**
   - In **Native** text, write all spoken words (including English loanwords, project/brand names, and acronyms like BHK) in that language's native script (for example `बीएचके`, `ब्रिकफोलियो`, `लोकेशन`, `ओके`).
   - In **Roman** text, write everything naturally in Latin letters using common Indian-English / Hinglish spelling.
2. **Numbers as digits:**
   - Write numbers as digits (`2`, `3`, `1.5`, `2028`, `40`, `9836254162`) in both native and romanized text.
3. **Verbatim speech:**
   - Transcribe what was actually spoken (fillers like `uh`, `umm`, `हाँ`, `हूँ`, restarts), without summarizing or translating.
   - Leave truly silent or non-speech pieces empty (or keep unverified if unclear).
4. **Two statuses only:**
   - Skip anything unclear or noisy — skipped pieces stay **Unverified** and never enter training.

## Team Rules (add one by one as decided)
- Add agreed spellings here (click Edit above) — e.g. `to` vs `toh`, `nahi` vs `nahin`, `हाँ` vs `हां`, `okay` vs `ok`.
"""
