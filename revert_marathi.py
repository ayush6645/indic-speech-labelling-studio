"""
Revert tool for Marathi Auto-Labeller
Rolls back all pieces verified by auto_labeller_marathi.py using automation_log_marathi.jsonl
Restores pieces to 'unverified', clears main_language or reverts to previous language,
and resyncs exports.
"""

import sys
import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(r"c:\Gyan Gupta\crm-llm-model-fine-tunning")
DATA_DIR = BASE_DIR / "labelling_data"
DB_PATH = DATA_DIR / "labelling.db"
LOG_PATH = DATA_DIR / "automation_log_marathi.jsonl"

def revert():
    if not LOG_PATH.exists():
        print(f"Log file not found: {LOG_PATH}. Nothing to revert.")
        return

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"Found {len(lines)} actions in {LOG_PATH.name} to revert.")
    if not lines:
        return

    conn = sqlite3.connect(str(DB_PATH))
    reverted_count = 0
    try:
        for entry in reversed(lines):
            pid = entry["piece_id"]
            prev_status = entry.get("prev_status", "unverified")
            prev_text = entry.get("prev_text", "")
            prev_text_roman = entry.get("prev_text_roman", "")
            prev_lang = entry.get("prev_language", "")

            conn.execute("""
                UPDATE pieces
                SET status = ?,
                    current_text = ?,
                    current_text_roman = ?,
                    main_language = ?,
                    verified_by = '',
                    verified_at = ''
                WHERE piece_id = ?
            """, (prev_status, prev_text, prev_text_roman, prev_lang, pid))
            reverted_count += 1

        conn.commit()
        print(f"Successfully reverted {reverted_count} Marathi pieces in database.")

        backup_log = LOG_PATH.with_suffix(".jsonl.bak")
        LOG_PATH.rename(backup_log)
        print(f"Archived log to {backup_log.name}")

    finally:
        conn.close()

    try:
        from labelling_tool.exporter import sync_all_verified
        sync_all_verified()
        print("Successfully resynced verified dataset files.")
    except Exception as e:
        print(f"Note on exporter sync: {e}")

if __name__ == "__main__":
    revert()
