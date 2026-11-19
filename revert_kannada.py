"""
Revert tool specifically for Kannada automation run.
Reads labelling_data/automation_log_kannada.jsonl and restores all pieces to their exact previous state.
"""

import json
import sqlite3
from pathlib import Path
from labelling_tool import exporter

BASE_DIR = Path(r"c:\Gyan Gupta\crm-llm-model-fine-tunning")
DATA_DIR = BASE_DIR / "labelling_data"
DB_PATH = DATA_DIR / "labelling.db"
LOG_PATH = DATA_DIR / "automation_log_kannada.jsonl"

def revert_kannada():
    if not LOG_PATH.exists():
        print("No Kannada automation log found.")
        return

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("Kannada automation log is empty.")
        return

    print(f"Reverting {len(lines)} Kannada piece(s)...")

    conn = sqlite3.connect(str(DB_PATH), timeout=60.0)
    cursor = conn.cursor()

    reverted_count = 0
    affected_calls = set()

    for line in lines:
        entry = json.loads(line)
        piece_id = entry["piece_id"]
        call_id = entry.get("call_id", "")
        prev_status = entry.get("prev_status", "unverified")
        prev_text = entry.get("prev_text", "")
        prev_text_roman = entry.get("prev_text_roman", "")

        cursor.execute("""
            UPDATE pieces
            SET status = ?,
                current_text = ?,
                current_text_roman = ?,
                verified_by = '',
                verified_at = '',
                updated_at = datetime('now')
            WHERE piece_id = ?
        """, (prev_status, prev_text, prev_text_roman, piece_id))

        if call_id:
            affected_calls.add(call_id)
        reverted_count += 1

    # Recompute call verified counts
    for cid in affected_calls:
        cursor.execute("SELECT COUNT(*) FROM pieces WHERE call_id = ? AND status = 'verified'", (cid,))
        v_count = cursor.fetchone()[0]
        cursor.execute("UPDATE calls SET verified_count = ? WHERE call_id = ?", (v_count, cid))

    conn.commit()
    conn.close()

    # Clear log
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    exporter.sync_all_verified(DB_PATH)
    print(f"Successfully reverted {reverted_count} Kannada pieces in database.")

if __name__ == "__main__":
    revert_kannada()
