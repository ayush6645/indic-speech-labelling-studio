"""
One-click Revert Utility for Automated Labelling Runs
Restores all pieces modified during automation back to their previous state.
"""

import sys
import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(r"c:\Gyan Gupta\crm-llm-model-fine-tunning")
DATA_DIR = BASE_DIR / "labelling_data"
DB_PATH = DATA_DIR / "labelling.db"
LOG_PATH = DATA_DIR / "automation_log.jsonl"

sys.path.insert(0, str(BASE_DIR))
from labelling_tool import db, exporter

def revert_last_run(batch_size: int | None = None):
    if not LOG_PATH.exists():
        print("No automation log found at labelling_data/automation_log.jsonl.")
        return
        
    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
                    
    if not entries:
        print("Automation log is empty.")
        return
        
    if batch_size is not None:
        to_revert = entries[-batch_size:]
        remaining = entries[:-batch_size]
    else:
        to_revert = entries
        remaining = []
        
    print(f"Reverting {len(to_revert)} piece(s)...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    affected_calls = set()
    
    with conn:
        for item in to_revert:
            pid = item["piece_id"]
            prev_status = item.get("prev_status", "unverified")
            prev_text = item.get("prev_text", "")
            prev_text_roman = item.get("prev_text_roman", "")
            call_id = item.get("call_id", "")
            
            conn.execute("""
                UPDATE pieces SET
                    status = ?,
                    current_text = ?,
                    current_text_roman = ?,
                    verified_by = '',
                    verified_at = '',
                    updated_at = ?
                WHERE piece_id = ?
            """, (prev_status, prev_text, prev_text_roman, db.now_iso(), pid))
            
            # Remove corresponding history entry
            conn.execute("DELETE FROM history WHERE piece_id = ? AND action = 'verify'", (pid,))
            if call_id:
                affected_calls.add(call_id)
                
        for cid in affected_calls:
            db.recompute_call_stats(conn, cid)
            
    print(f"Successfully reverted {len(to_revert)} pieces in database.")
    
    # Save remaining log entries
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for item in remaining:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Resync verified export files
    exporter.sync_all_verified(DB_PATH)
    print("Verified dataset re-synced.")
    conn.close()

if __name__ == "__main__":
    count = None
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            count = None
    revert_last_run(count)
