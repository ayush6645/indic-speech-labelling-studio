import sqlite3
import sys
from pathlib import Path

# Add labelling_tool to path
BASE_DIR = Path(r"c:\Gyan Gupta\crm-llm-model-fine-tunning")
sys.path.insert(0, str(BASE_DIR))
from labelling_tool import db, exporter

DB_PATH = BASE_DIR / "labelling_data" / "labelling.db"
BACKUP_PATH = BASE_DIR / "labelling_data" / "backups" / "labelling_2026-11-14.db"

piece_ids = [
    '151381_4219411_bangalore_s002', '151381_4219411_bangalore_s003', '151381_4219411_bangalore_s004',
    '151381_4219411_bangalore_s005', '151381_4219411_bangalore_s006', '151381_4219411_bangalore_s007',
    '151381_4219698_bangalore_s001', '151381_4219698_bangalore_s002',
    '151381_923236_bangalore_s001', '151381_923236_bangalore_s002', '151381_923236_bangalore_s003',
    '151381_923236_bangalore_s004', '151381_923236_bangalore_s005', '151381_923236_bangalore_s006',
    '151381_923236_bangalore_s007', '151381_923236_bangalore_s008'
]

calls = ['151381_4219411_bangalore', '151381_4219698_bangalore', '151381_923236_bangalore']

backup_conn = sqlite3.connect(str(BACKUP_PATH))
backup_conn.row_factory = sqlite3.Row

target_conn = sqlite3.connect(str(DB_PATH))
target_conn.row_factory = sqlite3.Row

cols = [c[1] for c in target_conn.execute("PRAGMA table_info(pieces)").fetchall()]

with target_conn:
    # 1. Restore the 16 pieces from backup
    for pid in piece_ids:
        b_row = backup_conn.execute("SELECT * FROM pieces WHERE piece_id = ?", (pid,)).fetchone()
        if b_row:
            vals = [b_row[c] for c in cols]
            set_clause = ", ".join([f"{c} = ?" for c in cols if c != "piece_id"])
            param_vals = [b_row[c] for c in cols if c != "piece_id"] + [pid]
            target_conn.execute(f"UPDATE pieces SET {set_clause} WHERE piece_id = ?", param_vals)
            
    # 2. Clean history entries created today during automation
    placeholders = ",".join(["?"] * len(piece_ids))
    target_conn.execute(
        f"DELETE FROM history WHERE piece_id IN ({placeholders}) AND saved_at >= '2026-11-14T13:00:00'",
        piece_ids
    )
    
    # 3. Recompute call stats for affected calls
    for cid in calls:
        db.recompute_call_stats(target_conn, cid)
        
    print(f"Successfully reverted {len(piece_ids)} pieces in database.")

# 4. Trigger dataset re-export to keep verified/ clean
exporter.full_export(target_conn)
print("Verified dataset re-exported cleanly.")

target_conn.close()
backup_conn.close()
