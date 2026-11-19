"""Worker 2 of 4: Resume-First Partial Calls Completion (rowid % 4 == 1)"""
import sys
from auto_labeller_core import run_partial_calls_worker

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 999999
    run_partial_calls_worker(
        worker_id=2,
        total_workers=4,
        profile_name=".chrome_profile_worker2",
        log_filename="automation_log_worker2.jsonl",
        max_pieces=count,
    )
