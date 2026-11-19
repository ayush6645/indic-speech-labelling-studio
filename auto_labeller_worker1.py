"""Worker 1 of 4: Resume-First Partial Calls Completion (rowid % 4 == 0)"""
import sys
from auto_labeller_core import run_partial_calls_worker

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 999999
    run_partial_calls_worker(
        worker_id=1,
        total_workers=4,
        profile_name=".chrome_profile_general",
        log_filename="automation_log_worker1.jsonl",
        max_pieces=count,
    )
