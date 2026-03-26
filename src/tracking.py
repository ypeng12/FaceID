import json
import os
import datetime
import subprocess

def get_git_hash() -> str:
    try:
        res = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL)
        return res.decode('utf-8').strip()
    except Exception:
        return "unknown"

class Tracker:
    def __init__(self, log_file="outputs/runs.json"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def log_run(self, run_id, config_name, data_version, is_sweep, threshold, metrics, note):
        with open(self.log_file, 'r') as f:
            runs = json.load(f)
            
        run_data = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "commit": get_git_hash(),
            "config": config_name,
            "data_version": data_version,
            "is_sweep": is_sweep,
            "threshold": threshold,
            "metrics": metrics,
            "note": note
        }
        runs.append(run_data)
        
        with open(self.log_file, 'w') as f:
            json.dump(runs, f, indent=2)
            
        return run_data
