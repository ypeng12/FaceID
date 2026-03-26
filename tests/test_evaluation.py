import os
import json
import pytest
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation import compute_metrics
from src.tracking import Tracker

def test_compute_metrics():
    # Synthetic scores and labels
    scores = [0.9, 0.8, 0.4, 0.3, 0.1]
    labels = [1, 1, 0, 1, 0]
    
    m = compute_metrics(scores, labels, threshold=0.5, score_is_distance=False)
    assert m['tp'] == 2
    assert m['fp'] == 0
    assert m['tn'] == 2
    assert m['fn'] == 1
    assert abs(m['accuracy'] - 0.8) < 1e-5

def test_tracker_logging(tmp_path):
    log_file = tmp_path / "runs.json"
    tracker = Tracker(log_file=str(log_file))
    
    tracker.log_run("TestRun", "config.yaml", "v_test", False, 0.5, {"accuracy": 0.5}, "note")
    
    assert log_file.exists()
    with open(log_file, 'r') as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]['run_id'] == "TestRun"
        assert data[0]['metrics']['accuracy'] == 0.5

def test_integration_pipeline_run(tmp_path):
    from PIL import Image
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    
    img1_path = test_dir / "1.jpg"
    img2_path = test_dir / "2.jpg"
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(img1_path)
    Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8)*255).save(img2_path)
    
    csv_path = test_dir / "pairs.csv"
    with open(csv_path, 'w') as f:
        f.write("left_path,right_path,label\n")
        # Fix paths so they are absolute or relative correctly. It's safer to use relative or absolute directly
        # the read_pairs function works fine with absolute.
        # Ensure we use forward-slashes for compat
        i1 = str(img1_path).replace("\\", "/")
        i2 = str(img2_path).replace("\\", "/")
        f.write(f"{i1},{i2},0\n")
        f.write(f"{i1},{i1},1\n")
        
    config_path = test_dir / "config.yaml"
    out_dir = test_dir / "out"
    config_content = f"""
run_name: "Int-Test"
data_version: "test"
split: "val"
pairs_file: "{str(csv_path).replace("\\", "/")}"
is_sweep: false
threshold: 0.5
output_dir: "{str(out_dir).replace("\\", "/")}"
note: "test"
"""
    with open(config_path, 'w') as f:
        f.write(config_content)
        
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    script_path = os.path.join(root_dir, "scripts", "run_evaluation.py")
    
    exit_code = os.system(f"python {script_path} --config {config_path}")
    assert exit_code == 0
    assert os.path.exists(os.path.join(out_dir, "errors_Int-Test.json"))
