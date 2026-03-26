import argparse
import os
import yaml
import csv
import json
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.embedding import FaceEmbedder
from src.similarity import numpy_vectorized_cosine
from src.evaluation import compute_metrics, get_confusion_matrix
from src.tracking import Tracker

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def read_pairs(csv_path):
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                'left_path': row['left_path'],
                'right_path': row['right_path'],
                'label': int(row['label'])
            })
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Evaluate face verification pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    os.makedirs(config.get("output_dir", "outputs/eval"), exist_ok=True)
    
    # Validation checks
    csv_path = config["pairs_file"]
    if not os.path.exists(csv_path):
         raise FileNotFoundError(f"Pairs file {csv_path} not found.")
         
    pairs = read_pairs(csv_path)
    if len(pairs) == 0:
        raise ValueError("Pairs file is empty.")
        
    labels = [p['label'] for p in pairs]
    if set(labels) - {0, 1}:
        raise ValueError("Labels must be strictly 0 or 1.")
        
    print(f"Loaded {len(pairs)} pairs from {csv_path}")
    
    # Extract unique embeddings
    unique_paths = set()
    for p in pairs:
        unique_paths.add(p['left_path'])
        unique_paths.add(p['right_path'])
    unique_paths = list(unique_paths)
    
    for p in unique_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
            
    print(f"Extracting embeddings for {len(unique_paths)} unique images...")
    embedder = FaceEmbedder()
    emb_matrix = embedder.batch_compute_embeddings(unique_paths, batch_size=64)
    path_to_emb = {p: emb_matrix[i] for i, p in enumerate(unique_paths)}
    
    # Compute similarity scores
    left_embs = np.array([path_to_emb[p['left_path']] for p in pairs])
    right_embs = np.array([path_to_emb[p['right_path']] for p in pairs])
    
    print("Computing cosine similarity scores...")
    scores = numpy_vectorized_cosine(left_embs, right_embs)
    target_labels = np.array(labels)
    
    if len(scores) != len(pairs):
         raise ValueError(f"Score count ({len(scores)}) does not match pair count ({len(pairs)}).")
         
    tracker = Tracker()
    
    if config.get("is_sweep", False):
        print("Running threshold sweep [0.0 to 1.0]...")
        thresholds = np.linspace(0.0, 1.0, 101)
        sweep_data = []
        best_f1 = -1
        best_th = 0
        for th in thresholds:
            m = compute_metrics(scores, target_labels, th, score_is_distance=False)
            # Must handle NaN f1 cases cleanly if possible, here handled in evaluation logic
            sweep_data.append({"threshold": th, "tpr": m["tpr"], "fpr": m["fpr"], "f1": m["f1"]})
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_th = th
                
        metrics = {"best_threshold": float(best_th), "best_f1": float(best_f1)}
        print(f"Sweep complete. Best F1: {best_f1:.4f} at threshold {best_th:.4f}")
        
        sweep_file = os.path.join(config["output_dir"], f"sweep_{config['run_name']}.json")
        with open(sweep_file, 'w') as f:
             json.dump(sweep_data, f, indent=2)
        print(f"Sweep data saved to {sweep_file}")
        
        tracker.log_run(config['run_name'], args.config, config['data_version'], True, float(best_th), metrics, config['note'])
        
    else:
        th = config["threshold"]
        print(f"Evaluating at chosen threshold {th}...")
        metrics = compute_metrics(scores, target_labels, th, score_is_distance=False)
        cm = get_confusion_matrix(scores, target_labels, th, score_is_distance=False)
        
        print("Metrics:", metrics)
        print("Confusion Matrix:", cm)
        
        preds = (scores >= th).astype(int)
        
        # Slices
        false_positives = [
            {"left": pairs[i]['left_path'], "right": pairs[i]['right_path'], "score": float(scores[i])}
            for i in range(len(pairs)) if preds[i] == 1 and target_labels[i] == 0
        ]
        false_negatives = [
            {"left": pairs[i]['left_path'], "right": pairs[i]['right_path'], "score": float(scores[i])}
            for i in range(len(pairs)) if preds[i] == 0 and target_labels[i] == 1
        ]
        
        error_file = os.path.join(config["output_dir"], f"errors_{config['run_name']}.json")
        with open(error_file, 'w') as f:
             json.dump({"false_positives": false_positives, "false_negatives": false_negatives}, f, indent=2)
        print(f"Error slices saved to {error_file}")
             
        metrics['cm'] = cm
        tracker.log_run(config['run_name'], args.config, config['data_version'], False, float(th), metrics, config['note'])

    print("Run logged successfully.")
    
if __name__ == "__main__":
    main()
