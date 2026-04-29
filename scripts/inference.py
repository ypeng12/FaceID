import argparse
import csv
import os
import time
import sys
import numpy as np
import yaml

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.similarity import numpy_vectorized_cosine
from src.evaluation import compute_confidence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, 'configs', 'inference_config.yaml')

def _load_embedder_class():
    # Lazy import keeps CLI helper paths lightweight and test-friendly.
    from src.embedding import FaceEmbedder
    return FaceEmbedder

def load_inference_config(config_path):
    defaults = {
        'model_name': 'Facenet',
        'threshold': 0.35,
        'confidence_k': 10.0,
        'score_is_distance': False,
    }

    if config_path is None or not os.path.exists(config_path):
        return defaults

    with open(config_path, 'r') as f:
        loaded = yaml.safe_load(f) or {}

    config = defaults.copy()
    config.update(loaded)

    config['threshold'] = float(config['threshold'])
    config['confidence_k'] = float(config['confidence_k'])
    config['score_is_distance'] = bool(config['score_is_distance'])
    config['model_name'] = str(config['model_name'])
    return config

def run_inference(
    img1_path,
    img2_path,
    threshold=0.35,
    model_name='Facenet',
    confidence_k=10.0,
    score_is_distance=False,
    embedder=None,
):
    """
    Run the full inference pipeline for one pair of images.
    Returns a dictionary with result details.
    """
    start_total = time.perf_counter()
    if embedder is None:
        FaceEmbedder = _load_embedder_class()
        embedder = FaceEmbedder(model_name=model_name)
    
    # 1. Preprocessing & Embedding Extraction
    start_emb = time.perf_counter()
    emb1 = embedder.compute_embedding(img1_path)
    emb2 = embedder.compute_embedding(img2_path)
    latency_emb = time.perf_counter() - start_emb
    
    # 2. Similarity Scoring
    start_scoring = time.perf_counter()
    # Reshape for vectorized function
    score = numpy_vectorized_cosine(emb1.reshape(1, -1), emb2.reshape(1, -1))[0]
    latency_scoring = time.perf_counter() - start_scoring
    
    # 3. Decision
    if score_is_distance:
        is_same = score < threshold
    else:
        is_same = score >= threshold
    decision = "SAME" if is_same else "DIFFERENT"
    
    # 4. Calibrated Confidence
    confidence = compute_confidence(
        score,
        threshold,
        k=confidence_k,
        score_is_distance=score_is_distance,
    )
    
    latency_total = time.perf_counter() - start_total
    
    return {
        "img1": img1_path,
        "img2": img2_path,
        "similarity_score": round(float(score), 4),
        "threshold": threshold,
        "decision": decision,
        "is_same": int(is_same),
        "confidence": round(confidence, 4),
        "model_name": model_name,
        "latency_total_ms": round(latency_total * 1000, 2),
        "latency_emb_ms": round(latency_emb * 1000, 2),
        "latency_scoring_ms": round(latency_scoring * 1000, 2)
    }

def _resolve_runtime_config(args):
    config = load_inference_config(args.config)
    if args.threshold is not None:
        config['threshold'] = float(args.threshold)
    if args.model_name is not None:
        config['model_name'] = str(args.model_name)
    return config

def main():
    parser = argparse.ArgumentParser(description="FaceID Inference CLI (Milestone 4)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to inference YAML config",
    )
    parser.add_argument("--img1", type=str, help="Path to first face image")
    parser.add_argument("--img2", type=str, help="Path to second face image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override. Default: 0.35",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional embedding model override.",
    )
    parser.add_argument("--batch", type=str, help="Path to batch CSV file (with left_path, right_path)")
    
    args = parser.parse_args()
    runtime_config = _resolve_runtime_config(args)
    
    FaceEmbedder = _load_embedder_class()
    embedder = FaceEmbedder(model_name=runtime_config['model_name'])
    
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"Error: Batch file {args.batch} not found.")
            return
            
        print(f"{'Img1':<40} | {'Img2':<40} | {'Score':<8} | {'Decision':<10} | {'Conf':<6} | {'Lat(ms)':<8}")
        print("-" * 130)
        
        with open(args.batch, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                res = run_inference(
                    row['left_path'],
                    row['right_path'],
                    threshold=runtime_config['threshold'],
                    model_name=runtime_config['model_name'],
                    confidence_k=runtime_config.get('confidence_k', 10.0),
                    score_is_distance=runtime_config.get('score_is_distance', False),
                    embedder=embedder,
                )
                print(f"{os.path.basename(res['img1']):<40} | {os.path.basename(res['img2']):<40} | {res['similarity_score']:<8.4f} | {res['decision']:<10} | {res['confidence']:<6.4f} | {res['latency_total_ms']:<8.2f}")
    
    elif args.img1 and args.img2:
        res = run_inference(
            args.img1,
            args.img2,
            threshold=runtime_config['threshold'],
            model_name=runtime_config['model_name'],
            confidence_k=runtime_config.get('confidence_k', 10.0),
            score_is_distance=runtime_config.get('score_is_distance', False),
            embedder=embedder,
        )
        print("\n--- FaceID Inference Result ---")
        print(f"Inputs:    {res['img1']} vs {res['img2']}")
        print(f"Score:     {res['similarity_score']:.4f} (Threshold: {res['threshold']})")
        print(f"Decision:  {res['decision']}")
        print(f"Confidence: {res['confidence']:.4f}")
        print(f"Latency:   Total: {res['latency_total_ms']}ms (Emb: {res['latency_emb_ms']}ms, Scoring: {res['latency_scoring_ms']}ms)")
        print("-------------------------------\n")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
