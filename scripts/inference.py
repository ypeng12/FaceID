import argparse
import os
import time
import sys
import numpy as np

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding import FaceEmbedder
from src.similarity import numpy_vectorized_cosine
from src.evaluation import compute_confidence

def run_inference(img1_path, img2_path, threshold=0.35, embedder=None):
    """
    Run the full inference pipeline for one pair of images.
    Returns a dictionary with result details.
    """
    start_total = time.time()
    
    # 1. Preprocessing & Embedding Extraction
    start_emb = time.time()
    if embedder is None:
        embedder = FaceEmbedder(model_name="Facenet")
        
    emb1 = embedder.compute_embedding(img1_path)
    emb2 = embedder.compute_embedding(img2_path)
    latency_emb = time.time() - start_emb
    
    # 2. Similarity Scoring
    start_scoring = time.time()
    # Reshape for vectorized function
    score = numpy_vectorized_cosine(emb1.reshape(1, -1), emb2.reshape(1, -1))[0]
    latency_scoring = time.time() - start_scoring
    
    # 3. Decision
    decision = "SAME" if score >= threshold else "DIFFERENT"
    is_same = (score >= threshold)
    
    # 4. Calibrated Confidence
    confidence = compute_confidence(score, threshold, k=10, score_is_distance=False)
    
    latency_total = time.time() - start_total
    
    return {
        "img1": img1_path,
        "img2": img2_path,
        "similarity_score": round(float(score), 4),
        "threshold": threshold,
        "decision": decision,
        "is_same": int(is_same),
        "confidence": round(confidence, 4),
        "latency_total_ms": round(latency_total * 1000, 2),
        "latency_emb_ms": round(latency_emb * 1000, 2),
        "latency_scoring_ms": round(latency_scoring * 1000, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="FaceID Inference CLI (Milestone 3)")
    parser.add_argument("--img1", type=str, help="Path to first face image")
    parser.add_argument("--img2", type=str, help="Path to second face image")
    parser.add_argument("--threshold", type=float, default=0.35, help="Similarity threshold (default: 0.35)")
    parser.add_argument("--batch", type=str, help="Path to batch CSV file (with left_path, right_path)")
    
    args = parser.parse_args()
    
    if args.batch:
        import csv
        if not os.path.exists(args.batch):
            print(f"Error: Batch file {args.batch} not found.")
            return
            
        print(f"{'Img1':<40} | {'Img2':<40} | {'Score':<8} | {'Decision':<10} | {'Conf':<6} | {'Lat(ms)':<8}")
        print("-" * 130)
        
        with open(args.batch, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                res = run_inference(row['left_path'], row['right_path'], args.threshold)
                print(f"{os.path.basename(res['img1']):<40} | {os.path.basename(res['img2']):<40} | {res['similarity_score']:<8.4f} | {res['decision']:<10} | {res['confidence']:<6.4f} | {res['latency_total_ms']:<8.2f}")
    
    elif args.img1 and args.img2:
        res = run_inference(args.img1, args.img2, args.threshold)
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
