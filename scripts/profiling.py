import argparse
import time
import os
import sys
import numpy as np

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embedding import FaceEmbedder
from src.similarity import numpy_vectorized_cosine

def run_profiling(iterations=5, batch_sizes=[1, 4, 8, 16]):
    # Use a dummy image or a real one from LFW if available
    # We'll try to find a real one first
    sample_img = "data/lfw/test/Alex_Ferguson/Alex_Ferguson_0000.jpg"
    if not os.path.exists(sample_img):
        # Fallback to any jpg in data/lfw
        import glob
        jpgs = glob.glob("data/lfw/**/*.jpg", recursive=True)
        if jpgs:
            sample_img = jpgs[0]
        else:
            print("Error: No sample images found in data/lfw for profiling.")
            return

    print(f"Starting Hardware-Aware Profiling...")
    print(f"Sample image: {sample_img}")
    
    embedder = FaceEmbedder(model_name="Facenet")
    
    # 1. Warm-up
    print("Warming up model...")
    _ = embedder.compute_embedding(sample_img)
    
    # 2. Latency Breakdown
    print(f"Measuring latency over {iterations} iterations...")
    emb_latencies = []
    sim_latencies = []
    
    for _ in range(iterations):
        # Embedding Latency
        t0 = time.perf_counter()
        emb1 = embedder.compute_embedding(sample_img)
        emb2 = embedder.compute_embedding(sample_img)
        emb_latencies.append((time.perf_counter() - t0) * 1000 / 2) # Latency per image
        
        # Similarity Latency
        t0 = time.perf_counter()
        _ = numpy_vectorized_cosine(emb1.reshape(1, -1), emb2.reshape(1, -1))
        sim_latencies.append((time.perf_counter() - t0) * 1000)

    mean_emb = np.mean(emb_latencies)
    p95_emb = np.percentile(emb_latencies, 95)
    mean_sim = np.mean(sim_latencies)
    p95_sim = np.percentile(sim_latencies, 95)
    
    # 3. Batch Sensitivity
    print("Measuring batch-size sensitivity...")
    batch_results = []
    for bs in batch_sizes:
        # Create a list of image paths (just repeat the same for profiling)
        paths = [sample_img] * bs
        t0 = time.perf_counter()
        _ = embedder.batch_compute_embeddings(paths, batch_size=bs)
        total_time = (time.perf_counter() - t0) * 1000
        lat_per_img = total_time / bs
        throughput = 1000 / lat_per_img
        batch_results.append({
            "batch_size": bs,
            "total_latency_ms": total_time,
            "latency_per_image_ms": lat_per_img,
            "throughput_fps": throughput
        })

    # Prepare Report
    report = []
    report.append("# FaceID Hardware-Aware Profiling Summary")
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Latency Breakdown (ms)")
    report.append("| Stage | Mean | p95 |")
    report.append("| :--- | :--- | :--- |")
    report.append(f"| Embedding Generation | {mean_emb:.2f} | {p95_emb:.2f} |")
    report.append(f"| Similarity Scoring | {mean_sim:.2f} | {p95_sim:.2f} |")
    
    report.append("\n## Batch Sensitivity")
    report.append("| Batch Size | Total Latency (ms) | Latency/Image (ms) | Throughput (FPS) |")
    report.append("| :--- | :--- | :--- | :--- |")
    for r in batch_results:
        report.append(f"| {r['batch_size']} | {r['total_latency_ms']:.2f} | {r['latency_per_image_ms']:.2f} | {r['throughput_fps']:.2f} |")
    
    report_str = "\n".join(report)
    print("\n" + report_str + "\n")
    
    output_path = "reports/profiling_summary.txt"
    os.makedirs("reports", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_str)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hardware-aware profiling for FaceID.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations for latency measurement.")
    args = parser.parse_args()
    
    run_profiling(iterations=args.iterations)
