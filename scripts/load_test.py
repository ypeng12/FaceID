import argparse
import json
import os
import time
import sys
import numpy as np
import csv
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.inference import run_inference, load_inference_config, DEFAULT_CONFIG_PATH


def load_test(
    pairs_csv,
    concurrency=4,
    num_requests=20,
    threshold=0.40,
    model_name='Facenet',
    confidence_k=10.0,
    score_is_distance=False,
    output_path='outputs/runtime/load_test_summary.json',
):
    """
    Simulate concurrent inference requests.
    """
    pairs = []
    with open(pairs_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['left_path'], row['right_path']))
    
    if not pairs:
        print("Error: No pairs found for load test.")
        return
        
    # Limit requests if needed
    test_pairs = pairs[:num_requests]
    if len(test_pairs) < num_requests:
        # Recycle pairs if needed
        test_pairs = (test_pairs * (num_requests // len(test_pairs) + 1))[:num_requests]
        
    print(f"Starting Load Test: {num_requests} requests with concurrency={concurrency}")
    
    latencies = []
    success_count = 0
    failure_count = 0
    failure_examples = []
    
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_inference,
                p[0],
                p[1],
                threshold,
                model_name,
                confidence_k,
                score_is_distance,
            )
            for p in test_pairs
        ]
        
        for future in as_completed(futures):
            try:
                res = future.result()
                latencies.append(res['latency_total_ms'])
                success_count += 1
            except Exception as e:
                print(f"Request failed: {e}")
                failure_count += 1
                if len(failure_examples) < 5:
                    failure_examples.append(str(e))
                
    total_time = time.perf_counter() - start_time
    
    # Calculate stats
    if latencies:
        avg_latency = float(np.mean(latencies))
        p50_latency = float(np.percentile(latencies, 50))
        p95_latency = float(np.percentile(latencies, 95))
        p99_latency = float(np.percentile(latencies, 99))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))
        throughput = success_count / total_time
    else:
        avg_latency = p50_latency = p95_latency = p99_latency = 0.0
        min_latency = max_latency = 0.0
        throughput = 0.0

    summary = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'pairs_csv': pairs_csv,
        'concurrency': int(concurrency),
        'total_requests': int(num_requests),
        'success_count': int(success_count),
        'failure_count': int(failure_count),
        'failure_examples': failure_examples,
        'total_time_s': round(float(total_time), 4),
        'throughput_rps': round(float(throughput), 4),
        'threshold': float(threshold),
        'model_name': str(model_name),
        'confidence_k': float(confidence_k),
        'score_is_distance': bool(score_is_distance),
        'latency_ms': {
            'avg': round(avg_latency, 4),
            'p50': round(p50_latency, 4),
            'p95': round(p95_latency, 4),
            'p99': round(p99_latency, 4),
            'min': round(min_latency, 4),
            'max': round(max_latency, 4),
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
    print("\n--- Load Test Results ---")
    print(f"Total Requests:   {num_requests}")
    print(f"Success Count:    {success_count}")
    print(f"Failure Count:    {failure_count}")
    print(f"Total Time:       {total_time:.2f}s")
    print(f"Throughput:       {throughput:.2f} req/s")
    print(f"Avg Latency:      {avg_latency:.2f} ms")
    print(f"p95 Latency:      {p95_latency:.2f} ms")
    print(f"p99 Latency:      {p99_latency:.2f} ms")
    if output_path:
        print(f"Summary JSON:     {output_path}")
    print("-------------------------\n")

    return summary

def main():
    parser = argparse.ArgumentParser(description="FaceID Load Test CLI (Milestone 3)")
    parser.add_argument("--pairs", type=str, default="outputs/pairs_v2/val.csv", help="Pairs CSV for test data")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=20, help="Total number of requests to perform")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Inference config used for model/threshold settings.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override. If omitted, uses config threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/runtime/load_test_summary.json",
        help="Path for JSON summary output.",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pairs):
        print(f"Error: Pairs file {args.pairs} not found. Run make_pairs.py or recalibrate first.")
        return

    runtime_config = load_inference_config(args.config)
    threshold = args.threshold if args.threshold is not None else runtime_config['threshold']

    load_test(
        args.pairs,
        concurrency=args.concurrency,
        num_requests=args.requests,
        threshold=threshold,
        model_name=runtime_config['model_name'],
        confidence_k=runtime_config['confidence_k'],
        score_is_distance=runtime_config['score_is_distance'],
        output_path=args.output,
    )

if __name__ == "__main__":
    main()
