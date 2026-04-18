import argparse
import os
import time
import sys
import numpy as np
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.inference import run_inference

def load_test(pairs_csv, concurrency=4, num_requests=20):
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
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_inference, p[0], p[1]) for p in test_pairs]
        
        for future in as_completed(futures):
            try:
                res = future.result()
                latencies.append(res['latency_total_ms'])
                success_count += 1
            except Exception as e:
                print(f"Request failed: {e}")
                failure_count += 1
                
    total_time = time.time() - start_time
    
    # Calculate stats
    if latencies:
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        throughput = success_count / total_time
    else:
        avg_latency = p95_latency = p99_latency = throughput = 0
        
    print("\n--- Load Test Results ---")
    print(f"Total Requests:   {num_requests}")
    print(f"Success Count:    {success_count}")
    print(f"Failure Count:    {failure_count}")
    print(f"Total Time:       {total_time:.2f}s")
    print(f"Throughput:       {throughput:.2f} req/s")
    print(f"Avg Latency:      {avg_latency:.2f} ms")
    print(f"p95 Latency:      {p95_latency:.2f} ms")
    print(f"p99 Latency:      {p99_latency:.2f} ms")
    print("-------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="FaceID Load Test CLI (Milestone 3)")
    parser.add_argument("--pairs", type=str, default="outputs/pairs_v2/val_subset.csv", help="Pairs CSV for test data")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=20, help="Total number of requests to perform")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pairs):
        print(f"Error: Pairs file {args.pairs} not found. Run make_pairs.py or recalibrate first.")
        return
        
    load_test(args.pairs, args.concurrency, args.requests)

if __name__ == "__main__":
    main()
