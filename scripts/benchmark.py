import argparse
import time
import numpy as np
import sys
import os

# Add src to the path so we can import similarity
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.similarity import (
    python_loop_cosine,
    python_loop_euclidean,
    numpy_vectorized_cosine,
    numpy_vectorized_euclidean
)

def main():
    parser = argparse.ArgumentParser(description="Benchmark similarity functions.")
    parser.add_argument("--N", type=int, default=5000, help="Number of vector pairs.")
    parser.add_argument("--D", type=int, default=128, help="Dimensionality of vectors.")
    parser.add_argument("--output", type=str, default="outputs/bench/results.txt", help="Output file for results.")
    args = parser.parse_args()

    print(f"Generating {args.N} pairs of {args.D}-dimensional vectors...")
    np.random.seed(42)
    a = np.random.rand(args.N, args.D)
    b = np.random.rand(args.N, args.D)

    # Benchmark Cosine
    print("--- Cosine Similarity Benchmark ---")
    t0 = time.perf_counter()
    loop_cos = python_loop_cosine(a, b)
    t_loop_cos = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_cos = numpy_vectorized_cosine(a, b)
    t_vec_cos = time.perf_counter() - t0

    cos_diff = np.max(np.abs(loop_cos - vec_cos))
    
    # Benchmark Euclidean
    print("--- Euclidean Distance Benchmark ---")
    t0 = time.perf_counter()
    loop_euc = python_loop_euclidean(a, b)
    t_loop_euc = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_euc = numpy_vectorized_euclidean(a, b)
    t_vec_euc = time.perf_counter() - t0

    euc_diff = np.max(np.abs(loop_euc - vec_euc))

    # Correctness assertions
    TOLERANCE = 1e-6
    cos_pass = cos_diff < TOLERANCE
    euc_pass = euc_diff < TOLERANCE
    assert cos_pass, f"Cosine correctness FAILED: max diff {cos_diff:.6e} >= {TOLERANCE}"
    assert euc_pass, f"Euclidean correctness FAILED: max diff {euc_diff:.6e} >= {TOLERANCE}"

    results = []
    results.append(f"Benchmark Configuration: N={args.N}, D={args.D}\n")
    results.append("Cosine Similarity:")
    results.append(f"  Python Loop Time: {t_loop_cos:.4f} s")
    results.append(f"  NumPy Vectorized Time: {t_vec_cos:.4f} s")
    results.append(f"  Speedup: {t_loop_cos / t_vec_cos:.2f}x")
    results.append(f"  Max Absolute Difference: {cos_diff:.6e}")
    results.append(f"  Correctness Check: {'PASS' if cos_pass else 'FAIL'} (tolerance={TOLERANCE})\n")
    
    results.append("Euclidean Distance:")
    results.append(f"  Python Loop Time: {t_loop_euc:.4f} s")
    results.append(f"  NumPy Vectorized Time: {t_vec_euc:.4f} s")
    results.append(f"  Speedup: {t_loop_euc / t_vec_euc:.2f}x")
    results.append(f"  Max Absolute Difference: {euc_diff:.6e}")
    results.append(f"  Correctness Check: {'PASS' if euc_pass else 'FAIL'} (tolerance={TOLERANCE})\n")

    results.append(f"Overall Correctness: {'ALL PASS' if (cos_pass and euc_pass) else 'FAIL'}\n")
    
    results_str = "\n".join(results)
    print(results_str)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(results_str)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
