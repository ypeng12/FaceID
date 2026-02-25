import numpy as np
import math

# Loop-based implementations
def python_loop_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity using Python loops.
    a, b: shape (N, D)
    Returns: shape (N,)
    """
    N, D = a.shape
    results = np.zeros(N)
    for i in range(N):
        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for j in range(D):
            dot_product += a[i, j] * b[i, j]
            norm_a += a[i, j] ** 2
            norm_b += b[i, j] ** 2
        
        results[i] = dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b) + 1e-10)
    return results

def python_loop_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance using Python loops.
    a, b: shape (N, D)
    Returns: shape (N,)
    """
    N, D = a.shape
    results = np.zeros(N)
    for i in range(N):
        sq_dist = 0.0
        for j in range(D):
            diff = a[i, j] - b[i, j]
            sq_dist += diff ** 2
        results[i] = math.sqrt(sq_dist)
    return results

# Vectorized implementations
def numpy_vectorized_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity using NumPy vectorization.
    a, b: shape (N, D)
    Returns: shape (N,)
    """
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot_product / (norm_a * norm_b + 1e-10)

def numpy_vectorized_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance using NumPy vectorization.
    a, b: shape (N, D)
    Returns: shape (N,)
    """
    return np.linalg.norm(a - b, axis=1)
