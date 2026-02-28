"""
Unit tests for src/similarity.py
Tests correctness of loop vs vectorized implementations and edge cases.
"""
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.similarity import (
    python_loop_cosine,
    python_loop_euclidean,
    numpy_vectorized_cosine,
    numpy_vectorized_euclidean,
)

TOLERANCE = 1e-7


def test_cosine_loop_vs_vectorized():
    """Loop and vectorized cosine must agree within tolerance."""
    np.random.seed(123)
    a = np.random.rand(100, 64)
    b = np.random.rand(100, 64)
    loop_result = python_loop_cosine(a, b)
    vec_result = numpy_vectorized_cosine(a, b)
    max_diff = np.max(np.abs(loop_result - vec_result))
    assert max_diff < TOLERANCE, f"Cosine max diff {max_diff} exceeds tolerance {TOLERANCE}"


def test_euclidean_loop_vs_vectorized():
    """Loop and vectorized Euclidean must agree within tolerance."""
    np.random.seed(123)
    a = np.random.rand(100, 64)
    b = np.random.rand(100, 64)
    loop_result = python_loop_euclidean(a, b)
    vec_result = numpy_vectorized_euclidean(a, b)
    max_diff = np.max(np.abs(loop_result - vec_result))
    assert max_diff < TOLERANCE, f"Euclidean max diff {max_diff} exceeds tolerance {TOLERANCE}"


def test_cosine_identical_vectors():
    """Cosine similarity of identical vectors should be ~1.0."""
    a = np.ones((5, 10))
    result = numpy_vectorized_cosine(a, a)
    assert np.allclose(result, 1.0, atol=1e-6), f"Expected ~1.0, got {result}"


def test_euclidean_identical_vectors():
    """Euclidean distance of identical vectors should be 0."""
    a = np.ones((5, 10))
    result = numpy_vectorized_euclidean(a, a)
    assert np.allclose(result, 0.0, atol=1e-9), f"Expected 0.0, got {result}"


def test_cosine_orthogonal_vectors():
    """Cosine similarity of orthogonal vectors should be ~0."""
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    result = numpy_vectorized_cosine(a, b)
    assert np.abs(result[0]) < 1e-6, f"Expected ~0, got {result[0]}"


def test_euclidean_known_distance():
    """Euclidean distance for known case: (0,0)-(3,4) = 5."""
    a = np.array([[0.0, 0.0]])
    b = np.array([[3.0, 4.0]])
    result = numpy_vectorized_euclidean(a, b)
    assert np.isclose(result[0], 5.0), f"Expected 5.0, got {result[0]}"


def test_output_shapes():
    """Output shape must be (N,) for input shape (N, D)."""
    a = np.random.rand(20, 32)
    b = np.random.rand(20, 32)
    assert numpy_vectorized_cosine(a, b).shape == (20,)
    assert numpy_vectorized_euclidean(a, b).shape == (20,)
    assert python_loop_cosine(a, b).shape == (20,)
    assert python_loop_euclidean(a, b).shape == (20,)


def test_determinism():
    """Same seed, same inputs -> identical outputs across repeated calls."""
    np.random.seed(999)
    a = np.random.rand(50, 16)
    b = np.random.rand(50, 16)
    r1 = numpy_vectorized_cosine(a, b).copy()
    r2 = numpy_vectorized_cosine(a, b).copy()
    assert np.array_equal(r1, r2), "Vectorized cosine is not deterministic across calls"


if __name__ == "__main__":
    test_cosine_loop_vs_vectorized()
    test_euclidean_loop_vs_vectorized()
    test_cosine_identical_vectors()
    test_euclidean_identical_vectors()
    test_cosine_orthogonal_vectors()
    test_euclidean_known_distance()
    test_output_shapes()
    test_determinism()
    print("All tests passed!")
