import pytest
import numpy as np
from src.gallery import FaceGallery

def test_gallery_enroll_and_search():
    gallery = FaceGallery()
    assert gallery.count() == 0
    
    # Fake 128D embeddings
    emb_alice = np.random.randn(128).astype(np.float32)
    emb_bob = np.random.randn(128).astype(np.float32)
    
    gallery.enroll("Alice", emb_alice)
    gallery.enroll("Bob", emb_bob)
    
    assert gallery.count() == 2
    
    # Search with exact Alice embedding
    results = gallery.search(emb_alice, top_k=2, threshold=0.5)
    assert len(results) == 2
    assert results[0]["name"] == "Alice"
    assert results[0]["similarity"] > 0.99
    assert results[0]["is_match"] is True

def test_gallery_clear():
    gallery = FaceGallery()
    gallery.enroll("Alice", np.zeros(128))
    gallery.clear()
    assert gallery.count() == 0
