import numpy as np
from typing import List, Dict, Any, Optional
import os

class FaceGallery:
    """In-memory Face Identification Gallery for 1:N face searching."""
    
    def __init__(self):
        self.identities: List[Dict[str, Any]] = []

    def enroll(self, name: str, embedding: np.ndarray, image_path: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        """Enroll a new identity with name, face embedding vector, and image reference."""
        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(embedding)
        norm_emb = embedding / (norm + 1e-10) if norm > 0 else embedding
        
        self.identities.append({
            "name": name,
            "embedding": norm_emb,
            "image_path": image_path,
            "metadata": metadata or {}
        })

    def search(self, query_embedding: np.ndarray, top_k: int = 3, threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Search query embedding against all enrolled identities.
        Returns top_k results sorted by cosine similarity descending.
        """
        if not self.identities:
            return []
            
        norm = np.linalg.norm(query_embedding)
        q_norm = query_embedding / (norm + 1e-10) if norm > 0 else query_embedding
        
        gallery_embs = np.vstack([id_dict["embedding"] for id_dict in self.identities]) # (N, D)
        similarities = np.dot(gallery_embs, q_norm) # (N,)
        
        results = []
        for idx, score in enumerate(similarities):
            item = self.identities[idx]
            sim_score = float(score)
            results.append({
                "name": item["name"],
                "similarity": sim_score,
                "is_match": bool(sim_score >= threshold),
                "image_path": item["image_path"],
                "metadata": item.get("metadata", {})
            })
            
        # Sort descending by similarity score
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        self.identities.clear()

    def count(self) -> int:
        return len(self.identities)
