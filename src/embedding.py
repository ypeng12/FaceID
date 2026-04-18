import numpy as np
from deepface import DeepFace
import os

class FaceEmbedder:
    def __init__(self, model_name="Facenet"):
        self.model_name = model_name
        # Trigger a dummy call to download/load the model if needed
        print(f"Initializing FaceEmbedder with model: {self.model_name}")
        # Note: DeepFace downloads models on first use.

    def compute_embedding(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # DeepFace.represent returns a list (for multiple faces)
        # We assume one face per image for LFW
        objs = DeepFace.represent(
            img_path=image_path, 
            model_name=self.model_name,
            enforce_detection=False, # LFW is mostly aligned, but we can set True if safe
            detector_backend="opencv", # fastest, "mtcnn" is more accurate
            align=True
        )
        if not objs:
            return np.zeros(128) # Generic fallback or handle error
            
        emb = np.array(objs[0]["embedding"], dtype=np.float32)
        return emb

    def batch_compute_embeddings(self, image_paths: list, batch_size: int = 16) -> np.ndarray:
        embeddings = []
        # DeepFace doesn't have a native "batch represent" for list of paths that is significantly faster
        # than sequential calls, but we can iterate.
        for path in image_paths:
            emb = self.compute_embedding(path)
            embeddings.append(emb)
        return np.vstack(embeddings)
