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
            
        objs = DeepFace.represent(
            img_path=image_path, 
            model_name=self.model_name,
            enforce_detection=False,
            detector_backend="opencv",
            align=True
        )
        if not objs:
            return np.zeros(128)
            
        emb = np.array(objs[0]["embedding"], dtype=np.float32)
        return emb

    def extract_face_details(self, image_input) -> dict:
        """Extract embedding, facial bounding box, and cropped face image.
        image_input can be a file path str or numpy ndarray (BGR/RGB image).
        """
        objs = DeepFace.represent(
            img_path=image_input,
            model_name=self.model_name,
            enforce_detection=False,
            detector_backend="opencv",
            align=True
        )
        if not objs:
            return {
                "embedding": np.zeros(128, dtype=np.float32),
                "facial_area": {"x": 0, "y": 0, "w": 0, "h": 0},
                "confidence": 0.0
            }
        
        target = objs[0]
        emb = np.array(target["embedding"], dtype=np.float32)
        facial_area = target.get("facial_area", {"x": 0, "y": 0, "w": 0, "h": 0})
        
        return {
            "embedding": emb,
            "facial_area": facial_area,
            "confidence": target.get("face_confidence", 1.0)
        }

    def batch_compute_embeddings(self, image_paths: list, batch_size: int = 16) -> np.ndarray:
        embeddings = []
        for path in image_paths:
            emb = self.compute_embedding(path)
            embeddings.append(emb)
        return np.vstack(embeddings)

