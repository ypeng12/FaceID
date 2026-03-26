import tensorflow as tf
import numpy as np
from PIL import Image

class FaceEmbedder:
    def __init__(self):
        # We use MobileNetV2 without top layer, pooling="avg" gives (None, 1280) vectors
        # Pretrained on ImageNet. A lightweight model to extract features.
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), 
            include_top=False, 
            weights='imagenet', 
            pooling='avg'
        )

    def compute_embedding(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        x = np.array(img, dtype=np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        emb = self.model.predict(x, verbose=0)
        return emb[0]

    def batch_compute_embeddings(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_imgs = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB').resize((224, 224))
                x = np.array(img, dtype=np.float32)
                batch_imgs.append(x)
            batch_x = np.stack(batch_imgs)
            batch_x = tf.keras.applications.mobilenet_v2.preprocess_input(batch_x)
            emb = self.model.predict(batch_x, verbose=0)
            embeddings.append(emb)
        return np.vstack(embeddings)
