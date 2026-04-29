import pytest
import numpy as np
import os
import sys
from PIL import Image

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.similarity import numpy_vectorized_cosine
from src.evaluation import compute_confidence
from src.embedding import FaceEmbedder

def test_similarity_zero_vectors():
    """测试当输入为零向量时，相似度引擎的行为（应处理除以零的情况）"""
    a = np.zeros((1, 128))
    b = np.random.rand(1, 128)
    # Cosine similarity with zero vector should be 0 or handle error
    # Our implementation uses dot / (norm*norm)
    score = numpy_vectorized_cosine(a, b)
    assert score[0] == 0

def test_confidence_boundary_conditions():
    """测试置信度校准的边界值"""
    threshold = 0.35
    
    # 刚好等于阈值
    conf_at_th = compute_confidence(threshold, threshold)
    assert 0.45 < conf_at_th < 0.55 # 应接近 0.5
    
    # 极高相似度
    conf_high = compute_confidence(0.99, threshold)
    assert conf_high > 0.95
    
    # 极低相似度
    conf_low = compute_confidence(0.01, threshold)
    assert conf_low > 0.90 # 即使判定不同，置信度也应该很高（因为它很确定不同）

def test_invalid_image_handling(tmp_path):
    """测试加载非法/损坏图片时的鲁棒性"""
    embedder = FaceEmbedder(model_name="Facenet")
    
    # 1. 不存在的文件
    with pytest.raises(FileNotFoundError):
        embedder.compute_embedding("non_existent.jpg")
        
    # 2. 损坏的文件 (创建一个非图片格式的伪图片)
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_text("this is not an image")
    
    with pytest.raises(Exception):
        embedder.compute_embedding(str(corrupt_file))

def test_embedding_determinism():
    """测试模型的确定性：同一张图两次生成的向量必须完全一致"""
    embedder = FaceEmbedder(model_name="Facenet")
    # Using a real sample if it exists, otherwise skip
    sample = "data/lfw/test/Alex_Ferguson/Alex_Ferguson_0000.jpg"
    if os.path.exists(sample):
        emb1 = embedder.compute_embedding(sample)
        emb2 = embedder.compute_embedding(sample)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)
