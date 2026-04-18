"""
Smoke tests for Milestone 3 inference and load-test paths.
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.inference import DEFAULT_CONFIG_PATH, load_inference_config, run_inference
import scripts.load_test as load_test_module


class DummyEmbedder:
    """Deterministic embedder used to avoid model downloads in tests."""

    def __init__(self, model_name='Dummy'):
        self.model_name = model_name

    def compute_embedding(self, image_path):
        seed = sum(ord(ch) for ch in os.path.basename(image_path))
        rng = np.random.default_rng(seed)
        return rng.random(16, dtype=np.float32)


def test_inference_config_loads_defaults():
    config = load_inference_config(DEFAULT_CONFIG_PATH)
    assert 'model_name' in config
    assert 'threshold' in config
    assert 'confidence_k' in config
    assert 'score_is_distance' in config
    assert isinstance(config['threshold'], float)
    assert isinstance(config['confidence_k'], float)


def test_run_inference_smoke_dummy_embedder():
    embedder = DummyEmbedder()
    result = run_inference(
        'left.jpg',
        'right.jpg',
        threshold=0.40,
        model_name='Dummy',
        confidence_k=10.0,
        score_is_distance=False,
        embedder=embedder,
    )

    expected_keys = {
        'img1',
        'img2',
        'similarity_score',
        'threshold',
        'decision',
        'is_same',
        'confidence',
        'model_name',
        'latency_total_ms',
        'latency_emb_ms',
        'latency_scoring_ms',
    }
    assert expected_keys.issubset(result.keys())
    assert result['decision'] in ('SAME', 'DIFFERENT')
    assert result['is_same'] in (0, 1)
    assert 0.5 <= result['confidence'] <= 1.0


def test_load_test_writes_runtime_summary(tmp_path, monkeypatch):
    pairs_csv = tmp_path / 'pairs.csv'
    pairs_csv.write_text(
        'left_path,right_path,label\n'
        'a.jpg,b.jpg,0\n'
        'a.jpg,a.jpg,1\n'
    )

    def fake_run_inference(*_args, **_kwargs):
        return {'latency_total_ms': 12.5}

    monkeypatch.setattr(load_test_module, 'run_inference', fake_run_inference)

    summary_path = tmp_path / 'summary.json'
    summary = load_test_module.load_test(
        str(pairs_csv),
        concurrency=2,
        num_requests=4,
        threshold=0.4,
        model_name='Dummy',
        confidence_k=10.0,
        score_is_distance=False,
        output_path=str(summary_path),
    )

    assert summary_path.exists()
    saved = json.loads(summary_path.read_text())
    assert saved['total_requests'] == 4
    assert saved['failure_count'] == 0
    assert 'throughput_rps' in saved
    assert 'latency_ms' in saved
    assert 'p95' in saved['latency_ms']
    assert summary['total_requests'] == 4
