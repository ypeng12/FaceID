Face Verification Engine - Milestone 3

## Project Overview
This repository implements a reproducible face verification pipeline on LFW.
Milestone 3 upgrades the representation to embedding-based inference (FaceNet via DeepFace),
packages the verifier for deployment, and adds runtime characterization under concurrent load.

## Milestone 3 Additions
- Embedding-based inference with FaceNet embeddings.
- Config-driven CLI operating threshold and confidence settings.
- Pair-level CLI output with decision, similarity, confidence, and latency.
- Concurrency/load-test script with throughput and latency distribution metrics.
- Dockerized runnable inference interface.

## Pipeline Summary
1. Deterministic preprocessing and pair generation (Milestone 1 and 2 backbone).
2. Embedding extraction (`FaceEmbedder`).
3. Cosine similarity scoring.
4. Threshold decision.
5. Calibrated confidence computation.
6. Latency measurement and runtime summary reporting.

## Clean Clone Reproducibility (Milestone 3)

### 1. Environment setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Build deterministic data and deterministic pairs
```bash
python scripts/ingest_lfw.py --config configs/data_config.yaml
python scripts/make_pairs.py --config configs/pairs_config_v2.yaml
```

### 3. Recalibrate threshold for embedding-based system
```bash
python scripts/run_evaluation.py --config configs/eval_ms3_recalibrate.yaml
```

This command writes:
- `outputs/eval/sweep_Run6-Milestone3-Val-Sweep.json`
- `outputs/eval/selected_threshold_Run6-Milestone3-Val-Sweep.json`
- `outputs/runs.json`

Inspect selected threshold:
```bash
python -c "import json; d=json.load(open('outputs/eval/selected_threshold_Run6-Milestone3-Val-Sweep.json')); print(d['best_threshold'])"
```

Update `configs/inference_config.yaml` with that threshold, or pass `--threshold` as a CLI override.

### 4. Inference CLI
Single pair:
```bash
python scripts/inference.py --config configs/inference_config.yaml --img1 path/to/img1.jpg --img2 path/to/img2.jpg
```

Batch pairs:
```bash
python scripts/inference.py --config configs/inference_config.yaml --batch outputs/pairs_v2/val.csv
```

### 5. Load testing and runtime summary
```bash
python scripts/load_test.py --pairs outputs/pairs_v2/val.csv --concurrency 4 --requests 20 --config configs/inference_config.yaml --output outputs/runtime/load_test_summary.json
```

### 6. Docker workflow
Build:
```bash
docker build -t faceid-m3 .
```

Run single-pair inference in container:
```bash
docker run --rm -v ${PWD}/data:/app/data faceid-m3 --config configs/inference_config.yaml --img1 data/lfw/test/Aaron_Patterson/Aaron_Patterson_0000.jpg --img2 data/lfw/test/Aaron_Patterson/Aaron_Patterson_0000.jpg
```

## Confidence Definition
Confidence is computed with a sigmoid centered at the operating threshold,
then mirrored to represent certainty of the predicted class:

- Similarity mode: `val = score - threshold`
- Distance mode: `val = threshold - score`
- `prob = 1 / (1 + exp(-k * val))`
- `confidence = max(prob, 1 - prob)`

Output range is `[0.5, 1.0]`, where higher means more certain.

## Tests
Run focused checks, including inference smoke tests:
```bash
python -m pytest tests/test_similarity.py tests/test_evaluation.py tests/test_inference_smoke.py
```

## Key Artifacts
- `configs/inference_config.yaml`: CLI runtime settings.
- `outputs/eval/sweep_*.json`: sweep points for threshold tradeoff analysis.
- `outputs/eval/selected_threshold_*.json`: selected threshold artifact.
- `outputs/runs.json`: tracked run metadata.
- `outputs/runtime/load_test_summary.json`: throughput and latency summary.
