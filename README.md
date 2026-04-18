Face Verification Engine - Milestone 3

## Project Overview
This repository operates a Face Verification evaluation pipeline powered by LFW. Milestone 3 transforms the prototype into a **deployable inference system**. We upgraded the representation from a weak baseline to **FaceNet (InceptionResNetV1)** embeddings, implemented a clean **Inference CLI**, added **calibrated confidence** metrics, and packaged the system via **Docker**.

## Milestone 3 Features
- **Embedding-based Inference**: Uses `DeepFace` (Facenet) for high-quality facial representations.
- **Inference CLI**: Clean user interface for single or batch pair verification.
- **Calibrated Confidence**: Maps similarity scores to a [0.5, 1.0] range using a sigmoid transformation relative to the operating threshold.
- **Runtime Characterization**: Load-test script to measure throughput and p95 latency.
- **Dockerized Deployment**: Reproducible environment for system evaluation and inference.

## How to Run (Milestone 3)

### 1. Local Environment Setup
```bash
pip install -r requirements.txt
pip install tf-keras  # Required for DeepFace with TensorFlow 2.21+
```

### 2. Inference CLI
Run verification on a single pair:
```bash
python scripts/inference.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg
```

Run batch inference from a CSV:
```bash
python scripts/inference.py --batch outputs/pairs_v2/val_subset.csv
```

### 3. Load Testing
Simulate concurrent requests and report throughput/latency:
```bash
python scripts/load_test.py --requests 20 --concurrency 4
```

### 4. Docker Deployment
Build the image:
```bash
docker build -t faceid-m3 .
```

Run inference inside the container:
```bash
docker run --rm -v ${PWD}/data:/app/data faceid-m3 --img1 data/lfw/test/Aaron_Patterson/Aaron_Patterson_0000.jpg --img2 data/lfw/test/Aaron_Patterson/Aaron_Patterson_0000.jpg
```

## Repository Layout
- `src/`: Core Python modules (Upgraded `embedding.py` to use FaceNet).
- `scripts/`: Added `inference.py` and `load_test.py`.
- `Dockerfile`: Containerization setup.
- `reports/`: `Milestone3_Walkthrough.md`.

## Milestone 2 Reproducibility
(Previous instructions remain valid for evaluating the Milestone 2 backbone).

## Tests
Execute checks for both evaluation and inference paths:
```bash
python -m pytest tests/
```
