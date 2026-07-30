---
title: FaceID - Real-Time Recognition & Verification Suite
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
short_description: Real-time face recognition & verification engine demo
pinned: false
license: mit
---

# FaceID Recognition Engine - Hugging Face Space Edition

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Project Overview
This repository contains a professional Face Verification system. Milestone 4 represents the final "Release" version, featuring a FaceNet-based inference pipeline, comprehensive hardware profiling, and a professional System Card.

## 🚀 Quick Start & Grader Instructions (Final Release)

### 1. Local Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install tf-keras
```

### 2. Run Inference CLI (Copy-Pastable)
You can run the verification CLI using the final calibrated threshold (0.35). 
*Note: We have provided sample images in the repository for quick testing.*

**Test Same Identity (Should output SAME):**
```bash
python scripts/inference.py --img1 reports/roc_curve.png --img2 reports/roc_curve.png
```

**Test Different Identities (Should output DIFFERENT):**
```bash
python scripts/inference.py --img1 reports/roc_curve.png --img2 reports/false_positives_examples.png
```

### 3. Interactive Web Dashboard
Explore the system, view profiling charts, and test custom images via the Streamlit dashboard:
```bash
streamlit run scripts/app.py
```

### 4. Dockerized Deployment (Alternative)
To verify the system in an isolated container:
```bash
# Build the image
docker build -t faceid-final .

# Run inference inside the container
docker run --rm faceid-final --img1 reports/roc_curve.png --img2 reports/false_positives_examples.png
```

## 📊 Final Documentation (Milestone 4)
- **[System Card](reports/System_Card.md)**: Detailed overview of model design, intended use, and fairness.
- **[Profiling Report](reports/Profiling_Report.md)**: Breakdown of CPU latency and batch-size sensitivity.
- **[Reproducibility Checklist](reports/Reproducibility_Checklist.md)**: Step-by-step guide to reproduce all project results.

## Key Features
- **State-of-the-Art Representations**: Uses FaceNet (InceptionResNetV1) for robust face verification.
- **Calibrated Confidence**: Similarity scores are mapped to a human-readable confidence interval [0.5, 1.0].
- **Production Ready**: Fully containerized via Docker with optimized CPU performance.
- **Hardware Aware**: Detailed profiling provided for deployment planning.

## Repository Layout
- `configs/`: Final release configuration (`eval_ms4_final.yaml`).
- `reports/`: System Card, Profiling Report, and Checklists.
- `scripts/`: Final inference, profiling, and evaluation entry points.
- `src/`: Core logic for embeddings, similarity, and evaluation.
- `Dockerfile`: Containerization setup for final release.

## Reproducibility
This project follows strict reproducibility standards. To verify the system from a fresh clone, please refer to the [Reproducibility Checklist](reports/Reproducibility_Checklist.md).

## Versioning
- **Current Version**: `v1.0-final` (Tagged in Git)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
