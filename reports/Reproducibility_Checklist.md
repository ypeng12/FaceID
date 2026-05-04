# Reproducibility Checklist (Milestone 4)

Follow these exact steps to reproduce the core results of the FaceID system from a clean clone.

## 1. Environment Setup
```bash
# Clone the repository
git clone <repo_url>
cd FaceID

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install tf-keras
```

## 2. Verify Inference CLI
```bash
# Run a single pair verification (Same)
python scripts/inference.py --img1 reports/roc_curve.png --img2 reports/roc_curve.png

# Run a single pair verification (Different)
python scripts/inference.py --img1 reports/roc_curve.png --img2 reports/false_positives_examples.png
```

## 3. Reproduce Profiling Results
```bash
# Run the hardware-aware profiling script
python scripts/profiling.py --iterations 10
# Check output in reports/profiling_summary.txt
```

## 4. Run Final Evaluation
```bash
# Note: Evaluation requires the full LFW dataset to be ingested.
# Run evaluation with the frozen threshold (requires data/lfw)
python scripts/run_evaluation.py --config configs/eval_ms4_final.yaml
# Results will be in outputs/eval/
```

## 5. Docker Deployment
```bash
# Build the image
docker build -t faceid-final .

# Run inference in container
docker run --rm -v ${PWD}/reports:/app/reports faceid-final --img1 reports/roc_curve.png --img2 reports/false_positives_examples.png
```

## 6. Key Artifact Locations
- **System Card**: `reports/System_Card.md`
- **Profiling Report**: `reports/Profiling_Report.md`
- **Final Config**: `configs/eval_ms4_final.yaml`
- **Release Tag**: `v1.0-final`
