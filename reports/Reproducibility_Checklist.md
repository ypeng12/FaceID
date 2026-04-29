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
python scripts/inference.py --img1 data/lfw/test/Alex_Ferguson/Alex_Ferguson_0000.jpg --img2 data/lfw/test/Alex_Ferguson/Alex_Ferguson_0000.jpg

# Run a single pair verification (Different)
python scripts/inference.py --img1 data/lfw/test/Alex_Ferguson/Alex_Ferguson_0000.jpg --img2 data/lfw/test/Alex_Barros/Alex_Barros_0000.jpg
```

## 3. Reproduce Profiling Results
```bash
# Run the hardware-aware profiling script
python scripts/profiling.py --iterations 10
# Check output in reports/profiling_summary.txt
```

## 4. Run Final Evaluation
```bash
# Run evaluation with the frozen threshold
python scripts/run_evaluation.py --config configs/eval_ms4_final.yaml
# Results will be in outputs/eval/
```

## 5. Docker Deployment
```bash
# Build the image
docker build -t faceid-final .

# Run inference in container
docker run --rm -v ${PWD}/data:/app/data faceid-final --img1 data/lfw/test/Alex_Ferguson/Alex_Ferguson_0000.jpg --img2 data/lfw/test/Alex_Ferguson/Alex_Ferguson_0000.jpg
```

## 6. Key Artifact Locations
- **System Card**: `reports/System_Card.md`
- **Profiling Report**: `reports/Profiling_Report.md`
- **Final Config**: `configs/eval_ms4_final.yaml`
- **Release Tag**: `v1.0-final`
