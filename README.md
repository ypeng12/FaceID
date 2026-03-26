Face Verification Engine - Milestone 2

## Project Overview
This repository operates a Face Verification evaluation pipeline powered by LFW. Milestone 2 replaces the static testings of Milestone 1 with a full machine-learning evaluation loop system. Using MobileNetV2 embedding extractors, the pipeline dynamically measures threshold tradeoff performance over different splits via a JSON tracking engine.

### Data-Centric Improvement
We observed that random pair extraction across uniform identities led to identical pairs being redundantly chosen for rare faces (e.g., identities with 2 total images). A deterministic tracking policy `enforce_unique` ensures positive and negative pairs are absolutely non-duplicated across tests, driving lower redundancy evaluations.

## Repository Layout
- `src/`: Core Python modules (`embedding.py`, `evaluation.py`, `similarity.py`, `tracking.py`).
- `scripts/`: Entrypoints for evaluation (`run_evaluation.py`) and dataset processing.
- `configs/`: YAML configurations tracking sweeps, data sources, and paired thresholds.
- `tests/`: End-to-end integration and computation unit tests (`test_evaluation.py`).
- `outputs/`: Metrics logs (`runs.json`) and processed datasets (Not Tracked).
- `reports/`: `Milestone2_Report.md` and visualizations showcasing confusion data and ROC sweeps.

## How to run (Milestone 2 Reproducibility)

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ingest LFW and Generate Baseline Pairs**
   ```bash
   python scripts/ingest_lfw.py --config configs/data_config.yaml
   python scripts/make_pairs.py --config configs/pairs_config.yaml
   ```

3. **Baseline Eval Pipeline (Runs 1-3)**
   Sweep validation pairs for max F1 threshold, and apply it to standard evaluations:
   ```bash
   python scripts/run_evaluation.py --config configs/eval_config.yaml
   python scripts/run_evaluation.py --config configs/eval_val_run2.yaml
   python scripts/run_evaluation.py --config configs/eval_test_run3.yaml
   ```

4. **Data-Centric Data Refactoring**
   Extract the unique pairs ensuring zero pairwise redundancy:
   ```bash
   python scripts/make_pairs.py --config configs/pairs_config_v2.yaml
   ```

5. **Post-Change Pipeline (Runs 4-5)**
   ```bash
   python scripts/run_evaluation.py --config configs/eval_val_sweep_v2.yaml
   python scripts/run_evaluation.py --config configs/eval_test_eval_v2.yaml
   ```

6. **Generate Report**
   *After tracking metrics have been fully rendered to `outputs/eval/`*:
   ```bash
   python scripts/generate_report.py
   ```

## Tests
Validate the complete setup by executing local integration and unit checks:
```bash
python -m pytest tests/test_evaluation.py -v
```
