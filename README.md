# MSML 605 - Milestone 1: Face Verification Foundations

## Project Overview
This milestone establishes the reproducible foundations for the Face Verification system. The goal is to build a deterministic pipeline that ingests the LFW dataset, creates train/val/test splits, generates verification pairs, and implements fast, vectorized similarity metrics for embeddings.

## Repository Layout
- `src/`: Core Python modules (e.g., similarity metrics).
- `scripts/`: Command-line entrypoints for dataset ingestion, pair generation, and benchmarking.
- `configs/`: YAML configurations controlling dataset policies and seed settings.
- `outputs/`: Generated artifacts (manifest, pairs CSV files, benchmark results). **(Not committed)**
- `data/`: Downloaded datasets cache. **(Not committed)**

## How to run (Reproducibility)
Execute the following copy-paste commands from the repository root to reproduce the milestone from a fresh clone:

1. **Environment Setup**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data Ingestion**
   This script loads the LFW dataset, creates deterministic splits based on identities, and saves images to `data/lfw`.
   ```bash
   python scripts/ingest_lfw.py --config configs/data_config.yaml
   ```

3. **Pair Generation**
   This script reads the available splits and deterministically generating matching and non-matching face image pairs.
   ```bash
   python scripts/make_pairs.py --config configs/pairs_config.yaml
   ```

4. **Benchmark Similarity Module**
   This script runs a python loop versus numpy vectorized benchmark for distance metrics and saves the output to `outputs/bench/results.txt`.
   ```bash
   python scripts/benchmark.py
   ```

## Outputs
- **Manifest File**: `outputs/manifest.yaml` - details the seed, split policy, and image counts per split.
- **Pair Files**: `outputs/pairs/train.csv`, `val.csv`, `test.csv` - containing paths to the paired image samples.
- **Benchmark Results**: `outputs/bench/results.txt` - comparison of loop versus vectorized logic.

## Determinism
The entire pipeline relies on sorting identities alphabetically to ensure a repeatable ordering. A fixed seed (`42`) is set within `data_config.yaml` and `pairs_config.yaml` which handles the shuffling of identities and the negative/positive sampling of pairs. The manifest guarantees verifiability.
