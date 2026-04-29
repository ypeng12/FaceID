# FaceID Hardware-Aware Profiling Report

This report documents the runtime characteristics of the FaceID system (Milestone 4) on a CPU baseline.

## 1. Measurement Environment
- **Processor**: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD
- **RAM**: 31.19 GB
- **OS**: Windows 11 (as per environment)
- **Model**: FaceNet (InceptionResNetV1)
- **Backend**: DeepFace with TensorFlow

## 2. Latency Breakdown
Measured over 5 iterations of a single pair verification.

| Stage | Mean Latency (ms) | p95 Latency (ms) |
| :--- | :--- | :--- |
| Embedding Generation | 464.76 | 495.87 |
| Similarity Scoring | 0.15 | 0.24 |
| **Total Pipeline** | **464.91** | **496.11** |

**Interpretation**: 
- Embedding generation (FaceNet inference) dominates 99.9% of the total latency. 
- Once the model is warm, latency stabilizes around 465ms per image.

## 3. Batch-Size Sensitivity
Measured throughput across different batch sizes.

| Batch Size | Total Latency (ms) | Latency per Image (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- |
| 1 | 476.36 | 476.36 | 2.10 |
| 4 | 1973.85 | 493.46 | 2.03 |
| 8 | 3921.93 | 490.24 | 2.04 |
| 16 | 7279.01 | 454.94 | 2.20 |

**Observations**:
- Throughput is relatively stable across batch sizes on this CPU, with a slight peak at batch size 16 (~2.20 FPS).
- Unlike GPU environments, larger batch sizes on CPU do not show massive speedups, as the bottleneck remains the per-image inference computation.

## 4. Operational Constraints
- **CPU Bound**: The system is highly CPU-dependent. For real-time applications ( > 30 FPS), a GPU or a smaller model (like MobileFaceNet) would be required.
- **Warm-up Time**: The first request after startup has a high latency overhead. Caching the model in memory is essential for interactive use.
