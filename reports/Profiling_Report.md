# FaceID Hardware-Aware Profiling Report

This report documents the runtime characteristics of the FaceID system (Milestone 4) on a CPU baseline.

## 1. Measurement Environment
- **Processor**: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
- **RAM**: 16 GB
- **OS**: macOS
- **Model**: FaceNet (InceptionResNetV1)
- **Backend**: DeepFace with TensorFlow

## 2. Latency Breakdown
Measured over 5 iterations of a single pair verification.

| Stage | Mean Latency (ms) | p95 Latency (ms) |
| :--- | :--- | :--- |
| Preprocessing (Detect/Align) | 170.54 | 227.79 |
| Embedding Generation | 237.70 | 267.43 |
| Similarity Scoring | 0.17 | 0.45 |
| **Total Pipeline** | **408.41** | **495.67** |

**Interpretation**: 
- Embedding generation (FaceNet inference) typically dominates the latency.
- Preprocessing (OpenCV-based face detection) is relatively fast but essential for alignment.
- Similarity scoring is negligible compared to model inference.

## 3. Batch-Size Sensitivity
Measured throughput across different batch sizes.

| Batch Size | Total Latency (ms) | Latency per Image (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- |
| 1 | 235.87 | 235.87 | 4.24 |
| 4 | 929.08 | 232.27 | 4.31 |
| 8 | 2016.02 | 252.00 | 3.97 |
| 16 | 3904.67 | 244.04 | 4.10 |

**Observations**:
- Throughput is relatively stable across batch sizes on this CPU, hovering around 4 FPS.
- Unlike GPU environments, larger batch sizes on CPU do not show massive speedups, as the bottleneck remains the per-image inference computation.

## 4. Operational Constraints
- **CPU Bound**: The system is highly CPU-dependent. For real-time applications ( > 30 FPS), a GPU or a smaller model (like MobileFaceNet) would be required.
- **Warm-up Time**: The first request after startup has a high latency overhead. Caching the model in memory is essential for interactive use.
