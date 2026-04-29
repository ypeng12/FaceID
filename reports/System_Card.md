# System Card: FaceID Verification Engine

## 1. System Overview
**FaceID** is a facial verification system designed to determine if two facial images belong to the same identity. It uses deep representation learning to map facial features into a 128-dimensional embedding space where similarity is measured via cosine distance.

- **Model Architecture**: FaceNet (InceptionResNetV1)
- **Framework**: DeepFace / TensorFlow
- **Dataset (Training)**: Pre-trained on MS-Celeb-1M
- **Inference Pipeline**: Preprocessing -> Feature Extraction -> Cosine Similarity -> Calibrated Confidence

## 2. Intended Use
- **Primary Use Case**: Identity verification for secure access or identity confirmation (1:1 matching).
- **Intended Users**: System administrators and developers integrating facial verification into applications.
- **Environment**: Containerized (Docker) or local CLI/Web interface.

## 3. Out-of-Scope Uses
- **Surveillance**: Not designed for real-time identification from large crowds (1:N search).
- **Demographic Classification**: Not intended to predict age, gender, or ethnicity.
- **High-Stakes Legal Decisions**: Should not be the sole basis for legal or judicial outcomes without human oversight.

## 4. Performance Metrics (Final Release)
Evaluated on the LFW (Labeled Faces in the Wild) validation subset.

- **Operating Threshold**: 0.35 (Cosine Similarity)
- **Accuracy**: 84.6%
- **F1-Score**: 0.8254
- **Avg. Latency (CPU)**: ~0.47s per image (single-pair pipeline ~0.94s)

## 5. Limitations and Failure Modes
- **Lighting**: Performance degrades in extreme low-light or high-contrast shadows.
- **Pose**: High-angle profiles or partially occluded faces (masks, sunglasses) significantly increase the False Negative Rate.
- **Resolution**: Low-resolution images ( < 64x64 pixels) result in unreliable embeddings.
- **Similarity in Identity**: High similarity scores may occur between look-alikes or biological relatives (False Positives).

## 6. Fairness and Risks
- **Demographic Bias**: Like most face recognition systems, performance may vary across different ethnicities and age groups. While evaluated on LFW, users should perform their own local fairness audits if deploying in diverse populations.
- **Privacy**: The system processes sensitive biometric data. Developers must ensure compliance with data protection regulations (e.g., GDPR, BIPA).

## 7. Operational Constraints
- **Hardware**: Minimum 4GB RAM required. CPU-only inference takes ~1.7s; GPU acceleration is recommended for batch processing.
- **Input Format**: Supports standard formats (JPG, PNG). Images must contain a detectable human face.

## 8. Reproducibility
The final release is tagged as `v1.0-final`. Full instructions to reproduce results are available in the [Reproducibility Checklist](../reports/Reproducibility_Checklist.md).
