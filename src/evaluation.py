import numpy as np

def compute_metrics(scores, labels, threshold, score_is_distance=False):
    """
    If score_is_distance, lower score means more similar. (e.g. Euclidean)
    If not, higher score means more similar. (e.g. Cosine)
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    
    if len(scores) == 0:
        return {}
        
    if score_is_distance:
        preds = (scores < threshold).astype(int)
    else:
        preds = (scores >= threshold).astype(int)
        
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    acc = (tp + tn) / len(labels)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": float(acc),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def get_confusion_matrix(scores, labels, threshold, score_is_distance=False):
    m = compute_metrics(scores, labels, threshold, score_is_distance)
    if not m:
        return {}
    return {
        "tp": m["tp"],
        "fp": m["fp"],
        "tn": m["tn"],
        "fn": m["fn"]
    }

def compute_confidence(score, threshold, k=10, score_is_distance=False):
    """
    Computes a calibrated confidence score in [0.5, 1.0] for the decision.
    Uses a sigmoid transformation centered at the threshold.
    """
    if score_is_distance:
        val = threshold - score
    else:
        val = score - threshold
        
    # prob in [0, 1], 0.5 at val=0 (score=threshold)
    prob = 1.0 / (1.0 + np.exp(-k * val))
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return float(confidence)
