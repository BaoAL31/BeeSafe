# BeeSafe Evaluation Results

**Generated:** 2026-05-08 18:24:41

## Metrics Explanation

### Classification & Acoustic Metrics
- **Accuracy**: Percentage of correct predictions across all classes
- **Recall**: Percentage of actual positive cases correctly identified (Infected/Abnormal recall)
- **Loss**: Cross-entropy loss value (lower is better)

### Performance Metrics
- **Latency**: Average inference time per image/audio sample in milliseconds
- **Throughput**: Number of images/audio samples processed per second
- **Peak CUDA Memory**: Maximum GPU memory allocated during inference (MB)
- **Peak RSS**: Maximum Resident Set Size (process memory usage) in MB
- **GPU Memory**: GPU VRAM usage snapshot (used/total MB)

---

## Classification

### Visual (Classification) Results

- **Accuracy**: 79.1%
- **Infected Recall**: 96.0%
- **Loss**: 0.6102
- **Samples**: 3408

**Performance Metrics:**
- Latency: 0.651 ms/img
- Throughput: ~1535 img/s
- Peak CUDA Memory: 85.3 MB
- Peak RSS: 1208.7 MB

---

## Acoustic

### Acoustic Results

- **Accuracy**: 92.4%
- **Abnormal Recall**: 97.6%
- **Loss**: 0.4267
- **Samples**: 5182

**Performance Metrics:**
- Latency: 1.079 ms/sample
- Throughput: ~927 samples/s
- Peak CUDA Memory: 22.3 MB
- Peak RSS: 1165.0 MB

---
