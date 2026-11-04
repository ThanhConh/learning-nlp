# Transformer Classification Results

## 1. Multi-Head Attention Configuration

| Heads | d_k | Parameters | Training time (s) | Accuracy | F1 |
|-------|-----|------------|-------------------|----------|---------|
| 1 | 128 | 39,602,049 | 80.39 | 0.8221 | 0.8291 |
| 2 | 64 | 39,602,049 | 81.81 | 0.8253 | 0.8294 |
| 4 | 32 | 39,602,049 | 84.54 | 0.8209 | 0.8300 |
| 8 | 16 | 39,602,049 | 89.94 | 0.8081 | 0.8192 |

**Best configuration**: 4 heads with d_k=32 (F1: 0.8300)

---

## 2. Model Comparison: LSTM vs Transformer

| Model | Parameters | Training time (s) | Accuracy | Precision | Recall | F1 |
|-------|------------|-------------------|----------|-----------|--------|---------|
| LSTM | 30,659,651 | 45.18 | 0.8085 | 0.8213 | 0.7844 | 0.8024 |
| Transformer | 30,871,901 | 75.55 | 0.8158 | 0.7956 | 0.8457 | 0.8199 |

**Winner**: Transformer (F1: 0.8199 vs 0.8024)
- +0.0175 improvement in F1 score
- +30.37s longer training time
- Better recall (+0.0613)

---

## 3. Sequence Length Impact

| Length | Tokens retained (%) | Attention memory (MB) | Training time (s) | Accuracy | F1 |
|--------|---------------------|----------------------|-------------------|----------|---------|
| 32 | 14.00 | 2.0 | 54.20 | 0.7004 | 0.7365 |
| 64 | 27.65 | 8.0 | 60.44 | 0.7651 | 0.7773 |
| 128 | 52.03 | 32.0 | 84.61 | 0.8177 | 0.8295 |
| 256 | 77.55 | 128.0 | 142.29 | 0.8486 | 0.8567 |

**Best configuration**: Length 256 (F1: 0.8567)
- Significant improvement with longer sequences
- Trade-off: 64x more memory and ~3x longer training time vs length 32

---

## 4. Pooling Strategy Comparison

| Pooling | Parameters | Training time (s) | Accuracy | Precision | Recall | F1 |
|---------|------------|-------------------|----------|-----------|--------|---------|
| MEAN | 39,602,177 | 85.07 | 0.8255 | 0.8283 | 0.8175 | 0.8229 |
| CLS | 39,602,177 | 84.05 | 0.8046 | 0.7749 | 0.8538 | 0.8125 |

**Winner**: MEAN pooling (F1: 0.8229 vs 0.8125)
- +0.0104 improvement in F1 score
- Better precision (+0.0534)
- CLS pooling has better recall but lower precision

---

## Summary of Findings

1. **Optimal Multi-Head Configuration**: 4 heads with d_k=32 provides the best balance
2. **Model Choice**: Transformer outperforms LSTM despite longer training time
3. **Sequence Length**: Longer sequences (256) yield significantly better results
4. **Pooling Strategy**: MEAN pooling is more effective than CLS token pooling

### Recommended Configuration
- Model: Transformer
- Heads: 4
- d_k: 32
- Sequence Length: 256 (if memory allows, otherwise 128)
- Pooling: MEAN
- Expected F1: ~0.85-0.86
