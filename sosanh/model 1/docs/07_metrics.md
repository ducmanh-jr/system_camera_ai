# 07 — Evaluation metrics (chi tiết)

## 7.1 Sequence Accuracy (exact match)

$$\text{SeqAcc} = \frac{\#\{i : \hat{y}_i = y_i\}}{N}$$

- Đây là metric **strict nhất**. Plate dài 10 ký tự sai 1 → sai cả.
- Là metric chính publish trong paper ALPR. Mục tiêu thực tế: ≥ 95% trên test.

## 7.2 Character Accuracy

$$\text{CharAcc} = 1 - \frac{\sum_i \text{Lev}(\hat{y}_i, y_i)}{\sum_i |y_i|}$$

- "Gần đúng" — sai 1 ký tự trong plate 10 = CharAcc đó vẫn 90%.
- Hữu ích khi debug: nếu CharAcc cao mà SeqAcc thấp → model gần đúng nhưng luôn lệch 1 ký tự nào đó.

## 7.3 CER (Character Error Rate)

$$\text{CER} = \frac{\sum_i \text{Lev}(\hat{y}_i, y_i)}{\sum_i |y_i|}$$

- Standard metric OCR. Càng thấp càng tốt. Mục tiêu < 1.0%.
- CharAcc = 1 - CER (chỉ đúng khi không có replacement vs insertion/deletion phân biệt; với Levenshtein chuẩn thì đẳng thức luôn đúng theo định nghĩa trên).

## 7.4 WER (Word Error Rate)

$$\text{WER} = \frac{\sum_i \text{Lev}(\hat{y}_i\!.\text{split}(), y_i\!.\text{split}())}{\sum_i |\text{words}(y_i)|}$$

- Cho plate VN, có 2 "từ" sau split (`"59A1"` và `"00128"`). WER → kém thông tin, ít hữu ích cho debug.
- Vẫn report để so sánh với paper khác.

## 7.5 Levenshtein distance — pure Python (nếu không có lib)

```python
def lev(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m+1))
    cur = [0]*(m+1)
    for i in range(1, n+1):
        cur[0] = i
        for j in range(1, m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
        prev, cur = cur, prev
    return prev[m]
```

Khi có `python-Levenshtein` cài, dùng `Levenshtein.distance(a, b)` (C-ext, nhanh hơn ~30x).

## 7.6 Cách compute trong code (`src/utils/metrics.py`)

```python
from src.utils.metrics import compute_metrics
m = compute_metrics(preds=[...], gts=[...])
print(m.seq_acc, m.cer, m.char_acc, m.wer)
```

## 7.7 Đánh giá per-class / per-error-type (advanced)

Build confusion matrix character-level (sau khi align bằng Hungarian hoặc Needleman-Wunsch):

```python
from collections import Counter
confusion = Counter()
for p, g in zip(preds, gts):
    if len(p) == len(g):
        for cp, cg in zip(p, g):
            if cp != cg:
                confusion[(cg, cp)] += 1
print(confusion.most_common(20))
```

Confusion phổ biến: `(0, O), (1, I), (8, B), (5, S), (2, Z)`. Cho thấy cần augment chữ in dataset.

## 7.8 Online evaluation (production)

Khi deploy:
- Track **rolling SeqAcc** trên 1000 frame gần nhất.
- Track **confidence histogram** (max softmax prob): nếu < 0.7 → reject prediction, request manual.
- A/B testing: 2 ckpt cùng chạy parallel, log để so sánh.
