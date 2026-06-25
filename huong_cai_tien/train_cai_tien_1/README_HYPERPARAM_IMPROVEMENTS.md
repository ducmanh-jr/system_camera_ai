# README — Hướng cải tiến hyperparameters (CRNN+CTC)

Tài liệu này ghi lại các thay đổi mình đã đề xuất + áp dụng vào `configs/crnn_base.yaml` để cải thiện hội tụ và giảm lỗi học CTC ở giai đoạn đầu.

---

## 1) Vấn đề quan sát được từ log huấn luyện

Trong log bạn cung cấp, giai đoạn đầu nhiều epoch có:

- `val seq_acc = 0.0000` (kéo dài tới khoảng epoch ~10–11)
- `val CER` cao (xấp xỉ 0.82 → 0.85 → ...)
- Sau đó mới cải thiện rõ rệt (CER giảm mạnh, seq_acc bắt đầu tăng)

→ Điều này thường gặp khi mô hình **chưa warm-up đủ tốt** cho loss CTC / mapping ảnh→chuỗi, hoặc LR schedule/regularization chưa “đúng nhịp”.

Ngoài ra log có warning quan trọng:

- `lr_scheduler.step()` gọi trước `optimizer.step()`

Điểm này ảnh hưởng trực tiếp tới LR schedule ở những bước đầu (không chỉ là hiệu năng).

---

## 2) Các thay đổi đã áp dụng trong `configs/crnn_base.yaml`

### 2.1 Optimizer / Scheduler

- `optim.lr`: **3.0e-3 → 1.5e-3**
- `optim.weight_decay`: **1.0e-4 → 2.0e-4**
- `optim.warmup_epochs`: **3 → 5**
- `optim.grad_clip`: **5.0 → 3.0**

**Mục tiêu:**

- giảm dao động do LR quá cao lúc đầu
- tăng warmup để CTC ổn định hơn ở giai đoạn đầu
- weight decay + grad clipping giúp generalization tốt hơn

### 2.2 Dropout (regularization nhẹ)

- `model.neck.dropout`: **0.2 → 0.25**
- `model.head.dropout`: **0.1 → 0.15**

### 2.3 Entropy regularizer (CTCWithEntropy)

- `loss.ctc.entropy_weight`: **0.01 → 0.02**

---

## 3) Khuyến nghị quan trọng (nên làm tiếp)

### 3.1 Fix thứ tự step scheduler (khuyến nghị cao)

Trong `src/training/trainer.py` có warning:

> Detected call of `lr_scheduler.step()` before `optimizer.step()`

Cần chỉnh logic để:

1. `optimizer.step()` được gọi trước
2. rồi mới `scheduler.step()`

Việc này là “training correctness”, ưu tiên hơn cả việc tinh chỉnh hyperparameter vì nó có thể làm warmup/LR schedule bị lệch.

---

## 4) Cách test nhanh để so sánh hiệu quả

1. Chạy training với `epochs` nhỏ (ví dụ 5–10).
2. So sánh giữa run cũ và run mới:
   - `val seq_acc` ở epoch 0–10
   - `val CER` ở epoch 0–10
   - độ ổn định/dao động của CER (ít nhảy hơn là tốt)
3. Nếu seq_acc vẫn 0 quá lâu:
   - ưu tiên fix scheduler step order trước
   - sau đó mới tinh chỉnh LR/warmup/entropy

---

## 5) Tóm tắt bộ cấu hình thử (run mới)

- lr: **1.5e-3**
- warmup: **5 epochs**
- wd: **2e-4**
- grad_clip: **3.0**
- neck.dropout: **0.25**
- head.dropout: **0.15**
- entropy_weight: **0.02**

---

## 6) Ghi chú về `data.root`

Trong `configs/crnn_base.yaml`, `data.root` đang để đường dẫn tuyệt đối (trên môi trường khác). Khi chạy ở máy bạn nên dùng:

- `python scripts/train.py --data-root <PATH_LOCAL_DATASET>`

---

Nếu bạn gửi thêm log giai đoạn epoch 0–10 của run mới, mình có thể đánh giá xu hướng và đề xuất next hyperparameter set tối ưu hơn.
