# 04 — Loss functions (chi tiết toán học)

## 4.1 CTC loss — định nghĩa hình thức

Cho:

- $x = (x_1, \dots, x_T)$ : sequence of feature vectors (output của CNN+RNN, length T)
- $y = (y_1, \dots, y_L)$ : target sequence (transcription, length L)
- Alphabet mở rộng: $\mathcal{C}' = \mathcal{C} \cup \{\text{blank}\}$
- $p(c \mid x_t)$ : posterior over $\mathcal{C}'$ tại time $t$ (output của softmax)

Một **alignment** là $\pi = (\pi_1, \dots, \pi_T) \in (\mathcal{C}')^T$.

Toán tử **collapse** $\mathcal{B}$: bỏ repeats liên tiếp rồi bỏ blank.

CTC loss:

$$\boxed{\;\;\mathcal{L}_{CTC}(x, y) = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p(\pi_t \mid x_t)\;\;}$$

Tính toán bằng **forward-backward DP** $O(T \cdot |y'|)$ với $y'$ là $y$ chèn blank xen kẽ. PyTorch `nn.CTCLoss` đã làm việc này trong CUDA.

## 4.2 Implementation chú ý

```python
log_probs = F.log_softmax(logits, dim=-1)   # (T, B, C)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

- `input_lengths`: số timestep "thật" của mỗi sample. Vì ta dùng input fixed size → input_lengths = T (constant).
- `target_lengths`: độ dài label gốc của mỗi sample (chính là `len(text)`).
- `targets`: 1D concatenation của tất cả label, hoặc 2D padded với pad index bất kỳ.
- **`zero_infinity=True`**: bắt buộc khi train from scratch — đôi khi alignment khả thi không có → loss = ∞ → grad = NaN; option này sẽ zero-out gradient cho sample đó.
- Tính loss bằng float32 (tránh underflow ở exp/log). PyTorch sẽ tự cast nếu input là half, nhưng cẩn thận khi dùng `autocast`.

## 4.3 Entropy regularizer (đã code trong `src/losses/ctc.py`)

Mục đích: làm posterior nhọn hơn → greedy decode chính xác hơn, và giúp model tránh "blank-only" trivial solution.

$$\mathcal{L}_{ent} = \frac{1}{T \cdot B} \sum_{t,b} \big( - \sum_c p_{tbc} \log p_{tbc} \big)$$

Total: $\mathcal{L} = \mathcal{L}_{CTC} + \lambda \cdot \mathcal{L}_{ent}$ với $\lambda = 0.01$ (mặc định).

Lưu ý: entropy regularizer giảm entropy → posterior nhọn. Code đặt dấu cộng $+\lambda \mathcal{L}_{ent}$ với $\mathcal{L}_{ent}$ chính là **entropy** (≥0). Thực tế *minimizing* entropy → nhọn. Một số tác giả ký hiệu dấu khác — kiểm tra trước khi đổi $\lambda$.

## 4.4 Label smoothing & CTC

CTC không tương thích trực tiếp với label smoothing (vì label là *target sequence*, không phải *one-hot per timestep*). Cách phổ biến:

1. **CTC + auxiliary CE branch**: thêm head dự đoán length, hoặc dự đoán ký tự per-timestep với label smoothing. Phức tạp, ít lợi cho plate ngắn.
2. **Focal CTC**: $\mathcal{L}_{focal} = (1 - p_y)^\gamma \mathcal{L}_{CTC}$, $p_y = e^{-\mathcal{L}_{CTC}}$. Có cải thiện ~0.2-0.5% nhưng cần tune $\gamma$.
3. **KHÔNG cần** nếu đã có đủ entropy regularizer + augmentation tốt.

Kết luận thực dụng cho bài này: giữ CTC + entropy reg là đủ.

## 4.5 Regularization khác

- **Weight decay**: AdamW với `weight_decay=1e-4`. Không dùng weight decay trên BiasLayer / Norm layers (PyTorch AdamW không tự bỏ — bạn có thể split param groups nếu muốn).
- **Dropout**: 0.2 sau BiLSTM, 0.1 trước FC head.
- **Grad clip**: max-norm 5.0 — quan trọng với LSTM.

## 4.6 Hỏi-đáp thường gặp

**Q: CTC loss của tôi dao động 1e2 → 1e0 → 1e1, có bình thường không?**
A: Bình thường ở 5-10 epoch đầu. Sau đó nên giảm đều. Nếu dao động mạnh quá → lr quá cao.

**Q: Tại sao greedy decode ra `1A1 11111` khi gt là `59A1 00128`?**
A: Model chưa converge. Kiểm tra epoch (cần ≥ 20-30 trước khi xem prediction "đẹp").
