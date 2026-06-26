# 03 — Training guide (from scratch)

## 3.1 Vì sao from-scratch khó hơn fine-tune

Khi fine-tune từ pretrained:
- Backbone đã có **edge / texture / corner** detectors sẵn (do train trên ImageNet hoặc OCR dataset lớn).
- Loss landscape gần một minimum tốt, bạn chỉ "leo dốc nhỏ".

From-scratch:
- Toàn bộ filters convolution là noise. Phải tự học edge detector từ chữ trên plate.
- Loss CTC là sum-of-exponential-many alignment → ban đầu rất bumpy.
- Dataset 10K mẫu là **nhỏ** so với 100K-1M điển hình của OCR. Dễ overfit nếu không augment đủ.

## 3.2 Hyper-parameter recipe (đã test)

```yaml
optimizer: AdamW
lr_max: 3e-3
weight_decay: 1e-4
scheduler: OneCycleLR (pct_start = warmup_epochs / total_epochs)
warmup_epochs: 3
batch_size: 64 (≥ 8GB GPU) hoặc 32 (4GB)
epochs: 200
grad_clip: 5.0
amp: true
ema_decay: 0.999
augmentation: heavy (xem docs/05)
```

Lý do từng số:
- **lr_max=3e-3**: cao hơn fine-tune (thường 1e-4) vì backbone init random, cần lr cao để thoát noise.
- **OneCycleLR**: hot-start ramp + cosine anneal → mượt, ít cần tune.
- **weight_decay=1e-4**: nhỏ. Nếu set 5e-4 dễ underfit ở plate.
- **AMP**: bắt buộc nếu GPU < 12 GB, không thì lãng phí.
- **EMA 0.999**: smooth val curve, thường +0.3-1% accuracy.

## 3.3 Cách stabilize training

| Triệu chứng                  | Khắc phục                                           |
| ---------------------------- | --------------------------------------------------- |
| Loss = NaN sau vài batch     | Hạ lr_max xuống 1e-3; `zero_infinity=True`; `grad_clip=5`; AMP scaler init 2^14 |
| Val tốt nhưng train tăng     | Dataset overfit → tăng dropout, augment mạnh hơn    |
| Train tốt nhưng val xấu      | Dataset bias → augment mạnh, thêm synthetic         |
| LR scheduler thiếu warmup    | Bật `warmup_epochs=3` → OneCycleLR sẽ ramp lr      |
| EMA bị "trễ"                 | EMA luôn lag — đánh giá best CER bằng EMA model, log cả 2 |

## 3.4 Convergence speed

Trick để model converge nhanh hơn:
1. **Init đúng**: Kaiming for Conv, orthogonal for LSTM weights, forget-gate bias = 1. Đã có trong `src/models/init.py`.
2. **Mixed precision**: tăng throughput 1.8-2.2x trên Turing+.
3. **Larger effective batch via grad accumulation**: nếu GPU bé, accumulate 2-4 steps để có effective batch ~128.
4. **Don't use cudnn benchmark off**: nên `torch.backends.cudnn.benchmark = True` khi shape cố định.

## 3.5 Khi dataset nhỏ (10K) — checklist

- [x] Bật toàn bộ augmentation
- [x] Generate ≥ 50K synthetic plates (`scripts/gen_synthetic.py`)
- [x] Train với mixed dataset (`synthetic_ratio=0.6-0.8`)
- [x] Bật EMA
- [x] Early stopping `patience=25`
- [x] SWA optional 20 epoch cuối
- [ ] **KHÔNG** dùng MixUp / CutMix với label CTC (sai label distribution)
- [ ] **KHÔNG** dùng horizontal flip

## 3.6 Cách chọn batch size & learning rate

Rule of thumb:
- Batch 32 → lr_max 2e-3
- Batch 64 → lr_max 3e-3
- Batch 128 → lr_max 4e-3 đến 5e-3
- Batch 256 → lr_max ~6e-3 (linear scaling rule)

Test bằng "LR finder" (Smith 2018): chạy 1 epoch tăng lr từ 1e-6 → 1, plot loss → lr_max là điểm trước khi loss bùng nổ.

## 3.7 Resume / Checkpoint strategy

Trainer auto-save:
- `best.pt`: theo val CER thấp nhất
- `last.pt`: epoch gần nhất (để resume khi mất điện)
- `epoch_xxx.pt`: mỗi `ckpt_every_epoch=5`

Resume: `--resume runs/.../last.pt` sẽ tiếp tục cả optimizer + scheduler + scaler state.

## 3.8 Debugging checklist khi loss không giảm

1. Forward pass dummy `torch.zeros(2,1,48,192)` không lỗi shape? → smoke test trong `scripts/`.
2. CTC `input_lengths >= target_lengths`? Đặt `T=logits.size(0)` và `target_lengths` đúng.
3. Vocab có trùng `<blank>` với ký tự thật không? → blank index = 0, không nằm trong charset.
4. Augmentation có làm trắng ảnh không? → tắt `CoarseDropout` rồi bật lại từ từ.
5. AMP underflow? → lúc đầu set `amp=False` để loại trừ.

## 3.9 Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/crnn_base.yaml --ddp
```

Lưu ý:
- Batch size trong config là **per-GPU**, hiệu quả total = bs × num_gpus → có thể tăng lr_max theo linear scaling rule.
- `DistributedSampler` đã tự shuffle epoch-dependent. Đừng quên `sampler.set_epoch(epoch)` (đã có trong trainer).
