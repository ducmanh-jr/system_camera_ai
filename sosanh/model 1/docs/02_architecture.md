# 02 — Kiến trúc CRNN (train-from-scratch)

## 2.1 Sơ đồ tổng quan

```
(B,1,48,192) Input
   │
   ▼  Conv block 1: [3×3 Conv 64] → GN → ReLU → [3×3 Conv 64] → GN → ReLU → MaxPool 2×2
(B,64,24,96)
   │
   ▼  Conv block 2: [3×3 Conv 128] ×2 → MaxPool 2×2
(B,128,12,48)
   │
   ▼  Conv block 3: [3×3 Conv 256] ×2 → MaxPool 2×1
(B,256,6,48)
   │
   ▼  Conv block 4: [3×3 Conv 512] ×2 → MaxPool 2×1
(B,512,3,48)
   │
   ▼  Conv block 5: [3×1 Conv 512] (no padding H) → GN → ReLU
(B,512,1,48)   ← H=3 collapses to H=1 cleanly with kernel 3, padding 0
   │
   ▼  squeeze H, permute  →  (T=48, B, 512)
   │
   ▼  BiLSTM × 2 (hidden=256, bidir)  →  (T=48, B, 512)
   │
   ▼  Linear (512 → 38)
(T=48, B, 38)  ← logits
```

T = 48 ≫ 11 (max label length) → đủ vùng cho CTC alignment.

## 2.2 Lý do tradeoff

### CRNN vs Transformer OCR

| Yếu tố                            | CRNN (chọn)                  | TrOCR / PARSeq                 |
| --------------------------------- | ---------------------------- | ------------------------------ |
| Data efficiency from scratch      | ★★★★★                       | ★★ (cần >100K real)            |
| Latency (RTX 3060 FP16)           | 3-5 ms                       | 25-80 ms                       |
| Robust với chuỗi ngắn 9-11        | Tối ưu                       | Overkill                       |
| Khả năng tuning                   | Đơn giản                     | Nhiều hyper-params             |

### Backbone: VGG-lite vs ResNet vs MobileNet

- **VGG-lite** (chọn): không có skip → gradient flow đơn giản hơn ở giai đoạn đầu khi train from scratch. Một vài paper OCR (CRNN gốc Shi et al., 2015) cũng dùng VGG.
- ResNet: skip connections tốt cho deep nets nhưng với 5 conv blocks ta không cần. Dễ overfit vì capacity dư.
- MobileNetV3 / EfficientNet: được thiết kế cho ImageNet, **không phù hợp from-scratch** trên dataset OCR nhỏ vì cần BN với batch lớn + cần pretrained để hội tụ tốt.

### Normalization: GroupNorm vs BatchNorm

- **GroupNorm** với 8 groups (chọn): không phụ thuộc batch size → ổn định trên Colab/Kaggle 1 GPU (batch 32-64).
- BN: chỉ tốt khi batch ≥ 64 và data IID. Khi train from scratch + DDP nhỏ thì BN thường wobble loss.

### Pooling pattern

Tại sao block 3-4-5 không pool theo W?
- Pool W = 4 lần thì T = 192/4 = 48. Plate dài tối đa 11 ký tự, CTC cần T ≥ 2·L+1 = 23 (theo bound chặt với blank chèn giữa) → T=48 thoải mái.
- Nếu pool W = 16 lần → T = 12, vừa sát ranh giới, dễ collide.

## 2.3 Số liệu cụ thể (config base)

| Layer       | Output                | Params    |
| ----------- | --------------------- | --------- |
| Conv1 (1→64)  | (B, 64, 48, 192)     | 36,864    |
| Conv2 (64→128) | (B, 128, 24, 96)    | 147,456   |
| Conv3 (128→256) | (B, 256, 12, 48)   | 589,824   |
| Conv4 (256→512) | (B, 512, 6, 48)    | 2,359,296 |
| Conv5 (512→512) 3×1 | (B, 512, 1, 48)| 786,432   |
| BiLSTM × 2 (512↔256) | (T, B, 512)     | ~3,150,000 |
| FC (512→38) |                       | 19,494    |
| **Total**   |                       | **~8.4M** |

Config `crnn_small` (channels [32,64,128,256,256], LSTM hidden 128) cho ~0.9M params, mất accuracy ~1-2% nhưng nhanh gấp 2-3x.

## 2.4 Khi nào dùng Transformer neck thay BiLSTM?

Đã code sẵn ở `src/models/neck.py::TransformerNeck`. Bật khi:
- Đã sinh được >100K mẫu synthetic và muốn ép accuracy thêm.
- Có GPU ≥ 12 GB.
- Chấp nhận latency tăng 2-3x.

Cấu hình thử: `num_layers=4, nhead=8, hidden=256` — ~3M params, val seq_acc tăng ~0.5-1% trên dataset ≥100K.

## 2.5 Optional: STN (Spatial Transformer Network)

Có thể thêm STN trước backbone để rectify perspective. Khi train từ đầu trên dataset 10K, STN khó hội tụ và thường gây bất ổn. Khuyên không dùng giai đoạn này; nếu cần, dùng `rectifier` truyền thống (homography) ở preprocessing.
