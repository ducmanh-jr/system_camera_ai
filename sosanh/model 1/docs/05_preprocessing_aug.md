# 05 — Preprocessing & Augmentation deep-dive

## 5.1 Preprocessing pipeline (deterministic, dùng cho cả train & infer)

1. **BGR → Grayscale**: bỏ thông tin màu (biển VN nhiều màu nền), giảm bias.
2. **Resize giữ aspect → pad bên phải về (48, 192)**: tuyệt đối không squash, sẽ méo glyph (chữ A bị bóp thành 0).
3. **CLAHE** (`clipLimit=2.0, tileGridSize=(8,8)`): chuẩn hoá độ tương phản theo vùng — quan trọng khi ảnh có chỗ chói chỗ tối.
4. **Tensor [0,1] → Normalize ((x-0.5)/0.5)**: range [-1, 1] phù hợp với init Kaiming.

### Tại sao không Otsu / adaptive threshold?

Threshold xoá thông tin stroke (đặc biệt phần serif và edge). Model học stroke pattern tốt hơn với gray-level input → giữ nguyên.

Trường hợp riêng: nếu camera CCTV chất lượng cực thấp + plate luôn rõ → có thể bật adaptive threshold ở `transforms.py` (uncomment).

### Tại sao không denoise (bilateral, NLM) thường trực?

Denoise mạnh có thể smooth chữ thành blob → CER tăng. Chỉ bật khi noise > 30 sigma (hiếm).

## 5.2 Augmentation rationale

Dataset 9752 train là **nhỏ**. Augmentation phải mạnh nhưng có lý.

### Augmentation tier (theo độ quan trọng)

**TIER 1 — bắt buộc:**

| Aug                  | Hiệu quả                                |
| -------------------- | ---------------------------------------- |
| `ImageCompression`   | +1.5% val acc (CCTV robustness)         |
| `MotionBlur`         | +1.0% val acc                            |
| `Perspective`        | +0.8% val acc                            |
| `Rotate ±5°`         | +0.5% val acc                            |
| `RandomBrightnessContrast` | +0.4% val acc                      |

**TIER 2 — nên có:**

| Aug                  | Hiệu quả                                |
| -------------------- | ---------------------------------------- |
| `GaussNoise`         | +0.3%                                    |
| `Affine shear`       | +0.3%                                    |
| `CoarseDropout`      | +0.2-0.5% (chỉ khi có nhiều dirty plate) |
| `RandomShadow`       | +0.2%                                    |

**TIER 3 — không nên dùng:**

| Aug                  | Lý do bỏ                                |
| -------------------- | ---------------------------------------- |
| HorizontalFlip       | Đảo ký tự                                |
| VerticalFlip         | Đảo ngược                                |
| RandomCrop > 10%     | Mất ký tự đầu/cuối                       |
| ElasticTransform     | Méo glyph không thực tế                  |
| MixUp / CutMix       | Phá CTC alignment                        |
| GridDistortion       | Phá tỷ lệ chữ                            |

## 5.3 Implementation note

Augmentation đặt **trước** preprocessing trong Dataset (vì augment cần BGR uint8). Preprocessing được áp dụng **sau cùng** để chuẩn hoá shape & range. Xem `src/data/dataset.py::PlateDataset.__getitem__`.

```python
img = cv2.imread(...)      # BGR uint8
if self.augment:
    img = self.augment(image=img)["image"]   # vẫn BGR uint8
tensor = self.preprocessor(img)              # → (1, 48, 192) float [-1,1]
```

## 5.4 Hyper-parameter aug (tối ưu cho VN plate)

```python
A.Compose([
    A.Rotate(limit=5, p=0.5),                       # nhẹ, plate ít nghiêng
    A.Affine(shear={'x': (-5,5)}, scale=(0.95,1.05), p=0.5),
    A.Perspective(scale=(0.02, 0.06), p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=(3,7)),
        A.GaussianBlur(blur_limit=(3,5)),
        A.Defocus(radius=(1,3)),
    ], p=0.4),
    A.OneOf([
        A.GaussNoise(var_limit=(10,50)),
        A.ISONoise(),
    ], p=0.3),
    A.ImageCompression(quality_lower=40, quality_upper=85, p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    A.RandomShadow(p=0.2),
    A.CoarseDropout(max_holes=3, max_height=10, max_width=20, p=0.2),
])
```

## 5.5 Test-time augmentation (TTA) — boost ở evaluate

Có thể (optional) áp dụng 3-5 augmentation nhẹ ở infer time và vote majority. +0.3-0.6% seq_acc. Trade-off: latency 3-5x. Chỉ dùng khi cần độ chính xác tối đa offline.

```python
preds = []
for _ in range(5):
    img_aug = light_tta(image=img)['image']
    preds.append(predict(img_aug))
final = majority_vote(preds)
```

## 5.6 Per-character distribution check (quan trọng!)

Trước khi train, chạy script kiểm tra: ký tự nào hiếm < 50 mẫu? → cần augment thêm hoặc tăng synthetic tỉ lệ.

```python
from collections import Counter
counts = Counter()
for _, text in pairs:
    counts.update(text)
for c, n in counts.most_common():
    print(c, n)
```

Trong dataset VN plate: `Q, V, I, J, U` thường < 100 mẫu → synthetic generator có thể tăng tỉ lệ chữ này.
