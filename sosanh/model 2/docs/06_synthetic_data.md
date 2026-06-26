# 06 — Synthetic dataset cho VN-plate

## 6.1 Vì sao cần synthetic?

Dataset 9752 mẫu là quá nhỏ để train CRNN from-scratch đạt accuracy cao (>95%). Synthetic giúp:

1. **Cân bằng class**: tăng tần suất ký tự hiếm (Q, V, J...).
2. **Đa dạng plate type**: trắng/vàng/xanh/đỏ.
3. **Đa dạng noise pattern**: cover các trường hợp mà real dataset không có.
4. **Robust với edge cases**: plate 4-số (xe máy biển cũ), plate có suffix digit.

## 6.2 Pipeline (đã code ở `src/data/synthetic.py` + `scripts/gen_synthetic.py`)

```
random_plate_text()                         # sinh chuỗi đúng cú pháp VN
   ↓
PIL.draw text với font biển VN              # render ảnh sạch
   ↓
perspective warp                            # mô phỏng camera angle
   ↓
in-plane rotate ±4°                         # mô phỏng nghiêng nhẹ
   ↓
gaussian noise + motion blur                # mô phỏng CCTV
   ↓
brightness/contrast jitter                  # mô phỏng ánh sáng
   ↓
JPEG re-encode quality 35-90                # mô phỏng compression
   ↓
ghi ra <out>/images/syn_*.png + labels.txt
```

## 6.3 Quy tắc sinh text (`random_plate_text`)

```python
prov = random.choice(VN_PROVINCE_CODES)              # 2 digits
series = random.choice("ABCDEFGHKLMNPRSTUVXYZ")      # 1 letter (no W,I,O,Q,J)
if random() < 0.10: series += random.choice(...)     # 2-letter (business)
suffix = random.choice("12345") if random()<0.55 else ""
tail_len = 5 (mostly) or 4 (10% chance)              # 5 chữ số chuẩn
tail = "".join(random digit for _ in range(tail_len))
text = f"{prov}{series}{suffix} {tail}"
```

Tỷ lệ này được calibrate từ phân bố thực trong dataset.

## 6.4 Chọn font

Yêu cầu: TTF sans-serif đậm, gần với font dập biển VN. Một số lựa chọn:

- **EuroPlate / LicensePlate USA** (free): gần đúng style.
- **Bien-so-xe-VN** (Vietnamese license plate font, có thể tìm trên các forum design VN).
- **Roboto Bold / Open Sans Bold**: fallback nếu không có font biển.

> Quan trọng: font phải hỗ trợ hết charset 0-9 + A-Z (không cần W).

Đặt font ở `assets/font.ttf` và gọi:

```bash
python scripts/gen_synthetic.py --font assets/font.ttf --num 50000 --out data/synth
```

## 6.5 Số lượng synthetic khuyến nghị

| Real train | Synthetic        | Mixed ratio (synth) | Kỳ vọng val seq_acc |
| ---------- | ---------------- | ------------------- | --------------------- |
| 9.7K       | 0                | 0                   | ~88-92%               |
| 9.7K       | 20K              | 0.5                 | ~93-95%               |
| 9.7K       | 50K              | 0.7                 | **~95-97%**           |
| 9.7K       | 100K             | 0.7                 | ~96-97% (diminishing)|

## 6.6 Tính hợp lệ với yêu cầu "không pretrained"

Synthetic là **rule-based rendering** dùng:
- Pillow (text render từ font TTF)
- OpenCV (image processing: warp, blur, noise)
- numpy (numeric ops)

**Không có pretrained neural network nào** tham gia. → Hoàn toàn hợp lệ với yêu cầu giảng viên.

Tip để defend: viết trong report rằng synthetic data là *data augmentation thông qua procedural rendering*, không phải pretrained model. Đây là kỹ thuật chuẩn trong SynthText (Gupta et al. 2016), Text90K (Jaderberg et al. 2014) — cite các paper này.

## 6.7 Synthetic + Real mixing chiến lược

Trong `src/data/dataset.py::MixedDataset`:

```python
if random.random() < synthetic_ratio:
    return self.synth[random_idx]
return self.real[random_idx]
```

Mỗi epoch length = max(len(real), len(synth)) để giữ wallclock tương đương.

Khuyến nghị `synthetic_ratio = 0.7` đầu, giảm dần xuống 0.5 ở 50 epoch cuối (curriculum). Để đơn giản, để cố định 0.7 cũng được.

## 6.8 Validation/Test KHÔNG dùng synthetic

Validation và test **chỉ** dùng real data. Synthetic chỉ ở train split.

## 6.9 Demo sinh thử nhanh

```python
from src.data.synthetic import random_plate_text, render_plate
import random, cv2
rng = random.Random(0)
for i in range(5):
    text = random_plate_text(rng)
    img = render_plate(text, "assets/font.ttf", rng)
    cv2.imwrite(f"demo_{i}.png", img)
    print(text)
```
