# 01 — Phân tích bài toán OCR biển số xe Việt Nam

## 1.1 Bối cảnh ALPR ở Việt Nam

ALPR (Automatic License Plate Recognition) ở Việt Nam là bài toán phức tạp vì:

1. **Đa dạng định dạng plate**: Ô tô 1 dòng, xe máy 2 dòng, biển vàng (kinh doanh), biển xanh (nhà nước), biển đỏ (quân đội), biển ngoại giao (NG).
2. **Camera nguồn đa dạng**: CCTV bitrate thấp, smartphone, dashcam, camera giao thông tốc độ cao.
3. **Điều kiện môi trường khắc nghiệt**: Mưa to, sương mù, đèn pha chói, bùn đất bám.

Bài toán của bạn ở giai đoạn này là **OCR thuần** — input đã là ảnh crop plate, output là chuỗi ký tự. Phần detect (YOLO / SSD) tách rời.

## 1.2 Charset (từ dataset thực)

```
0 1 2 3 4 5 6 7 8 9
A B C D E F G H I J K L M N O P Q R S T U V X Y Z
Đ           ← (U+0110) biển quân đội / công an đặc biệt, ví dụ "60MĐ2"
(space)
```

Tổng cộng **38 ký tự + 1 blank = 39 classes**. Không có `W` (đúng quy chuẩn VN).
Phát hiện trong dataset: 305 / 9752 mẫu train có chữ `Đ`, các series `MĐ`, `TĐ` thuộc biển đặc biệt — phải đưa vào charset, không được bỏ.

Ký tự hiếm trong dataset (số mẫu): `O=1, I=2, J=3, Y=42, R=44, U=47, Z=49, X=81, S=124, V=149`. Với những ký tự < 100 mẫu nên **resample** khi train (xem `WeightedRandomSampler`) hoặc dựa vào synthetic.

## 1.3 Cấu trúc biển số

Phổ biến:

```
<prov>(2d) <series>(1-2 chữ cái)<digit?>  <số đăng ký>(4-5d)
```

Ví dụ chi tiết:

| Plate         | Province (2d) | Series | Suffix | Number    | Type             |
|---------------|---------------|--------|--------|-----------|------------------|
| `59A1 00128`  | 59 (HCM)      | A      | 1      | 00128     | Ô tô              |
| `68HC 00042`  | 68 (Kiên Giang) | HC   | -      | 00042     | Doanh nghiệp     |
| `29Z 5270`    | 29 (Hà Nội)   | Z      | -      | 5270 (4d) | Xe máy biển cũ   |
| `30E 99077`   | 30 (Hà Nội)   | E      | -      | 99077     | Xe máy           |

## 1.4 Khó khăn cụ thể (mapping → augmentation)

| Khó khăn                  | Mô tả                                      | Aug tương ứng                        |
| ------------------------- | ------------------------------------------ | ------------------------------------- |
| Motion blur               | xe chạy 30-60 km/h, shutter chậm           | `MotionBlur(blur_limit=(3,7))`        |
| Low light                 | đêm, hầm để xe, ngược sáng                 | `RandomBrightnessContrast`, CLAHE     |
| Skew                      | camera lệch trục đường                     | `Rotate(limit=5)`                     |
| Perspective distortion    | camera trên cao, plate xa                  | `Perspective(scale=(0.02,0.06))`     |
| Dirty plate               | bùn, dán băng keo                          | `CoarseDropout`, occlusion patches    |
| Compression artifacts     | CCTV bitrate thấp, NVR re-encode           | `ImageCompression(40-85)`             |
| Plate 2 dòng (xe máy)     | nội dung chia làm 2 hàng                   | Cần preprocess tách dòng riêng        |

## 1.5 Quyết định kỹ thuật rút ra

- **Input size**: 48 × 192 (giữ aspect ratio plate ~3:1, có buffer cho biển 2 dòng được "stack" lại bằng preprocess).
- **Grayscale**: bỏ thông tin màu (đỡ overfit theo loại biển), tăng tốc backbone, tiết kiệm bộ nhớ.
- **Char-level CTC**: không dùng word-level vì plate VN không có "từ" rõ ràng.
- **Confidence calibration**: trainer hỗ trợ entropy regularizer → posterior nhọn, dùng với threshold để filter low-confidence prediction trong production.
