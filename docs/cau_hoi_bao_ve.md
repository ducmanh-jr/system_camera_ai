# Câu hỏi & Trả lời Bảo vệ Đồ án — VietTraffic AI

> Nguồn: Tổng hợp từ buổi ôn tập thực tế.  
> Tham chiếu mã nguồn chính: `train_ocr_input/kaggle_scratch_ocr/train_best.py`

---

## Q1: Với 1 ảnh đơn lẻ, làm sao biết được ảnh thừa hay thiếu sáng để tăng cường?

**Cách nhận biết** (chuyển ảnh về grayscale rồi đo):

| Trạng thái | Điều kiện |
| :--- | :--- |
| **Thiếu sáng** | Mean < 80 **hoặc** > 55% pixel có giá trị < 40 |
| **Thừa sáng** | Mean > 180 **hoặc** > 55% pixel có giá trị > 220 |
| **Đủ sáng** | Mean nằm trong khoảng 80–180 |

**Cách tăng cường:**
- **Ảnh thiếu sáng:** Dùng **CLAHE** (cân bằng sáng cục bộ thích ứng) hoặc **Gamma Correction (γ < 1.0)** để kéo sáng vùng tối mà không cháy vùng sáng.
- **Ảnh thừa sáng:** Dùng **Gamma Correction (γ > 1.0)** hoặc **Adaptive Thresholding** để bóc tách nét chữ bị lóa.

---

## Q2: Cách xử lý ảnh nghiêng trước khi đưa vào mô hình OCR?

Có 2 loại nghiêng:

**Nghiêng phẳng (Rotation):** Dùng `cv2.minAreaRect` tìm góc nghiêng θ → xoay ngược bằng ma trận Affine.

**Nghiêng phối cảnh (Perspective Skew):** Xác định 4 góc biển số → dùng `cv2.getPerspectiveTransform` ánh xạ về hình chữ nhật chuẩn `48×192`.

**Vị trí code trong dự án:**

| Mục đích | Thư mục | File |
| :--- | :--- | :--- |
| Augmentation khi train (xoay, skew) | `train_ocr_input/kaggle_scratch_ocr/` | `train_best.py` L224–258 |
| Sinh ảnh synthetic bị nghiêng phối cảnh | `huong_cai_tien/train_cai_tien_1/src/data/` | `synthetic.py` L91–104 |
| Nơi tích hợp nắn thẳng khi inference | `train_ocr/src/` | `pipeline.py` L448–475 |

> **Lưu ý:** Mô hình hiện tại **không** chạy bước nắn thẳng chủ động lúc suy luận. Thay vào đó, mô hình được huấn luyện với dữ liệu nghiêng để tự chịu đựng (robust).

---

## Q3: Kết quả trước và sau hậu xử lý (tiền xử lý) thế nào?

| Giai đoạn | Plate Accuracy | Char Accuracy | CER |
| :--- | :---: | :---: | :---: |
| **Thô (Raw)** — chỉ mô hình, chưa qua luật | 51.31% | — | — |
| **Toàn pipeline** — sau bộ giải mã luật + Ensemble | **75.39%** | **93.06%** | **6.96%** |

**Top lỗi nhầm thường gặp (từ `error_report.csv`):**
`4→A` (31 lần), `0→1` (24), `0→6` (18), `T→1` (18), `T→7` (17).

**Vị trí code:**
- Bộ giải mã luật: [`train_best.py:L701-L742`](../train_ocr_input/kaggle_scratch_ocr/train_best.py) — `normalize_plate_prediction()`
- Tính `raw_ok` vs `rule_ok`: [`train_best.py:L845-L865`](../train_ocr_input/kaggle_scratch_ocr/train_best.py) — trong `run_epoch()`

---

## Q4: Mô hình có bị overfitting không? Xảy ra ở giai đoạn nào?

**Có**, bị overfitting rõ ràng:

| Tập | Độ chính xác thô |
| :--- | :---: |
| Train | ~95% |
| Validation | 51.31% |
| **Khoảng cách** | **~43.7%** |

**Giai đoạn xảy ra (theo curriculum):**

```
epoch 1-40:    Học synthetic sạch → Train Acc tăng nhanh
epoch 41-160:  Synthetic khó hơn → khoảng cách bắt đầu xuất hiện
epoch 161-280: Tăng tỉ lệ ảnh thật → OVERFITTING rõ từ ~epoch 160-200
```

**Nguyên nhân:**
- Transformer 8 lớp (~14M tham số) quá lớn cho ~9.700 ảnh thật.
- `dropout=0.12` chưa đủ mạnh để kìm hãm.
- Phụ thuộc `synthetic_ratio=4` → phân phối lệch so với ảnh thật.

**Cơ chế chống overfitting đã có:**  
Early Stopping (`patience=30`) + EMA (`decay=0.999`) + Weight Decay (`0.05`).

---

## Q5: Tổng số tham số là bao nhiêu? Dùng metric gì để đánh giá?

### Tổng tham số (~14M)

| Khối | Tham số |
| :--- | :---: |
| CNN Stem (3→64→128→256) | ~600K |
| Local Mixing Blocks (×2) | ~530K |
| Positional Encoding (512×256) | ~131K |
| **Transformer Encoder (8 lớp)** | **~12.6M** |
| CTC Head + Semantic Head | ~108K |
| **Tổng** | **~14M** |

> ~90% tham số nằm ở Transformer → đây là lý do chính dẫn đến overfitting khi data nhỏ.

### Các Metric đánh giá (từ dataclass `Metrics` tại `train_best.py:L776`)

| Metric | Biến | Ý nghĩa |
| :--- | :--- | :--- |
| CTC Loss | `loss` | Loss chính |
| **Rule Exact Match** | `rule_exact` | **Metric chọn best checkpoint** — % biển đúng sau luật |
| Raw Exact Match | `raw_exact` | % biển đúng thô (trước luật) |
| CER | `cer` | Tỷ lệ lỗi ký tự |
| Đ Accuracy | `d_stroke_acc` | Độ chính xác riêng biển có chữ Đ |
| Confusions | `confusions` | Bảng cặp ký tự nhầm → xuất `error_report.csv` |

---

## Q6: Tại sao mô hình vẫn nhận diện tốt dù bị overfitting?

Pipeline được thiết kế **5 lớp bù trừ** độc lập nhau, mỗi lớp giải quyết một nguồn lỗi khác nhau:

---

### Lớp 1 — Constraint Decoder (Bộ giải mã ràng buộc luật)
📍 [`train_best.py:L701-L742`](../train_ocr_input/kaggle_scratch_ocr/train_best.py) — `normalize_plate_prediction()`

**Nguyên lý:** Kể cả khi mô hình đọc sai 1-2 ký tự, bộ giải mã biết rằng biển số Việt Nam có cấu trúc cố định:
```
[Mã tỉnh 2 số] + [Chữ loại xe 1-3 ký tự] + [Khoảng trắng] + [4-5 số cuối]
```
Dựa trên cấu trúc đó, nó sinh ra danh sách ứng viên (`candidates`) bằng cách thay thế ký tự nhầm theo bảng tra (`NUMERIC_CANDIDATES`, `LETTER_CANDIDATES`), sau đó chọn ứng viên có số lần thay đổi ít nhất và khớp đúng định dạng.

**Ví dụ thực tế:**
- Mô hình đọc `59A4 00128` → Decoder nhận biết `4` ở vị trí chữ loại xe → sửa thành `59AA 00128`.
- Mô hình đọc `3OF 78286` → Decoder thấy `O` ở vùng mã tỉnh phải là số → sửa thành `30F 78286`.

**Đây là lớp đóng góp nhiều nhất:** nâng độ chính xác từ **51.31% → ~70%+**.

---

### Lớp 2 — EMA Checkpoint (Trọng số trung bình trượt)
📍 [`train_best.py:L622-L636`](../train_ocr_input/kaggle_scratch_ocr/train_best.py) — `ModelEMA(decay=0.999)`

**Nguyên lý:** Thay vì lưu trọng số của epoch cuối cùng (thường là đỉnh overfitting), EMA duy trì một bản sao trọng số được tính bằng trung bình trượt hàm mũ của **tất cả các epoch trước**:
```
EMA_weight = 0.999 × EMA_weight_cũ + 0.001 × weight_epoch_hiện_tại
```
Kết quả là bản sao EMA có tính **mượt mà** và **ổn định** hơn — các epoch overfit ở cuối chỉ ảnh hưởng rất ít (hệ số `0.001`), trong khi toàn bộ quá trình học ở giữa được ghi nhớ đầy đủ.

---

### Lớp 3 — Chọn Best Checkpoint theo `rule_exact`
📍 [`train_best.py:L1162`](../train_ocr_input/kaggle_scratch_ocr/train_best.py) — `is_best = val_m.rule_exact >= best_score`

**Nguyên lý:** Hệ thống **không lưu checkpoint theo loss thấp nhất** (loss thấp nhất thường là epoch overfit nhất). Thay vào đó, `best_ema.pt` chỉ được cập nhật khi `rule_exact` — độ chính xác toàn biển **sau bộ giải mã luật trên tập Validation** — đạt giá trị cao nhất từ trước đến nay. Điều này tự động chọn epoch mà mô hình tổng quát hóa tốt nhất trên ảnh thật.

---

### Lớp 4 — Ensemble OCR (Bình chọn 2 mô hình)
📍 [`pipeline.py:L496-L548`](../train_ocr/src/pipeline.py) — `PlateOCREnsemble`

**Nguyên lý:** Kết quả cuối cùng là sự hợp nhất (`merge_train2_train1`) giữa:
- **`train2` (BestOCR):** Mạnh ở cấu trúc chuỗi, nhận diện loại biển đặc biệt.
- **`train1` (PaddleOCR):** Mạnh ở nhận diện từng ký tự độc lập nhờ pretrained trên hàng triệu ảnh văn bản.

Khi `train2` bị overfit và đọc sai cụm số, `train1` thường đọc đúng hơn và ngược lại. Hàm `merge` ưu tiên ứng viên có điểm hợp lệ (`plate_validity_score`) cao hơn và có thể bổ sung các số bị thiếu từ mô hình kia (`patch_missing_tail`).

---

### Lớp 5 — Chọn khung hình sắc nét nhất
📍 [`find_license_plate/main.py`](../find_license_plate/main.py) — Laplacian variance ≥ 1800

**Nguyên lý:** Mô hình OCR không nhận ảnh ngẫu nhiên từ video. Thay vào đó, trong suốt quá trình theo dõi một chiếc xe (có thể hàng chục đến hàng trăm khung hình), hệ thống liên tục chấm điểm chất lượng từng ảnh biển số và chỉ giữ lại ảnh **sắc nét nhất** (phương sai Laplacian cao nhất). Ảnh rõ hơn → mô hình vốn đã bị overfit vẫn đọc được chính xác hơn.

---

### Tổng kết

```
Mô hình thô (Raw 51%)
    + Chọn frame sắc nét nhất     → giảm lỗi do ảnh mờ
    + EMA + Best Checkpoint       → giảm lỗi do overfitting epoch cuối
    + Ensemble với PaddleOCR      → bù lỗi ký tự đơn lẻ
    + Constraint Decoder          → sửa lỗi theo cấu trúc biển số VN
= Kết quả cuối 75.39%
```

> **Kết luận:** Mô hình không mạnh về năng lực thô, nhưng từng lớp trong pipeline được thiết kế để giải quyết đúng từng loại lỗi cụ thể, tạo thành một hệ thống phòng thủ nhiều tầng hiệu quả.

