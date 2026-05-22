# Vehicle License Plate Recognition (VN) — Pipeline YOLO + OpenCV + EasyOCR

> Repo này chứa một notebook triển khai nhận dạng **biển số xe Việt Nam** theo mô hình lai:
> 1) **YOLO** phát hiện xe 
> 2) **Heuristic + OpenCV** crop và tinh chỉnh vùng biển số 
> 3) **Xử lý ảnh** (resize/denoise/CLAHE/threshold/morphology/deskew/erosion/segmentation) 
> 4) **OCR ký tự** bằng **EasyOCR** (kèm augment và rule nhận dấu chấm `.`) 
> 5) **Rule-check biển số VN** bằng regex theo chuẩn form và map tỉnh/thành.

---

## 1. Mục tiêu

- Nhận vào 1 ảnh đầu vào (ví dụ `image.jpg`).
- Tự động:
  - Phát hiện **phương tiện** trên ảnh.
  - Xác định **vùng chứa biển số** và crop ra ảnh biển.
  - Chuẩn hoá ảnh biển để OCR chính xác hơn.
  - OCR để đọc biển theo định dạng chuỗi.
  - Kiểm tra tính hợp lệ theo **chuẩn biển số Việt Nam**.
- Xuất ra:
  - Biển số nhận được (chuỗi cuối)
  - Kết quả hợp lệ/không hợp lệ
  - Ảnh tóm tắt (notebook lưu `ket_qua_bien_so.png`)

---

## 2. Kiến trúc tổng quan (pipeline end-to-end)

Quy trình chạy theo luồng sau:

1. **Detect xe bằng YOLO** (model `yolo11n.pt`)
2. Từ các bbox xe thu được, **chọn bbox gần tâm ảnh nhất**
3. **Crop vùng dưới của xe** (ROI) để giảm nhiễu, tập trung tìm biển
4. Dùng OpenCV **threshold + contours** để ước lượng bbox biển (`lp_rect`)
5. Crop biển theo `lp_rect`
6. **Zoom + chuẩn hoá kích thước** (đưa chiều cao biển về xấp xỉ 256px)
7. Chuyển màu → xám, **denoise** (NLM), tăng tương phản (**CLAHE**)
8. **Threshold** (Otsu) → nhị phân
9. **Morphology** (Open/Close/Dilate) + **làm sạch viền rìa**
10. **Deskew** (ước lượng góc nghiêng bằng HoughLines rồi rotate)
11. **Padding trắng** và **Erosion** để tách ký tự
12. **Segmentation ký tự** (chia dòng 2 phần + tách theo cột)
13. **OCR từng ký tự** bằng EasyOCR:
    - Tạo nhiều biến thể ảnh (augment)
    - Lấy top candidate theo confidence
    - Rule đặc biệt cho **dấu chấm `.`**
14. **Ghép 2 dòng** → chuỗi `plate_final`
15. **Rule-check biển số VN** bằng regex + map tỉnh/thành
16. Vẽ/hiển thị kết quả cuối

---

## 3. Chi tiết từng bước (bám theo notebook)

### Cell 1 — YOLO detect xe

- Khởi tạo model:
  - `model = YOLO('yolo11n.pt')`
- Đọc ảnh bằng OpenCV (`cv2.imread`).
- Chạy inference:
  - `results = model(image, conf=0.4)[0]`
- Lấy bbox ở dạng `xyxy` và vẽ khung + label.

**Điểm quan trọng:**
- Với các bbox xe detect được, notebook chọn **best_box** là bbox có **tâm gần tâm ảnh nhất**.
  - Tính khoảng cách Euclid giữa `(cx_box, cy_box)` và `(cx_img, cy_img)`.
  - Chọn bbox có `dist` nhỏ nhất.

Lý do chọn gần tâm:
- Thông thường biển số nằm trên xe chính ở giữa khung hình; chọn gần tâm giúp tránh nhiễu khi ảnh có nhiều đối tượng.

---

### Cell 2 — Tạo mask “vùng xe” (bọc nền trắng)

- Tạo ảnh trắng `result_image`.
- Dựng mask bằng cách tô vùng bbox xe bằng 255.
- Chỉ giữ pixel trong bbox xe, ngoài bbox trở thành trắng.

Mục đích:
- Hỗ trợ trực quan và/hoặc nền tách biệt để các bước xử lý kế tiếp dễ ổn định hơn.

---

### Cell 3 — Giữ lại đúng bbox xe mục tiêu

- Lấy bbox `best_box`.
- Tạo `result_image` chỉ chứa vùng xe.

---

### Cell 4 — Tìm biển số bằng heuristic (không dùng YOLO cho biển)

**Pipeline heuristic:**

1. Crop xe theo bbox xe: `car_crop = image[y1:y2, x1:x2]`
2. Chỉ lấy phần ROI phía dưới xe:
   - `roi = car_crop[int(crop_h * 0.55):, :]`
   - Lý do: biển số thường nằm ở nửa dưới của xe.
3. Convert sang grayscale:
   - `gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)`
4. Threshold nhị phân đơn giản:
   - `_, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)`
5. Tìm contours trên ảnh nhị phân:
   - `contours, _ = cv2.findContours(...)`
6. Với mỗi contour, lấy bounding rect và lọc theo điều kiện dạng biển:
   - Aspect ratio: `1.5 < aspect_ratio < 5.0`
   - Area đủ lớn: `area > 800`
   - Chiều rộng hợp lý: `w < crop_w * 0.6`
   - Chiều cao tối thiểu: `h > 15`
7. Chọn ứng viên có `score = area` lớn nhất.
8. `lp_rect` được lưu theo hệ toạ độ ROI, sau đó đổi về toạ độ ảnh gốc.
9. Crop `lp_crop = image[abs_y1:abs_y1+lh, abs_x1:abs_x1+lw]`
10. Lưu `license_plate_crop.jpg`.

---

### Cell 5 — “Tô màu” để minh hoạ vùng xe vs vùng biển

- Tạo nền trắng.
- Vùng xe tô kem nhạt.
- Vùng biển giữ nguyên pixel gốc.

Chỉ phục vụ trực quan.

---

### Cell 6 — Zoom biển số

- Resize `lp_crop` theo `scale=4` (dùng `INTER_CUBIC`).
- Tạo canvas nền kem để biển nằm giữa.

Mục đích:
- Tăng chi tiết ký tự trước khi đưa qua denoise/threshold/OCR.

---

### Cell 7 — Resize về chuẩn 256px + sharpen

- Chọn target height `target_h = 256`.
- Tính tỉ lệ `scale_factor = target_h / lp_crop.shape[0]`.
- Resize về `(target_w, target_h)` bằng `INTER_LANCZOS4`.
- Áp kernel sharpen (Laplacian-like):
  - `[[0,-1,0],[-1,5,-1],[0,-1,0]]`

---

### Cell 8 — Chuyển grayscale

- `lp_gray = cv2.cvtColor(lp_resized, cv2.COLOR_RGB2GRAY)`

---

### Cell 9 — Denoise (so sánh + chọn NLM)

Trong notebook có phần so sánh nhiều biến thể, nổi bật:
- NLM: `cv2.fastNlMeansDenoising(lp_gray, h=10, ...)`
- Bilateral filter cũng được thử để đối chiếu.

Cuối cùng thực tế dùng:
- `lp_denoised = cv2.fastNlMeansDenoising(lp_gray, h=10, ...)`

---

### Cell 10 — CLAHE để tăng tương phản cục bộ

- Dùng CLAHE để xử lý vùng tối/sáng không đồng đều:
  - `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))`
  - `lp_clahe = clahe.apply(lp_denoised)`

---

### Cell 11 — Threshold

- So sánh Otsu vs Adaptive threshold.
- Dùng Otsu trong bước chính:
  - `_, lp_thresh = cv2.threshold(lp_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`

---

### Cell 12 — Morphology để làm sạch nhiễu

- Có phần so sánh Open/Close/Dilate.
- Bản dùng:
  - `lp_morph = cv2.morphologyEx(lp_thresh, cv2.MORPH_OPEN, kernel)`
  - `kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))`

---

### Cell 13 — Làm sạch viền rìa (loại nhiễu quanh mép)

Notebook triển khai nhiều phương pháp so sánh, cuối cùng chọn hướng:
- **Crop pad** (cắt lề một dải quanh rìa)
- Sau đó **pad trắng trở lại** để giữ kích thước hợp lệ.

Cách thực tế:
- `pad = 10`
- `lp_clean = lp_morph[pad:-pad, pad:-pad]`
- `cv2.copyMakeBorder(... value=255)`

---

### Cell 14 — Deskew (chỉnh nghiêng)

- Ước lượng góc bằng:
  1. `Canny` tạo edges
  2. `cv2.HoughLines` để tìm các đường thẳng
  3. Đổi `theta` → angle và lấy median các angle trong [-45°, 45°]
- Rotate ảnh về thẳng với:
  - `lp_deskew = rotate(lp_clean, angle, reshape=False, cval=255)`

Trong notebook, phần deskew/rotate phục vụ minh hoạ và chuẩn hoá hình học.

---

### Cell 14b — Padding trắng

- `pad = 20`
- `lp_padded = cv2.copyMakeBorder(lp_clean, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)`

---

### Cell 15 — Erosion (mỏng nét) để tách ký tự

- Dùng erosion với kernel (2x2) iterations=1:
  - `lp_erode = cv2.erode(lp_padded, kernel, iterations=1)`

Mục tiêu:
- Làm nét ký tự “tách chữ rõ hơn”, hỗ trợ segmentation theo cột.

---

### Cell 15b — Segment ký tự (tách từng ký tự)

Notebook dùng cách segmentation theo tổng cột trong ảnh đảo (inverse).

Luồng segmentation:
1. Tách biển thành 2 dòng:
   - `line1 = lp_erode[:h//2, :]`
   - `line2 = lp_erode[h//2:, :]`
2. Với mỗi dòng, gọi `segment_chars(line_img)`:
   - `inv = bitwise_not(line_img)`
   - `col_sum = sum(inv, axis=0)`
   - Chọn ngưỡng `thr = col_sum.max() * threshold_ratio`
   - Duyệt cột để tìm đoạn có col_sum > thr (tức có nét ký tự)
   - Merge các đoạn gần nhau (`merge_gap`)
   - Lọc theo width tối thiểu / tương đối median
3. Với mỗi đoạn `(x1,x2)` lấy `char_img` bằng cách crop theo padding ngang (p=6)

Kết quả:
- Tạo danh sách ảnh ký tự để đưa vào OCR.

---

### Cell 16 — OCR từng ký tự bằng EasyOCR + augment

**Khởi tạo:**
- `reader = easyocr.Reader(['en'], gpu=False)`
- `allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.'`

#### 16.1 Hàm tạo biến thể ảnh (augmentation variants)

`make_variants(img)` trả về nhiều ảnh:
- Pad + inverse
- Dilate/erode (2x2)
- Dilate/erode trên ảnh inverse
- Gaussian blur
- CLAHE trên pad và inverse
- Resize về 64x64

Mục đích:
- EasyOCR có thể nhạy với kiểu nét/contrast; augment giúp tăng cơ hội đọc đúng.

#### 16.2 Rule nhận dấu chấm `.`

Vì dấu chấm nhỏ và dễ bị OCR nhầm:
- Hàm `is_dot(char_img, all_segs_widths)` xác định nếu width nhỏ hơn ~25% median width và aspect ratio phù hợp.
- Nếu là dấu chấm → trả thẳng `('.', 1.0)`.

#### 16.3 OCR ký tự theo best confidence

`ocr_char(char_img)`:
- Với mỗi variant:
  - `reader.readtext(variant, detail=1, allowlist=..., text_threshold=0.3, ...)`
- Tính `scores[ch] = max(conf)` theo từng chữ xuất hiện.
- Trả về dict {ký tự: confidence}.

#### 16.4 OCR theo từng dòng rồi ghép chuỗi

`ocr_line(line_img, line_name)`:
- Segment ra `segs`.
- Với mỗi segment:
  - Nếu là dot → `.`
  - Ngược lại → gọi `ocr_char` và lấy candidate có conf cao nhất.
- Ghép chuỗi theo thứ tự segment.

Cuối cùng:
- `plate_final = f"{text1}-{text2}"`

---

### Cell 17 — Visualize top candidates

- In bảng top-3 candidate cho từng ký tự.
- Vẽ ảnh ký tự đã segment.

---

### Cell 18 — Rule-check biển số Việt Nam

Mục tiêu của bước này:
- Chống OCR sai bằng cách xác thực theo chuẩn thực tế.

#### 18.1 Map tỉnh/thành

- `TINH_THANH`: map mã 2 chữ số → tên tỉnh/thành (được điền khá đầy đủ).

#### 18.2 Định dạng (patterns) bằng regex

Notebook định nghĩa nhiều pattern:
- Ô tô: `30K-216.25` (dạng 5 số + '.' + 2 số) hoặc biến thể không dấu '.'
- Ô tô khác: `30K-216.25` dạng có dấu '.'
- Xe máy: `30K1-12345` (5 số)
- Xe máy dạng có '.' : `30K1-123.45`

Mỗi pattern gắn với nhãn loại xe (`Ô tô ...`, `Xe máy ...`).

#### 18.3 Kiểm tra hợp lệ

Hàm `kiem_tra_bien_so(plate)`:
1. Upper + strip.
2. Với mỗi regex pattern:
   - Nếu match:
     - Lấy `ma_tinh` từ group tương ứng
     - Lấy `series` (chữ cái)
     - Lấy số thứ tự
     - Kiểm tra `ma_tinh` có trong map hay không
     - Kiểm tra `series` nằm trong `SERIES_VALID`
3. Nếu không match bất kỳ pattern → invalid.

Kết quả cuối:
- `hợp lệ = matched và không có lỗi`

---

## 4. Output / Artifact

- `license_plate_crop.jpg`: crop biển theo `lp_rect` (trong Cell 4)
- `ket_qua_bien_so.png`: ảnh tổng hợp kết quả cuối (Cell 19)

Ngoài ra notebook hiển thị nhiều figure trung gian:
- mask bbox xe
- vùng biển và khung chữ nhật
- ảnh preprocess qua từng bước
- ảnh ký tự segment

---

## 5. Quy trình pipeline (dạng sơ đồ text)

```
Ảnh đầu vào
  |
  v
YOLO detect xe (conf=0.4)
  |
  v
Chọn bbox gần tâm ảnh nhất
  |
  v
Crop ROI nửa dưới xe
  |
  v
Threshold + contours -> chọn lp_rect theo điều kiện hình học
  |
  v
Crop biển: lp_crop
  |
  v
Zoom -> Resize chuẩn height=256 + Sharpen
  |
  v
Gray -> Denoise (NLM)
  |
  v
CLAHE -> Threshold (Otsu)
  |
  v
Morphology (Open)
  |
  v
Clean viền rìa (crop pad)
  |
  v
Deskew (HoughLines + rotate)
  |
  v
Padding -> Erosion
  |
  v
Segment ký tự (2 dòng + tách theo cột)
  |
  v
OCR ký tự (EasyOCR + augment + allowlist + rule dấu chấm .)
  |
  v
Ghép chuỗi: text1-text2
  |
  v
Rule-check biển số VN (regex + tỉnh/thành)
  |
  v
Hiển thị + lưu ảnh kết quả
```

---

## 6. Thách thức lớn nhất

1. **Crop vùng biển bằng heuristic**
   - Không dùng YOLO trực tiếp cho biển nên phụ thuộc mạnh vào:
     - góc chụp
     - độ sáng/contrast
     - nền và vật cản
     - tham số threshold/điều kiện aspect ratio, area, w/h

2. **Chất lượng ảnh biển khi scale nhỏ**
   - Nếu biển quá nhỏ hoặc mờ, OCR sẽ “thiếu nét”.
   - Notebook cố gắng bù bằng zoom/upscale + sharpen + denoise + CLAHE, nhưng không thể phục hồi thông tin bị mất.

3. **Threshold/morphology tuning**
   - Chỉ cần sai ngưỡng có thể làm:
     - mất ký tự (nét biến mất)
     - hoặc dính ký tự (nét nối thành cụm)
   - Erosion quá mạnh cũng dễ làm hỏng dáng chữ.

4. **Segmentation theo tổng cột dễ gãy**
   - Khi nét bị đứt đoạn hoặc dính lại, các đoạn x1-x2 có thể sai.
   - Hàm lọc theo `min_width`, `median_w` giúp giảm, nhưng không đảm bảo mọi trường hợp.

5. **OCR confusion giữa ký tự gần giống**
   - Ví dụ: `O/0`, `I/1`, `S/5`, `B/8`, v.v.
   - Notebook giảm bằng augment và rule dấu chấm, nhưng vẫn phụ thuộc vào ảnh.

6. **Rule-check regex cần phủ đủ biến thể thực tế**
   - Nếu regex thiếu trường hợp hợp lệ hoặc tỉnh/thành map chưa đầy đủ → có thể đánh invalid dù OCR đúng.

---

## 7. Gợi ý cải tiến (ngắn gọn, mang tính thực dụng)

- Thay heuristic crop biển bằng **model YOLO riêng cho biển số** (dùng `yolov8n_plate.pt` nếu có dataset biển) để giảm lỗi bước 4.
- Thay segmentation cột đơn giản bằng:
  - connected components + contour filtering
  - hoặc dựa trên projection profile có post-processing theo tỷ lệ ký tự.
- Thay OCR ký tự “independent” bằng decoding theo chuỗi (beam search kiểu sequence) hoặc dùng một mô hình OCR end-to-end biển.
- Rule-check có thể kết hợp thêm:
  - kiểm tra hình học khoảng cách ký tự
  - hoặc xác suất candidate theo n-gram (dạng chuỗi hợp lệ).

---

## 8. Ghi chú

- Notebook hiện set `gpu=False` cho EasyOCR.
- `allowlist` được giới hạn để giảm false-positive.
- Các hình vẽ trung gian trong notebook nhằm mục đích “tuning tham số” và debug.

---

Nếu bạn muốn, có thể tách pipeline thành file `.py` (ví dụ `predict.py`) để chạy nhanh cho nhiều ảnh thay vì chạy từng cell trong notebook.
