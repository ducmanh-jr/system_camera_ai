# Bảng đánh giá metrics theo module

Dự án: Vietnamese License Plate OCR Ensemble

Nguồn tham chiếu:
- Pipeline chính: `main.py`, `src/pipeline.py`
- Kết quả hiện có: `output/results.csv`
- Benchmark đã xác minh: `README.md`

## 1. Bảng metrics đánh giá theo module

| STT | Module | Vai trò | Metrics cần đánh giá | Cách tính / ý nghĩa | Dữ liệu cần có | Kết quả hiện có |
|---:|---|---|---|---|---|---|
| 1 | Input loader | Đọc ảnh từ `input/` | Số ảnh hợp lệ, tỉ lệ ảnh lỗi, coverage theo loại ảnh | Đếm file ảnh đọc được / tổng file ảnh; thống kê prefix như `car`, `mb`, `type*` | Tập ảnh đầu vào | 44 ảnh trong `output/results.csv`; 42 `car`, 2 `mb` |
| 2 | Plate detector YOLO | Phát hiện và crop biển số | Detection rate, confidence trung bình, bbox quality, false crop rate | Detection rate = số ảnh có bbox / tổng ảnh; bbox quality cần IoU với label bbox | Ảnh + bbox ground truth để tính IoU chuẩn | Detection rate 44/44 = 100%; confidence TB 0.6648 |
| 3 | Crop fallback | Dùng crop detector khi prediction tốt hơn | Fallback usage rate, fallback win rate, fallback harm rate | Usage = số lần dùng fallback / tổng ảnh; win/harm cần so sánh prediction với ground truth | Label biển số đúng | Usage 0/44 = 0%; benchmark README cho thấy fallback tăng full plate accuracy 74.82% -> 75.39% |
| 4 | Train1 PaddleOCR | OCR bổ trợ, sửa lỗi serial/tail | Character accuracy, full plate accuracy, latency, error rate | Char acc = 1 - Levenshtein(pred, gt) / max_len; full plate acc = pred_norm == gt_norm | Label biển số đúng | Có log `train1_pred`; chưa có label trong CSV hiện tại để tính accuracy riêng |
| 5 | Train2 PyTorch CTC | OCR backbone chính | Character accuracy, full plate accuracy, latency, error rate | Tương tự Train1; đo thêm thời gian inference/model | Label biển số đúng | Có log `train2_pred`; README ghi mode default sau merge: char acc 93.04%, full plate acc 74.82% |
| 6 | OCR ensemble merge | Hợp nhất Train2 + Train1 | Selected accuracy, consensus rate, correction gain | So sánh output merge với từng model và ground truth; correction gain = merge đúng khi model đơn sai | Label + prediction từng model | 44/44 ảnh chọn `merged_train2_train1` |
| 7 | Normalize text | Chuẩn hóa ký tự biển số | Normalization validity, invalid char rate | Loại ký tự không hợp lệ; đo tỉ lệ output đúng format sau normalize | Prediction raw + normalized | CSV có `prediction` và `prediction_raw`; có thể thêm script tính invalid char rate nếu cần |
| 8 | Plate validity rules | Chấm điểm format biển số Việt Nam | Validity score, rule hit rate, rule false positive | Rule hit = prediction thỏa pattern biển số; false positive cần label | Prediction + label | Có logic `plate_validity_score`; `main.py` chưa xuất score ra CSV |
| 9 | Color classifier HSV | Phân loại màu nền biển số | Color distribution, color accuracy, unknown rate | Accuracy cần label màu; unknown rate = số `unknown` / tổng ảnh | Label màu biển số nếu cần accuracy | White 40/44 = 90.91%; Yellow 1/44 = 2.27%; Unknown 3/44 = 6.82% |
| 10 | Output writer | Ghi CSV, crop, ảnh annotate | CSV completeness, output image rate, crop saved rate | Số row CSV = số ảnh input; file output tồn tại / tổng ảnh | Ảnh input + output | CSV có 44 row; các trường chính đã có prediction, color, selected model, detector info |

## 2. Bảng kết quả tổng hợp hiện có

| Nhóm metric | Giá trị |
|---|---:|
| Tổng số ảnh đã ghi trong `output/results.csv` | 44 |
| Detector detected | 44/44 = 100% |
| Detector confidence trung bình | 0.6648 |
| Detector confidence nhỏ nhất | 0.5301 |
| Detector confidence lớn nhất | 0.7777 |
| Fallback được sử dụng | 0/44 = 0% |
| Selected model là `merged_train2_train1` | 44/44 = 100% |
| Màu biển `white` | 40/44 = 90.91% |
| Màu biển `yellow` | 1/44 = 2.27% |
| Màu biển `unknown` | 3/44 = 6.82% |

## 3. Benchmark accuracy đã xác minh

| Mode | Character Accuracy | Full Plate Accuracy | Ghi chú |
|---|---:|---:|---|
| Default OCR merge | 93.04% | 74.82% | Kết hợp Train2 backbone + Train1 repair/rule |
| Detector fallback | 93.06% | 75.39% | Tăng nhẹ accuracy khi crop detector được dùng an toàn |

## 4. Metrics nên báo cáo khi có ground truth

| Metric | Công thức | Áp dụng cho module |
|---|---|---|
| Character Accuracy | `1 - levenshtein(pred_norm, gt_norm) / max(len(pred_norm), len(gt_norm))` | Train1, Train2, Ensemble, Detector fallback |
| Full Plate Accuracy | `pred_norm == gt_norm` | Train1, Train2, Ensemble, Detector fallback |
| Detection Precision | `TP / (TP + FP)` | Plate detector |
| Detection Recall | `TP / (TP + FN)` | Plate detector |
| mAP@0.5 | AP trung bình với ngưỡng IoU 0.5 | Plate detector |
| Color Accuracy | `color_pred == color_gt` | Color classifier HSV |
| Avg Latency | Tổng thời gian inference / số ảnh | Detector, OCR model, pipeline tổng |
| Error Rate | Số ảnh bị exception / tổng ảnh | Tất cả module inference |

