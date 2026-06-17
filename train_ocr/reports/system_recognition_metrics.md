# Metrics đánh giá toàn bộ hệ thống nhận diện biển số

Phạm vi đánh giá:
- Đánh giá **toàn bộ hệ thống**, không đánh giá riêng từng module.
- Dữ liệu test: `ocr_dataset/test`
- Ground truth: `ocr_dataset/test_labels.txt`
- Trọng tâm: **khả năng nhận diện đúng biển số**
- Kết quả prediction: `reports/system_eval_predictions.csv`

## 1. Metrics chính

| Metric | Giá trị | Ý nghĩa |
|---|---:|---|
| Tổng số ảnh test | 1219 | Số ảnh được dùng để đánh giá |
| Số ảnh có label | 1219 | Toàn bộ ảnh test đều có ground truth |
| Character Accuracy | 93.04% | Mức độ đúng theo từng ký tự sau chuẩn hóa |
| Character Error Rate (CER) | 6.96% | Tỉ lệ lỗi ký tự, bằng `1 - Character Accuracy` |
| Full Plate Accuracy | 74.82% | Tỉ lệ đọc đúng toàn bộ biển số |
| Plate Error Rate | 25.18% | Tỉ lệ đọc sai toàn bộ biển số, bằng `1 - Full Plate Accuracy` |
| Average Edit Distance | 0.5751 | Trung bình số thao tác sửa ký tự để prediction khớp ground truth |
| Exact Match Count | 912/1219 | Số ảnh đọc đúng hoàn toàn |
| Average Latency | 74.56 ms/ảnh | Thời gian inference trung bình toàn pipeline |
| Median Latency | 65.75 ms/ảnh | Trung vị thời gian inference |
| P95 Latency | 136.74 ms/ảnh | 95% ảnh có thời gian xử lý không vượt quá mức này |
| Throughput | 13.41 ảnh/giây | Số ảnh hệ thống xử lý trung bình trong 1 giây |
| Error Rate | 0.00% | Tỉ lệ ảnh phát sinh lỗi inference |
| Average Validity Score | 12.25 | Điểm hợp lệ format biển số trung bình |

## 1.1. Điều kiện đánh giá

| Hạng mục | Giá trị |
|---|---|
| Dataset | `ocr_dataset/test` |
| Ground truth | `ocr_dataset/test_labels.txt` |
| Số lượng ảnh | 1219 |
| Số lượng label hợp lệ | 1219 |
| Phạm vi đánh giá | Toàn bộ hệ thống nhận diện biển số |
| Đối tượng đánh giá | Chuỗi ký tự biển số sau chuẩn hóa |
| Không đánh giá trong bảng này | Độ chính xác bbox detector, màu biển số, từng model/module riêng lẻ |
| Output prediction | `reports/system_eval_predictions.csv` |
| Script chạy đánh giá | `py src/pipeline.py --input-dir ocr_dataset/test --labels ocr_dataset/test_labels.txt --output-csv reports/system_eval_predictions.csv --evaluate` |

## 2. Phân phối lỗi theo edit distance

| Edit Distance | Số ảnh | Ý nghĩa |
|---:|---:|---|
| 0 | 912 | Đúng hoàn toàn |
| 1 | 142 | Sai/lệch 1 ký tự |
| 2 | 60 | Sai/lệch 2 ký tự |
| 3 | 46 | Sai/lệch 3 ký tự |
| 4 | 28 | Sai/lệch 4 ký tự |
| 5 | 9 | Sai/lệch 5 ký tự |
| 6 | 13 | Sai/lệch 6 ký tự |
| 7 | 6 | Sai/lệch 7 ký tự |
| 8 | 3 | Sai/lệch 8 ký tự |

## 3. Kết quả theo nhóm ảnh

| Nhóm ảnh | Số ảnh | Character Accuracy | Full Plate Accuracy |
|---|---:|---:|---:|
| `car` | 53 | 95.60% | 81.13% |
| `mb` | 65 | 97.12% | 84.62% |
| `type1` | 214 | 96.83% | 87.38% |
| `type2` | 179 | 88.14% | 54.19% |
| `type3` | 216 | 92.27% | 68.52% |
| `type4` | 203 | 93.56% | 78.33% |
| `type5` | 109 | 94.76% | 87.16% |
| `type6` | 12 | 91.67% | 66.67% |
| `type7` | 168 | 90.40% | 71.43% |

## 4. Kết luận ngắn

Hệ thống đạt độ chính xác ký tự cao, **93.04%**, nhưng độ chính xác đúng toàn bộ biển số là **74.82%**. Điều này cho thấy phần lớn lỗi là lỗi nhỏ theo ký tự, vì có thêm 142 ảnh chỉ sai 1 ký tự và 60 ảnh sai 2 ký tự. Nhóm ảnh khó nhất hiện tại là `type2`, với Full Plate Accuracy chỉ **54.19%**, nên đây là nhóm nên ưu tiên phân tích lỗi nếu muốn cải thiện hệ thống.
