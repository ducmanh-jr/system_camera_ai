# Báo cáo đánh giá metrics OCR

Ngày chạy: 2026-06-23  
Dataset test: `../content/ocr_dataset/test_labels.txt`  
Số mẫu test: `1219`  
Decoder: `greedy`  
Batch size: `128`  
Thiết bị chạy: tự động theo `scripts/evaluate.py` (`cuda` nếu có, ngược lại `cpu`)

## File đánh giá có sẵn trong repo

Repo đã có sẵn file code đánh giá metrics:

- `scripts/evaluate.py`: chạy model trên split `val` hoặc `test`
- `src/utils/metrics.py`: tính `Sequence Accuracy`, `Character Accuracy`, `CER`, `WER`

Lệnh đã chạy:

```bash
python scripts/evaluate.py --ckpt runs/crnn_base/best.pt --split test --data-root ../content/ocr_dataset --batch-size 128 --num-workers 0
python scripts/evaluate.py --ckpt runs/crnn_base/last.pt --split test --data-root ../content/ocr_dataset --batch-size 128 --num-workers 0
```

## Kết quả metrics

| Experiment | Checkpoint | Split | Samples | Sequence Accuracy | Character Accuracy | CER | WER |
|---|---|---:|---:|---:|---:|---:|---:|
| `crnn_base` | `runs/crnn_base/best.pt` | `test` | 1219 | 0.8310 | 0.9568 | 0.0432 | 0.1107 |
| `crnn_base` | `runs/crnn_base/last.pt` | `test` | 1219 | 0.8269 | 0.9572 | 0.0428 | 0.1120 |

## Nhận xét nhanh

- `best.pt` có `Sequence Accuracy` tốt hơn: `83.10%` so với `82.69%`.
- `last.pt` có `CER` thấp hơn một chút: `4.28%` so với `4.32%`.
- Nếu ưu tiên dự đoán đúng toàn bộ biển số, nên dùng `best.pt`.
- Nếu ưu tiên lỗi ký tự trung bình thấp hơn rất nhẹ, `last.pt` nhỉnh hơn theo `CER`, nhưng chênh lệch rất nhỏ.

## File mismatch

Script đánh giá sinh file các mẫu dự đoán sai tại:

```text
runs/crnn_base/eval_test_mismatches.csv
```

Lưu ý: cả `best.pt` và `last.pt` cùng ghi vào đường dẫn này, nên file CSV hiện tại là kết quả của lần chạy cuối cùng (`last.pt`).
