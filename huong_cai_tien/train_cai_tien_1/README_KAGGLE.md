# Hướng dẫn chạy training trên Kaggle

Tài liệu này hướng dẫn chi tiết cách chạy huấn luyện `vn_alpr_ocr_from_scratch` trên môi trường Kaggle Notebook (GPU). Ngắn gọn, trực tiếp và có ví dụ cell để copy-paste.

---

## 1. Tổng quan workflow

- Upload toàn bộ project `vn_alpr_ocr_from_scratch` và thư mục dataset `content/ocr_dataset` lên một Kaggle Dataset duy nhất (gọi là `<DATASET_NAME>`). Hoặc chỉ upload `content/ocr_dataset` và clone repo code nếu notebook có internet.
- Tạo một Kaggle Notebook mới, attach dataset `<DATASET_NAME>`, bật GPU.
- Cài dependencies, set `--data-root` trỏ tới `/kaggle/input/<DATASET_NAME>/content/ocr_dataset`.
- Chạy `scripts/train.py` (single-GPU). Lưu checkpoints vào `/kaggle/working/`.
- Sau khi xong, tải checkpoints/outputs từ tab `Output` hoặc tạo Dataset mới để lưu lâu dài.

---

## 2. Chuẩn bị Dataset trên Kaggle

1. Nén (zip) hoặc upload trực tiếp thư mục project + dataset lên Kaggle Datasets:
   - Vào `https://www.kaggle.com/` → `Datasets` → `New Dataset` → Upload
   - Tên dataset: `my-vn-ocr-dataset` (ví dụ). Ghi nhớ slug dataset (ví dụ `username/my-vn-ocr-dataset`).
2. Nội dung dataset cần có (ít nhất):
   - `vn_alpr_ocr_from_scratch/` (toàn bộ mã nguồn và `requirements.txt`)
   - `content/ocr_dataset/` (thư mục `train/`, `val/`, `test/` và `*_labels.txt`)

Ghi chú: nếu bạn không muốn upload code, bạn có thể chỉ upload `content/ocr_dataset` và clone mã từ GitHub trong Notebook (yêu cầu Internet bật).

---

## 3. Tạo và cấu hình Kaggle Notebook

1. Tạo Notebook: `New Notebook` → Settings:
   - `Accelerator`: GPU (T4/RTX tùy có sẵn)
   - `Internet`: Bật nếu bạn cần `git clone` hoặc pip install packages không có sẵn
2. Add data → chọn dataset `<DATASET_NAME>` (sẽ mount vào `/kaggle/input/<DATASET_NAME>`)
3. Working directory: mặc định là `/kaggle/working/`.

---

## 4. Các cell mẫu trong Notebook

1. Kiểm tra GPU & Python

```python
# Kiểm tra GPU và PyTorch
!nvidia-smi
import torch
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
```

2. Cài dependencies

Trường hợp bạn đã upload `vn_alpr_ocr_from_scratch/requirements.txt` trong dataset:

```bash
pip install -r /kaggle/input/<DATASET_NAME>/vn_alpr_ocr_from_scratch/requirements.txt
# Nếu pip gặp lỗi với một số package, thử thêm --no-deps hoặc cài từng package
```

Nếu bạn đã clone repo vào `/kaggle/working/` (internet bật):

```bash
git clone https://github.com/<you>/vn_alpr_ocr_from_scratch.git
cd vn_alpr_ocr_from_scratch
pip install -r requirements.txt
```

3. Kiểm tra cấu trúc dataset

```python
import os
DATA_ROOT = '/kaggle/input/<DATASET_NAME>/content/ocr_dataset'
print(os.listdir('/kaggle/input/<DATASET_NAME>')[:20])
print('train files:', len(os.listdir(os.path.join(DATA_ROOT,'train'))))
print('train_labels:', os.path.exists(os.path.join(DATA_ROOT,'train_labels.txt')))
```

4. Chạy training (single-GPU) — lệnh mẫu

```bash
python /kaggle/input/<DATASET_NAME>/vn_alpr_ocr_from_scratch/scripts/train.py \
  --config /kaggle/input/<DATASET_NAME>/vn_alpr_ocr_from_scratch/configs/crnn_base.yaml \
  --data-root /kaggle/input/<DATASET_NAME>/content/ocr_dataset \
  --out-dir /kaggle/working/runs/exp_kaggle
```

Ghi chú:

- Nếu bạn đã `git clone` repo vào working dir cho sẵn, thay đường dẫn vào `scripts/train.py` và `configs/...` tương ứng (không cần `/kaggle/input/...`).
- `--out-dir /kaggle/working/` là bắt buộc để kết quả xuất ra tab `Output` của Kaggle.

5. Chạy nhanh thử với cấu hình nhẹ (debug)

```bash
python vn_alpr_ocr_from_scratch/scripts/train.py \
  --config vn_alpr_ocr_from_scratch/configs/crnn_small.yaml \
  --data-root /kaggle/input/<DATASET_NAME>/content/ocr_dataset \
  --out-dir /kaggle/working/runs/debug
```

6. Resume từ checkpoint

```bash
python vn_alpr_ocr_from_scratch/scripts/train.py \
  --config vn_alpr_ocr_from_scratch/configs/crnn_base.yaml \
  --resume /kaggle/working/runs/exp_kaggle/last.pt \
  --data-root /kaggle/input/<DATASET_NAME>/content/ocr_dataset
```

---

## 5. Lưu / tải checkpoint và kết quả

- Files được ghi vào `/kaggle/working/` sẽ xuất hiện trong tab `Output` → bạn có thể tải file trực tiếp.
- Để giữ lâu dài, tạo Dataset mới từ Notebook: `Add to Dataset` → chọn files trong `/kaggle/working/` → `Create New Dataset`.

---

## 6. Mẹo và lưu ý quan trọng

- Kaggle Notebook thường có giới hạn runtime (12 giờ). Chia training thành nhiều runs bằng cách checkpoint & resume.
- Kaggle chỉ cung cấp single GPU cho Notebook (không hỗ trợ DDP đa-GPU trong notebook). Multi-GPU chỉ khả dụng trên các cluster riêng/compute.
- Nếu runtime bị reset: lưu checkpoint định kỳ vào `/kaggle/working/` (Trainer trong repo đã lưu best/last mặc định).
- Nếu không có Internet: upload toàn bộ mã + dữ liệu lên Dataset trước khi mở Notebook.
- Để nhanh debug, dùng `crnn_small.yaml` hoặc giảm `epochs`/`batch_size`.

---

## 7. Troubleshooting nhanh

- Lỗi thiếu package khi pip install: cài thủ công package thiếu hoặc bật `Internet` trong Notebook và thử lại.
- Lỗi CUDA out-of-memory: giảm `batch_size` trong config hoặc giảm `image size`/dùng `crnn_small.yaml`.
- Không tìm thấy data: kiểm tra `--data-root` trỏ đúng tới `/kaggle/input/<DATASET_NAME>/content/ocr_dataset`.

---

## 8. Muốn mình làm tiếp gì?

- Tạo sẵn một Notebook mẫu (`kaggle_train_example.ipynb`) với các cell ở trên (mình có thể tạo file ini trong repo).
- Hoặc mình có thể cập nhật `README.md` chính để chèn mục tóm tắt Kaggle.

---

File này do assistant sinh tự động — cần chỉnh `DATASET_NAME` trước khi chạy.
