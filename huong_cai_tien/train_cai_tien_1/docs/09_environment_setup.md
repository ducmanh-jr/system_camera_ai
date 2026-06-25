# 09 — Environment setup & training trên Kaggle / Colab / Local

## 9.1 Local (Linux/Windows + GPU)

```bash
# 1. Tạo virtual env
python -m venv .venv
source .venv/bin/activate           # (Linux/Mac) hoặc .venv\Scripts\activate (Windows)

# 2. PyTorch (chọn phiên bản phù hợp CUDA)
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Phần còn lại
pip install -r requirements.txt

# 4. Train
python scripts/train.py \
    --config configs/crnn_base.yaml \
    --data-root /path/to/ocr_dataset \
    --out-dir runs/exp1

# 5. Monitor (mở tab khác)
tensorboard --logdir runs/
```

## 9.2 Google Colab

```python
# Cell 1: clone & install
!git clone <your_repo_url> vn_alpr_ocr_from_scratch
%cd vn_alpr_ocr_from_scratch
!pip install -q -r requirements.txt

# Cell 2: mount Drive để lưu ckpt
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: copy dataset (nếu chưa upload Drive sẵn)
!cp -r /content/drive/MyDrive/ocr_dataset /content/

# Cell 4: train
!python scripts/train.py \
    --config configs/crnn_base.yaml \
    --data-root /content/ocr_dataset \
    --out-dir /content/drive/MyDrive/runs/colab_exp1

# Cell 5: TensorBoard inline
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/runs/colab_exp1/tb
```

Thời gian tham khảo Colab T4 / L4: 200 epoch ~ 4-8h.

## 9.3 Kaggle

Setup `kaggle.json` để upload dataset, hoặc dùng dataset đã có công khai trên Kaggle.

Kaggle Notebook:

```python
# Cell 1
!pip install -q albumentations==1.3.1 python-Levenshtein
!git clone <your_repo_url> /kaggle/working/vn_alpr_ocr_from_scratch
%cd /kaggle/working/vn_alpr_ocr_from_scratch

# Cell 2 — train
!python scripts/train.py \
    --config configs/crnn_base.yaml \
    --data-root /kaggle/input/vn-license-plate-ocr/ocr_dataset \
    --out-dir /kaggle/working/runs/kaggle_exp1
```

Kaggle giới hạn session ~9h GPU → cấu hình `early_stop_patience=20` và checkpoint mỗi 5 epoch để có thể resume sang session sau.

## 9.4 DDP Multi-GPU local

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
    --config configs/crnn_base.yaml \
    --ddp \
    --data-root /path/to/ocr_dataset \
    --out-dir runs/ddp_exp1
```

## 9.5 Monitor

- **TensorBoard**: real-time `train/loss`, `train/lr`, `val/seq_acc`, `val/cer`. Mở `tensorboard --logdir runs/`.
- **train.log**: file log text-based ở `runs/.../train.log`.
- **W&B** (optional): bật `output.wandb=true` sau khi cài và setup token.

## 9.6 Reproducibility

- `torch.manual_seed(cfg.seed)` ở entry.
- `torch.backends.cudnn.benchmark = True` (nhanh hơn, nhưng non-deterministic). Set `False` + `cudnn.deterministic = True` nếu cần reproducibility tuyệt đối.
- Cố định data sampler seed: `DataLoader(..., generator=torch.Generator().manual_seed(s))`.

## 9.7 Disk usage

- Real dataset: 200-500 MB (depending on resolution)
- 50K synthetic: ~1-2 GB
- 1 checkpoint base: ~25 MB
- TensorBoard logs: ~50-200 MB sau 200 epoch

Chuẩn bị ~5 GB free.

## 9.8 Trouble-shooting nhanh

| Lỗi                                                    | Fix                                          |
| ------------------------------------------------------ | --------------------------------------------- |
| `CUDA out of memory`                                   | Hạ batch_size, bật grad accumulation         |
| `RuntimeError: zero infinity`                          | Set `zero_infinity=True` (đã default)        |
| `albumentations` import error trên Colab               | `pip install -U albumentations`              |
| ONNX export báo "unsupported op"                        | Tăng `--opset 17` hoặc `19`                   |
| TensorBoard "no scalar"                                 | Chờ ≥ 1 epoch hoặc check `output.tensorboard:true` |
