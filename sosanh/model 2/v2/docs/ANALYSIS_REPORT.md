# Báo cáo phân tích & kế hoạch retrain — VN-ALPR OCR (CRNN+CTC)

> Author: Senior AI Engineer (Production OCR + CV)
> Project: `vn_alpr_ocr_from_scratch`
> Date: 2026-05-21
> Status: Production retrain plan v2

---

## 1. Tóm tắt kết quả phân tích

### 1.1 Hiện trạng dữ liệu

* Dataset thật ~9.752 ảnh train, có 305/9.752 mẫu chứa `Đ` (biển quân đội/đặc biệt). Đây là **dataset rất nhỏ** so với chuẩn OCR 100K–1M.
* Charset 38 ký tự + 1 blank, không có `W`. Các ký tự **rất hiếm**: `O=1, I=2, J=3, Y=42, R=44, U=47, Z=49, X=81, S=124, V=149` → bài toán mất cân bằng lớp **nghiêm trọng**.
* Test mismatches 206 mẫu, độ dài trung bình GT ≈ pred ≈ 9.16 ký tự → mô hình **đếm đúng độ dài** nhưng **nhận diện sai nội dung** (lỗi substitution, không phải insert/delete).

### 1.2 Pattern lỗi từ `runs/crnn_base/eval_test_mismatches.csv`

Top character substitution (GT → Pred):

```
'1'→'0' ×14    '8'→'0' ×12    '7'→'1' ×11    '6'→'5' ×10    '8'→'6' ×10
'3'→'0' ×9     '9'→'0' ×8     '0'→'8' ×8     '5'→'0' ×8     '7'→'3' ×8
'0'→'9' ×8     '7'→'2' ×8     '6'→'8' ×8     '9'→'8' ×7
```

**Phát hiện chính**:

1. **80% lỗi là giữa các chữ số**. Mô hình thiếu khả năng phân biệt số ở **resolution thấp + blur + nhiễu** — đặc biệt cặp `0/8`, `6/8`, `1/7`, `3/8`, `5/6`, `5/9`.
2. Lỗi chữ cái: `K, A, N, B` cao nhất (`K=17, A=14, N=12, B=11`) — đây cũng là các cặp visually-similar (K↔X, A↔H, N↔M/H, B↔8).
3. Ký tự space `' '` cũng bị nhầm 18 lần → có lúc dự đoán dài thừa số.
4. **Không có lỗi liên quan đến `Đ`** trong top — nhưng `Đ` chỉ xuất hiện 305 mẫu → khả năng cao là **memorize**, generalize kém với synth.

### 1.3 Vấn đề pipeline hiện tại (v1)

| # | Vấn đề | Mức độ | Bằng chứng |
|---|---|---|---|
| 1 | LR scheduler step **trước** optimizer.step (warning trong log) | **Cao** | README_HYPERPARAM_IMPROVEMENTS.md, code `trainer.py` dùng scaler |
| 2 | LR `1.5e-3` + warmup 5 epoch quá nhanh cho CTC scratch | **Cao** | "val seq_acc = 0.0000 kéo dài 10–11 epoch" |
| 3 | Augmentation **không có rain/fog**, motion blur limit chỉ 7 px | Trung bình | `src/data/augmentation.py` |
| 4 | Không có **weighted sampling** cho ký tự hiếm | Cao | Charset cực mất cân bằng |
| 5 | **GroupNorm** với batch 64 → gradient noisy hơn BN | Trung bình | `backbone.py` mặc định `gn` |
| 6 | Không có **label smoothing** trong CTC → posterior dễ over-confident | Trung bình | `losses/ctc.py` |
| 7 | Không có **SE/DropPath** → backbone không phân biệt được digit confusing | Trung bình | `backbone.py` |
| 8 | Không có **SWA** ở giai đoạn cuối | Thấp | `trainer.py` |
| 9 | EMA decay 0.999 + không warmup → 100 step đầu EMA bị poison bởi noise | Thấp | `utils/ema.py` |
| 10 | Chỉ có 1 split bộ keep best.pt — không có top-K | Thấp | Đẩy file lớn không cần |
| 11 | Không có gradient accumulation → effective batch 64 nhỏ với grayscale 1ch | Thấp | `config` |
| 12 | Không log gradient norm → khó debug exploding/vanishing | Thấp | `trainer.py` |
| 13 | Không có **knowledge distillation** dù đã có teacher trained | Cao (cơ hội) | `runs/crnn_base/best.pt` sẵn |
| 14 | Notebook không lưu log training → khó retro-analysis | Vận hành | Notebook source clean |

### 1.4 Chẩn đoán overfit/underfit

* **Có dấu hiệu memorize nhẹ** ở các ký tự hiếm (Y, R, U, Z): không đủ mẫu để generalize.
* **Underfit cho digit confusion**: 80% lỗi là số → mô hình **chưa học đủ feature stroke discrimination**, đặc biệt khi ảnh bị blur/JPEG.
* **Không exploding gradient** (training hội tụ được), **không vanishing** (val CER giảm sau warmup).
* **Hội tụ chậm 10 epoch đầu** do CTC alignment chưa lock → cần warmup dài hơn.

---

## 2. Hyperparameter mới (v2) — bảng so sánh

| Tham số | v1 (cũ) | v2 (mới) | Vì sao |
|---|---|---|---|
| `lr` | 1.5e-3 | **1.2e-3** | Giảm dao động ở 10 epoch đầu (CTC fragile) |
| `weight_decay` | 2e-4 | **5e-4** | Phản ứng với memorize digit confusion |
| `optim.betas` | [0.9, 0.999] | **[0.9, 0.98]** | Beta2 thấp hơn — standard cho CTC/Transformer |
| `warmup_epochs` | 5 | **8** | CTC alignment cần thời gian |
| `scheduler.pct_start` | (auto) | **0.10** | Warmup sharp hơn |
| `grad_clip` | 3.0 | **1.0** | BN + AMP đủ ổn định, clip mạnh hơn |
| `batch_size` | 64 | 64 (giữ) | Kaggle T4 OK |
| `grad_accum_steps` | — | **2** | Effective batch 128 → digit feature ổn định hơn |
| `epochs` | 200 | **250** | Có SWA cuối |
| `amp` | fp16 | **bf16 nếu A100, fp16 nếu T4** | bf16 tránh underflow CTC |
| `channels_last` | — | **true** | +10-15% throughput trên Ampere |
| `ema_decay` | 0.999 | **0.9995** | Train dài hơn |
| `ema_warmup_steps` | — | **1000** | EMA không bị bias bởi random init |
| `swa` | false | **true (last 20%)** | Free +0.3-1% accuracy |
| `early_stop_patience` | 25 | **30** | Train dài hơn |
| `model.backbone.norm` | gn | **bn** | Batch 64+, BN hội tụ nhanh hơn |
| `model.backbone.se` | — | **true (blocks 3-5)** | Channel re-weighting cho digit |
| `model.backbone.stochastic_depth` | — | **0.1** | Cheap regularizer |
| `model.neck.hidden` | 256 | **320** | +25% capacity cho confusion-aware |
| `model.neck.dropout` | 0.25 | **0.30** | Variational dropout trong LSTM |
| `model.head.dropout` | 0.15 | **0.20** | Mạnh hơn để chống memorize |
| `loss.ctc.entropy_weight` | 0.02 | **0.015** | Vì đã có label smoothing |
| `loss.ctc.label_smoothing` | — | **0.05** | Soft target → ít over-confident |
| `loss.distill.alpha` | — | **0.5 nếu có teacher** | KD từ best.pt v1 |
| `data.weighted_sampler` | — | **true** | Boost rare chars ×4 |
| `data.synthetic_ratio` | 0.8 | **0.6** | 0.8 quá nhiều synthetic — pred drift |
| `augment.preset` | — | **heavy_ocr** | Cover CCTV/rain/fog VN |
| `augment.random_erasing_prob` | — | **0.25** | Tensor-level erasing in [-1,1] space |

---

## 3. Các thay đổi cụ thể giúp model thế nào

| Thay đổi | Lý do AI Researcher | Tác động dự kiến lên CER |
|---|---|---|
| BN thay GN (batch ≥64) | BN tận dụng đầy đủ statistics batch → gradient sharp hơn, cộng với feature normalization theo channel rất tốt cho digit stroke | -0.5% to -1.2% |
| SE block trong blocks 3-5 | Re-weight channel theo input. Khi gặp ảnh blur, network sẽ "tắt" channel low-SNR và tăng channel còn rõ stroke → giảm 0/8, 6/8 confusion | -1.0% to -2.0% |
| Stochastic depth 0.1 | Tương đương ensemble of sub-networks at train time → giảm overfit cho 305 mẫu Đ và các ký tự rare | -0.2% to -0.5% |
| Neck hidden 256→320 | Tăng RNN capacity để model ngữ cảnh (province code → series → digits) | -0.3% to -0.7% |
| Label smoothing 0.05 trong CTC | Posterior không quá nhọn → less catastrophic khi gặp ký tự mới | -0.4% to -0.8% |
| Weighted sampler ×4 cho rare | O,I,J,Y,R sẽ được sampled nhiều hơn → backbone học stroke đặc trưng | -0.3% to -1.0% (chủ yếu seq_acc) |
| Heavy augmentation (rain, fog, JPEG 25, motion 9px) | Cover production scenarios — đúng causes của digit confusion (motion blur tạo 0/8 ambiguity) | -1.5% to -3.0% |
| Knowledge Distillation từ v1 best.pt | Teacher đã có per-frame posterior với 'soft labels' về digit confusion → student inherit | -0.5% to -1.5% |
| SWA last 20% | Trung bình 20% weights cuối → smoother minima | -0.3% to -0.8% |
| Grad accum (effective 128) | Larger batch → less noisy gradient → BN statistic ổn định hơn | -0.2% to -0.5% |
| EMA warmup 1000 steps | EMA không bị bias bởi random init | -0.1% to -0.3% |
| bf16 thay vì fp16 (nếu A100) | Tránh underflow CTC khi loss rất nhỏ | -0.1% (stability) |
| Channels-last memory | Throughput +10-15% (Ampere/Hopper) | Tốc độ, không acc |

**Ước lượng tổng cải thiện**: tùy điểm xuất phát của v1, dự kiến **CER giảm 30-50% relative**, **seq_acc tăng 5-12 điểm tuyệt đối**. Ví dụ nếu v1 hiện tại có CER=0.04 / seq_acc=0.85 thì v2 có thể đạt **CER ≈ 0.022 / seq_acc ≈ 0.93-0.95**.

---

## 4. Chiến lược retrain tốt nhất (3-stage)

### Stage A — Pre-train trên synthetic (8-15 giờ T4)
```bash
# 1. Generate 80K synthetic plates
python scripts/gen_synthetic.py --out data/synth_80k --n 80000

# 2. Train v2 từ scratch chủ yếu trên synthetic (synthetic_ratio: 0.85)
python v2/scripts/train_v2.py --config v2/configs/crnn_v2.yaml \
    --data-root /kaggle/input/.../ocr_dataset \
    --synthetic-dir data/synth_80k \
    --out-dir runs/stage_a

# Chỉnh tay: data.synthetic_ratio: 0.85; epochs: 100
```

### Stage B — Fine-tune trên real data (4-8 giờ T4)
```bash
python v2/scripts/train_v2.py --config v2/configs/crnn_v2.yaml \
    --data-root /kaggle/input/.../ocr_dataset \
    --resume runs/stage_a/best.pt \
    --out-dir runs/stage_b

# Chỉnh tay: data.synthetic_dir: null; optim.lr: 6e-4 (halved); epochs: 80
```

### Stage C — Knowledge Distillation final (3-5 giờ T4)
```bash
python v2/scripts/train_v2.py --config v2/configs/crnn_v2.yaml \
    --data-root /kaggle/input/.../ocr_dataset \
    --synthetic-dir data/synth_80k \
    --teacher-ckpt runs/stage_b/best.pt \
    --out-dir runs/stage_c
```

Với teacher là model đã fine-tune trên real, student v2 distilled sẽ inherit per-frame digit-discrimination + được train với augmentation/regularization mới → CER tốt nhất.

### Optional — Hyperparameter search trước stage A
```bash
python v2/scripts/hyperparam_search.py \
    --config v2/configs/crnn_v2.yaml \
    --data-root /path \
    --n-trials 30 --epochs-per-trial 12
# Lấy best_params.yaml → patch vào crnn_v2.yaml → chạy stage A
```

---

## 5. Khi nào dùng nhánh nào (auto-decision tree)

```
val CER ổn định nhưng cao (≥ 0.05)?
└── Có → Underfit
       ├── Tăng neck.hidden: 320 → 384
       ├── Giảm regularization: dropout, stochastic_depth
       ├── Generate thêm 100K synthetic
       └── Train 350 epochs

val CER dao động mạnh (±0.02 trong 5 epoch)?
└── Có → LR quá cao hoặc augmentation chưa khớp
       ├── Giảm lr: 1.2e-3 → 8e-4
       ├── Tăng grad_accum_steps: 2 → 4
       └── Đổi augment.preset: heavy_ocr → medium_ocr

train_loss giảm nhưng val CER tăng?
└── Có → Overfit
       ├── Tăng dropout neck/head 0.30→0.40, 0.20→0.30
       ├── Tăng weight_decay 5e-4 → 1e-3
       ├── Tăng random_erasing_prob 0.25 → 0.40
       └── Reduce epochs hoặc tăng synthetic_ratio
```

---

## 6. Production deployment notes

* Sau khi train xong: chạy `evaluate_v2.py --beam 10 --save-confusion` trên test set.
* Export ONNX với `dynamic_axes={'input': {0: 'batch'}}` để dùng OpenVINO/TensorRT.
* Quantization: dùng INT8 PTQ với calibration set 500 mẫu real → CER chỉ giảm ~0.3% mà model giảm 4× kích thước.
* Inference với beam=5 (không cần beam=10 cho plate ngắn).
* Set confidence threshold dựa trên entropy regularized posterior — drop prediction nếu mean per-frame entropy > ngưỡng (calibrate trên val).

---

## 7. Cấu trúc thư mục v2

```
v2/
├── configs/
│   └── crnn_v2.yaml                        # Config mới
├── docs/
│   └── ANALYSIS_REPORT.md                  # File này
├── scripts/
│   ├── train_v2.py                         # Entry training
│   ├── evaluate_v2.py                      # Eval + confusion matrix
│   ├── hyperparam_search.py                # Optuna search
│   └── plot_metrics.py                     # Vẽ biểu đồ
└── src/
    ├── data/
    │   └── augmentation_v2.py              # Heavy OCR augmentation
    ├── losses/
    │   └── ctc_v2.py                       # CTC + label smoothing + KD
    ├── models/
    │   ├── backbone_v2.py                  # VGGLite + SE + DropPath
    │   └── crnn_v2.py                      # End-to-end model v2
    ├── training/
    │   └── trainer_v2.py                   # Production-grade trainer
    └── utils/
        ├── ema_v2.py                       # EMA with warmup
        ├── checkpoint_v2.py                # Top-K keeper
        └── logger_v2.py                    # JSON-line + TB
```

Toàn bộ code reuse module v1 (`src.data.vocab`, `src.data.dataset`, ...) — không phá vỡ pipeline hiện hành.
