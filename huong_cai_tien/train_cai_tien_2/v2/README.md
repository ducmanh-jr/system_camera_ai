# VN-ALPR OCR v2 — Optimized Training Pipeline

Production-grade rewrite of the CRNN+CTC training pipeline for Vietnamese
license-plate OCR. See `docs/ANALYSIS_REPORT.md` for the full diagnosis and
hyperparameter justification.

## Quick start (Kaggle / Colab)

```bash
# 1. Install (uses same requirements.txt as v1; add optuna for HP search)
pip install -r requirements.txt
pip install optuna lion-pytorch        # optional but recommended

# 2. Train
python v2/scripts/train_v2.py \
    --config v2/configs/crnn_v2.yaml \
    --data-root /kaggle/input/datasets/.../ocr_dataset \
    --out-dir /kaggle/working/runs/crnn_v2

# 3. Evaluate
python v2/scripts/evaluate_v2.py \
    --ckpt /kaggle/working/runs/crnn_v2/best.pt \
    --data-root /kaggle/input/datasets/.../ocr_dataset --split test \
    --prefer-ema --beam 5 \
    --save-mismatches /kaggle/working/runs/crnn_v2/test_mismatches.csv \
    --save-confusion /kaggle/working/runs/crnn_v2/test_confusion.csv

# 4. Plot curves
python v2/scripts/plot_metrics.py --run /kaggle/working/runs/crnn_v2

# 5. (Optional) Hyperparameter search
python v2/scripts/hyperparam_search.py \
    --config v2/configs/crnn_v2.yaml \
    --data-root /kaggle/input/.../ocr_dataset \
    --n-trials 30 --epochs-per-trial 12

# 6. (Recommended) Knowledge distillation from v1 best.pt
python v2/scripts/train_v2.py \
    --config v2/configs/crnn_v2.yaml \
    --data-root /kaggle/input/.../ocr_dataset \
    --teacher-ckpt runs/crnn_base/best.pt \
    --out-dir /kaggle/working/runs/crnn_v2_kd
```

## What changed vs v1

* **Backbone**: VGGLite + Squeeze-Excitation in blocks 3-5 + Stochastic Depth.
* **Norm**: BatchNorm (with effective batch 128 via grad-accum) — faster
  convergence than GroupNorm at this batch size.
* **Loss**: CTC + label smoothing + entropy reg + Knowledge Distillation hook.
* **Augmentation**: heavy_ocr preset with rain, fog, motion blur up to 9 px,
  JPEG quality down to 25, random erasing in normalized space.
* **Optimizer**: AdamW with beta2=0.98, lr=1.2e-3, wd=5e-4, OneCycleLR with
  10% warmup.
* **Training**: bf16 (auto-fallback to fp16 on Turing/Volta), channels_last,
  grad-accum=2, EMA-warmup=1000, SWA last 20% of epochs, top-K checkpoints.
* **Sampling**: WeightedRandomSampler boosts samples containing rare chars
  (O, I, J, Y, R, U, Z, X, S, V, Đ) by ×4.
* **Diagnostics**: per-epoch metrics.jsonl, character confusion matrix,
  curves.png.

## Expected uplift

Based on the v1 mismatch analysis (~80% digit confusion, rare-char
underfitting), expect:

* CER: relative reduction of **30-50%** (e.g. 0.040 → 0.022).
* Sequence accuracy: absolute uplift of **+5 to +12 points**.

See `docs/ANALYSIS_REPORT.md` for the per-change effect estimates.
