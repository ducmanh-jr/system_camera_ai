# 08 — Inference & Deploy

## 8.1 PyTorch FP32 (baseline)

```python
from src.inference.predictor import Predictor
pred = Predictor.from_checkpoint("runs/crnn_base/best.pt")
text = pred.predict_image("plate.jpg")
```

Latency tham khảo (CRNN base, batch=1, 48×192):

| Device           | FP32   | FP16   |
| ---------------- | ------ | ------ |
| RTX 3060         | ~10 ms | ~4 ms  |
| RTX A6000        | ~6 ms  | ~2 ms  |
| Jetson Orin Nano | ~40 ms | ~22 ms |
| CPU i7-12700H    | ~25 ms | (same) |

## 8.2 Half precision (FP16)

```python
model = model.half().eval()
x = x.half()
with torch.inference_mode():
    out = model(x)
```

Lưu ý: CTCLoss không hỗ trợ FP16 → chỉ inference, không train. Trainer đã dùng AMP (mixed) đúng cách.

## 8.3 TorchScript

```python
traced = torch.jit.trace(model.cpu(), torch.zeros(1,1,48,192))
torch.jit.save(traced, "model_ts.pt")
```

Hoặc `torch.jit.script` (yêu cầu type annotations đầy đủ — CRNN của chúng ta tương thích).

## 8.4 ONNX export

```bash
python scripts/export_onnx.py --ckpt runs/crnn_base/best.pt --out model.onnx
```

Đã dynamic_axes batch + width. Có thể chạy với:

```python
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
logits = sess.run(["logits"], {"image": np_input})[0]   # (T, B, C)
```

## 8.5 TensorRT (cho NVIDIA edge)

```bash
trtexec --onnx=model.onnx --fp16 --saveEngine=model_fp16.trt \
        --minShapes=image:1x1x48x96 \
        --optShapes=image:1x1x48x192 \
        --maxShapes=image:8x1x48x256
```

Trên Jetson Orin Nano FP16: ~12-18 ms.

## 8.6 Quantization (INT8)

- **PTQ (Post-training quantization)** với calibration set: dùng `torch.quantization` static. Conv layers OK; **LSTM khó quant tốt** → fall back FP16 hoặc bỏ neck (chỉ giữ CNN+CTC).
- **QAT (Quantization-Aware Training)** dynamic giả lập INT8 trong training: phức tạp với LSTM, chỉ dùng nếu deploy mobile bắt buộc.

Khuyên: deploy FP16 đủ, INT8 chỉ khi bắt buộc edge.

## 8.7 Batch inference trong ALPR pipeline

Trong 1 frame thường có nhiều plate. Gom thành batch GPU:

```python
plates = detect_plates(frame)              # list of cropped images
texts = predictor.predict_batch(plates)    # 1 forward pass, batch=N
```

Speedup ~5-10x so với gọi từng cái.

## 8.8 Postprocessing (rule-based)

Sau decode, có thể áp rule VN plate để loại bỏ false positive:

```python
import re
PATTERN = re.compile(r"^\d{2}[A-HKLMNPRSTUVXYZ]{1,2}\d? \d{4,5}$")
def is_valid_vn_plate(text):
    return bool(PATTERN.match(text))
```

Nếu prediction không match → fallback beam search hoặc reject.

## 8.9 Confidence scoring

Lấy `max(softmax)` trung bình theo timestep không-blank:

```python
log_probs = model.predict_logp(x)
probs = log_probs.exp()
maxp, idx = probs.max(dim=-1)        # (T, B)
# mask out blank timesteps
mask = idx != blank_idx
conf = (maxp * mask).sum(0) / mask.sum(0).clamp(min=1)
```

Threshold `conf < 0.85` → flag reject hoặc retry.

## 8.10 Online deploy stack tham khảo

```
[CCTV/IP cam]
   ↓ RTSP
[Frame grabber]
   ↓
[YOLO plate detector (separate model)]
   ↓ crops
[VN-Plate-OCR (this project) — TensorRT FP16]
   ↓ texts + conf
[Rule postprocess + DB lookup]
   ↓
[REST API / message queue]
```

End-to-end Jetson Orin Nano FP16: ~80-120 ms / frame với 1-3 plates.
