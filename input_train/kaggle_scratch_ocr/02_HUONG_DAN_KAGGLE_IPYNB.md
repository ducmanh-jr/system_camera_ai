# Hướng dẫn upload, chạy Kaggle Notebook và tải kết quả

Tài liệu này hướng dẫn từng bước để chạy OCR biển số Việt Nam trên Kaggle bằng
notebook `.ipynb`.

Các tên dataset dùng trong hướng dẫn:

```text
Dataset ảnh: vietnam-plate-ocr-data
Dataset code: kaggle-scratch-ocr
GPU: T4 x2
```

Nếu bạn đặt tên dataset khác, hãy sửa lại đường dẫn trong các cell tương ứng.

## 1. Chuẩn bị dữ liệu trên máy

Bạn cần có 2 thư mục gốc:

```text
ocr_dataset/
kaggle_scratch_ocr/
```

Folder data phải có:

```text
ocr_dataset/
  train/
  val/
  test/
  train_labels.txt
  val_labels.txt
  test_labels.txt
```

Folder code phải có:

```text
kaggle_scratch_ocr/
  README.md
  01_Y_TUONG_CHINH.md
  02_HUONG_DAN_KAGGLE_IPYNB.md
  kaggle_train_t4x2.ipynb
  configs/
    t4x2_safe.yaml
    t4x2_strong.yaml
  prepare_dataset.py
  train_best.py
  train_scratch.py
  infer.py
```

Bạn có thể upload trực tiếp thư mục hoặc nén mỗi thư mục thành một file `.zip` rồi upload.
Trong trường hợp đang làm hiện tại, bạn đã nén đúng:

```text
ocr_dataset.zip
kaggle_scratch_ocr.zip
```

Khi upload file `.zip`, Kaggle thường tự giải nén và hiển thị lại thành thư mục bên trong dataset.
Nếu trên trang dataset bạn thấy:

```text
vietnam-plate-ocr-data/
  ocr_dataset/

kaggle-scratch-ocr/
  kaggle_scratch_ocr/
```

thì không cần upload lại. Đó là trạng thái đúng.

## 2. Upload dataset ảnh lên Kaggle

1. Vào [kaggle.com](https://www.kaggle.com) và đăng nhập.
2. Chọn `Datasets`.
3. Bấm `New Dataset`.
4. Kéo thả `ocr_dataset.zip` vào. Nếu chưa nén thì có thể kéo thả folder `ocr_dataset`.
5. Đặt tên dataset:

```text
vietnam-plate-ocr-data
```

6. Bấm `Create` hoặc `Create Dataset`.

Sau khi upload, đường dẫn trong notebook thường là:

```text
/kaggle/input/vietnam-plate-ocr-data/ocr_dataset
```

Nếu Data Explorer của Kaggle đang hiện `ocr_dataset` với các thư mục `train`, `val`, `test`
và các file label `.txt`, bạn đã upload đúng.

## 3. Upload code lên Kaggle

1. Chọn `Datasets`.
2. Bấm `New Dataset`.
3. Kéo thả `kaggle_scratch_ocr.zip` vào. Nếu chưa nén thì có thể kéo thả folder `kaggle_scratch_ocr`.
4. Đặt tên dataset:

```text
kaggle-scratch-ocr
```

5. Bấm `Create` hoặc `Create Dataset`.

Sau khi upload, đường dẫn trong notebook thường là:

```text
/kaggle/input/kaggle-scratch-ocr
```

Nếu Data Explorer của Kaggle đang hiện `kaggle_scratch_ocr` và bên trong có `train_best.py`,
`prepare_dataset.py`, `infer.py`, `configs/`, `kaggle_train_t4x2.ipynb`, bạn đã upload đúng.

## 4. Tạo Kaggle Notebook

1. Chọn `Code`.
2. Bấm `New Notebook`.
3. Ở panel bên phải, mở `Settings`.
4. Chọn GPU:

```text
Accelerator -> GPU T4 x2
```

5. Bấm `Add Data`.
6. Add dataset ảnh:

```text
vietnam-plate-ocr-data
```

7. Bấm `Add Data` lần nữa.
8. Add dataset code:

```text
kaggle-scratch-ocr
```

Bạn có 2 cách chạy:

- Mở file `kaggle_train_t4x2.ipynb` đã upload cùng dataset code.
- Hoặc tạo notebook mới và copy từng cell bên dưới.

Khuyến nghị chạy theo tầng:

```text
Ready check
-> Sanity run 3 epoch
-> Medium run 30 epoch
-> Full run
```

Không nên chạy full ngay từ đầu.

## 5. Cell 1: kiểm tra GPU và input

```python
import os
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

print("\n/kaggle/input:")
print(os.listdir("/kaggle/input"))
```

Kết quả mong muốn:

```text
CUDA available: True
GPU count: 2
0 Tesla T4
1 Tesla T4
```

Nếu chỉ có 1 GPU, vẫn chạy được nhưng nên giảm `batch-size`.

## 6. Cell 2: kiểm tra đường dẫn dataset

```python
from pathlib import Path
import shutil

INPUT_ROOT = Path("/kaggle/input")

print("/kaggle/input:")
for p in INPUT_ROOT.iterdir():
    print(" ", p)

data_candidates = [
    Path("/kaggle/input/vietnam-plate-ocr-data/ocr_dataset"),
    Path("/kaggle/input/vietnam-plate-ocr-data"),
]
code_candidates = [
    Path("/kaggle/input/kaggle-scratch-ocr"),
    Path("/kaggle/input/kaggle-scratch-ocr/kaggle_scratch_ocr"),
]

# Một số Kaggle session mount input dưới /kaggle/input/datasets/... thay vì trực tiếp
# /kaggle/input/<dataset-slug>. Vì vậy quét thêm bên dưới /kaggle/input.
data_candidates += [p.parent for p in INPUT_ROOT.rglob("train_labels.txt")]
code_candidates += [p.parent for p in INPUT_ROOT.rglob("train_best.py")]

DATA_ROOT = next((p for p in data_candidates if (p / "train_labels.txt").exists()), None)
CODE_ROOT = next((p for p in code_candidates if (p / "train_best.py").exists()), None)
WORK_CODE = Path("/kaggle/working/kaggle_scratch_ocr")

print("Detected DATA_ROOT:", DATA_ROOT)
print("Detected CODE_ROOT:", CODE_ROOT)

if DATA_ROOT is None:
    raise FileNotFoundError("Không tìm thấy train_labels.txt trong dataset ảnh. Kiểm tra lại Add Data.")

if CODE_ROOT is None:
    raise FileNotFoundError("Không tìm thấy train_best.py trong dataset code. Kiểm tra lại Add Data.")

if WORK_CODE.exists():
    shutil.rmtree(WORK_CODE)
shutil.copytree(CODE_ROOT, WORK_CODE)
print("Copied code to:", WORK_CODE)

print("\nData root:")
for p in DATA_ROOT.iterdir():
    print(" ", p)

print("\nCode files:")
for p in WORK_CODE.iterdir():
    print(" ", p)
```

Từ sau cell này, mọi lệnh sẽ dùng code tại:

```text
/kaggle/working/kaggle_scratch_ocr
```

Cách này tránh lỗi do Kaggle có thể đặt file code trực tiếp ở dataset root hoặc
trong thư mục con.

## 7. Cell 3: chuẩn bị dữ liệu

```python
!python /kaggle/working/kaggle_scratch_ocr/prepare_dataset.py \
  --data-root "{DATA_ROOT}" \
  --out-root /kaggle/working/ocr_prepared
```

Kết quả mong muốn có dạng:

```text
train: 9752
val: 1219
test: 1219
charset:  0123456789ABCDEFGHIJKLMNOPQRSTUVXYZĐ
```

## 7.1. Cell 3b: ready check trước khi train

Cell này kiểm tra data, charset, model forward và decoder ký tự dễ nhầm.

```python
!python /kaggle/working/kaggle_scratch_ocr/check_ready.py \
  --data-root /kaggle/working/ocr_prepared \
  --multi-sizes 40x160,48x192,56x224 \
  --img-h 48 \
  --img-w 192
```

Kết quả cuối phải có:

```text
READY_CHECK_PASS
```

Nếu chưa PASS, không chạy train full.

## 8. Cell 4: xem nhanh dữ liệu đã chuẩn bị

```python
import pandas as pd
from pathlib import Path

PREPARED = Path("/kaggle/working/ocr_prepared")

train_df = pd.read_csv(PREPARED / "train.csv")
val_df = pd.read_csv(PREPARED / "val.csv")

display(train_df.head())
print("train:", len(train_df))
print("val:", len(val_df))
print("charset:", (PREPARED / "charset.txt").read_text(encoding="utf-8"))
```

## 9. Cell 5: sanity run 3 epoch

Chạy nhanh để kiểm tra toàn bộ pipeline tạo checkpoint/log/output.

```python
!python /kaggle/working/kaggle_scratch_ocr/train_best.py \
  --config /kaggle/working/kaggle_scratch_ocr/configs/t4x2_safe.yaml \
  --data-root /kaggle/working/ocr_prepared \
  --out-dir /kaggle/working/sanity_plate_ocr_t4x2 \
  --epochs 3 \
  --train-limit 512 \
  --val-limit 256 \
  --batch-size 64
```

Kiểm tra output:

```python
import pandas as pd
from pathlib import Path

SANITY = Path("/kaggle/working/sanity_plate_ocr_t4x2")
display(pd.read_csv(SANITY / "history.csv"))
print("Files:", sorted(p.name for p in SANITY.iterdir()))
```

## 10. Cell 6: medium run 30 epoch

Chạy bước này để xem loss/accuracy có cải thiện trước khi chạy full.

```python
!python /kaggle/working/kaggle_scratch_ocr/train_best.py \
  --config /kaggle/working/kaggle_scratch_ocr/configs/t4x2_safe.yaml \
  --data-root /kaggle/working/ocr_prepared \
  --out-dir /kaggle/working/medium_plate_ocr_t4x2 \
  --epochs 30 \
  --batch-size 128
```

Xem kết quả medium run:

```python
import pandas as pd
from pathlib import Path

MEDIUM = Path("/kaggle/working/medium_plate_ocr_t4x2")
history = pd.read_csv(MEDIUM / "history.csv")
display(history.tail(10))

pred = pd.read_csv(MEDIUM / "predictions_val.csv")
display(pred.head(30))
print("Raw accuracy:", pred["correct"].mean())
print("Rule accuracy:", pred["rule_correct"].mean())

err = MEDIUM / "error_report.csv"
if err.exists():
    display(pd.read_csv(err).head(30))
```

## 11. Cell 7: full run trên Kaggle T4 x2

Cấu hình ổn định:

```python
!python /kaggle/working/kaggle_scratch_ocr/train_best.py \
  --config /kaggle/working/kaggle_scratch_ocr/configs/t4x2_safe.yaml \
  --data-root /kaggle/working/ocr_prepared \
  --out-dir /kaggle/working/scratch_plate_ocr_t4x2
```

Nếu bị CUDA out of memory, thêm override:

```text
--batch-size 96
```

Ví dụ:

```python
!python /kaggle/working/kaggle_scratch_ocr/train_best.py \
  --config /kaggle/working/kaggle_scratch_ocr/configs/t4x2_safe.yaml \
  --data-root /kaggle/working/ocr_prepared \
  --out-dir /kaggle/working/scratch_plate_ocr_t4x2 \
  --batch-size 96
```

Nếu vẫn lỗi, dùng:

```text
--batch-size 64
```

## 12. Cell 8: xem log training

```python
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/kaggle/working/scratch_plate_ocr_t4x2")

history = pd.read_csv(OUT_DIR / "history.csv")
display(history.tail(20))
```

## 13. Cell 9: xem prediction validation

```python
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/kaggle/working/scratch_plate_ocr_t4x2")

pred = pd.read_csv(OUT_DIR / "predictions_val.csv")
display(pred.head(30))

print("Raw accuracy:", pred["correct"].mean())
if "rule_correct" in pred.columns:
    print("Rule accuracy:", pred["rule_correct"].mean())

err = OUT_DIR / "error_report.csv"
if err.exists():
    display(pd.read_csv(err).head(30))
```

## 13.1. Cell 9b: phân tích ký tự dễ nhầm

Cell này giúp xem model đang nhầm những cặp ký tự nào nhiều nhất trên validation.

```python
from collections import Counter
import pandas as pd
from pathlib import Path

OUT_DIR = Path("/kaggle/working/scratch_plate_ocr_t4x2")
pred = pd.read_csv(OUT_DIR / "predictions_val.csv")

def levenshtein_ops(a, b):
    # trả về các cặp thay thế ký tự để phân tích lỗi OCR
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"
    bt[0][0] = None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            choices = [
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
                (dp[i - 1][j - 1] + cost, "eq" if cost == 0 else "sub"),
            ]
            dp[i][j], bt[i][j] = min(choices, key=lambda x: x[0])
    i, j = n, m
    subs = []
    while i > 0 or j > 0:
        op = bt[i][j]
        if op in ("eq", "sub"):
            if op == "sub":
                subs.append((a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif op == "del":
            i -= 1
        elif op == "ins":
            j -= 1
    return subs[::-1]

counter = Counter()
for _, row in pred.iterrows():
    label = str(row["label"])
    guess = str(row.get("rule_prediction", row["prediction"]))
    if label != guess:
        counter.update(levenshtein_ops(guess, label))

confusions = pd.DataFrame(
    [(src, tgt, count) for (src, tgt), count in counter.most_common(50)],
    columns=["model_read", "should_be", "count"],
)
display(confusions)
```

## 13.2. Bảng ký tự dễ nhầm cần kiểm tra

Khi đọc `confusions`, nên so với bảng dưới đây.

Nhóm chữ dễ bị đọc thành số:

```text
O -> 0
Q -> 0
D -> 0
Đ -> 0   chỉ cân nhắc khi chắc chắn đang ở vùng số
I -> 1
L -> 1
T -> 1 hoặc 7 tùy font, chỉ dùng khi candidate hợp lệ
Z -> 2
S -> 5
G -> 6
B -> 8
A -> 4   rất hạn chế, chỉ dùng khi candidate hợp lệ
```

Nhóm số dễ bị đọc thành chữ:

```text
0 -> O
0 -> D
0 -> Q
1 -> I
1 -> L
1 -> T
2 -> Z
4 -> A
5 -> S
6 -> G
7 -> T
8 -> B
```

Nhóm chữ dễ nhầm với chữ:

```text
D <-> Đ
O <-> Q
C <-> G
H <-> N
M <-> N
K <-> X
U <-> V
Y <-> V
P <-> R
E <-> F
F <-> P
T <-> I
L <-> I
```

Nhóm số dễ nhầm với số:

```text
0 <-> 8
1 <-> 7
2 <-> 7
3 <-> 8
5 <-> 6
6 <-> 8
8 <-> 9
```

Lưu ý quan trọng:

```text
Không sửa ký tự mù quáng trên toàn chuỗi.
Chỉ sửa theo vùng: mã tỉnh/thành, cụm chữ/loại xe, cụm số cuối.
```

## 14. Cell 10: inference thử một ảnh

```python
!python /kaggle/working/kaggle_scratch_ocr/infer.py \
  --checkpoint /kaggle/working/scratch_plate_ocr_t4x2/best.pt \
  --image /kaggle/input/vietnam-plate-ocr-data/ocr_dataset/test/car_90.jpg
```

## 15. Cell 11: nén output để tải về

Chạy cell này sau khi train xong:

```python
!cd /kaggle/working && zip -r scratch_plate_ocr_t4x2.zip scratch_plate_ocr_t4x2
```

File zip sẽ nằm ở:

```text
/kaggle/working/scratch_plate_ocr_t4x2.zip
```

## 16. Cell 12: tạo link tải trực tiếp trong notebook

```python
from IPython.display import FileLink, display

display(FileLink("/kaggle/working/scratch_plate_ocr_t4x2.zip"))
```

Sau khi chạy, Kaggle sẽ hiện một link để bấm tải file zip về máy.

## 17. Cell 13: tải riêng từng file quan trọng

Nếu không muốn tải cả zip, tạo link cho từng file:

```python
from IPython.display import FileLink, display
from pathlib import Path

OUT_DIR = Path("/kaggle/working/scratch_plate_ocr_t4x2")

for name in [
    "best.pt",
    "best_ema.pt",
    "last.pt",
    "charset.txt",
    "history.csv",
    "predictions_val.csv",
    "error_report.csv",
    "hard_examples.csv",
]:
    path = OUT_DIR / name
    if path.exists():
        display(FileLink(str(path)))
```

## 18. Cell 14: copy kết quả ra `/kaggle/working/output`

Nếu muốn gom kết quả vào một folder riêng:

```python
from pathlib import Path
import shutil

SRC = Path("/kaggle/working/scratch_plate_ocr_t4x2")
DST = Path("/kaggle/working/output")
DST.mkdir(exist_ok=True)

for name in [
    "best.pt",
    "best_ema.pt",
    "last.pt",
    "charset.txt",
    "history.csv",
    "predictions_val.csv",
    "error_report.csv",
    "hard_examples.csv",
]:
    src = SRC / name
    if src.exists():
        shutil.copy2(src, DST / name)

!cd /kaggle/working && zip -r output.zip output
```

Tạo link tải:

```python
from IPython.display import FileLink, display
display(FileLink("/kaggle/working/output.zip"))
```

## 19. Resume nếu notebook bị ngắt

Nếu Kaggle bị ngắt giữa chừng nhưng folder `/kaggle/working/scratch_plate_ocr_t4x2`
vẫn còn trong phiên hiện tại, có thể resume từ `last.pt`:

```python
!python /kaggle/working/kaggle_scratch_ocr/train_best.py \
  --config /kaggle/working/kaggle_scratch_ocr/configs/t4x2_safe.yaml \
  --data-root /kaggle/working/ocr_prepared \
  --out-dir /kaggle/working/scratch_plate_ocr_t4x2 \
  --resume /kaggle/working/scratch_plate_ocr_t4x2/last.pt
```

`--resume` chỉ dùng để tiếp tục chính run scratch của bạn, không dùng checkpoint
bên ngoài.

## 20. Những file cần giữ lại

Sau khi train, tối thiểu cần tải về:

```text
best.pt
best_ema.pt
charset.txt
history.csv
predictions_val.csv
error_report.csv
hard_examples.csv
```

Nên tải thêm:

```text
last.pt
```

để có thể resume hoặc phân tích sau.
