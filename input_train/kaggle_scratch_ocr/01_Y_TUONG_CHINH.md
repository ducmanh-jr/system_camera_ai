# Ý tưởng chính: OCR biển số Việt Nam train từ đầu trên Kaggle T4 x2

Tài liệu này chốt phương pháp tốt nhất cho bài toán OCR biển số Việt Nam trong
3 giới hạn bắt buộc:

```text
GPU: Kaggle T4 x2
Dữ liệu: chỉ dùng bộ ocr_dataset hiện tại
Training: train từ đầu 100%, không fine-tune, không pretrained
```

Mục tiêu là đạt chất lượng tốt nhất có thể trong giới hạn trên. Vì vậy, hướng
được chọn không phải mô hình rất lớn cần pretrained hoặc data khổng lồ, mà là
một pipeline OCR hiện đại, gọn, ổn định và phù hợp với biển số.

## Kết luận phương pháp tối ưu

Pipeline được chốt:

```text
SVTRv2-inspired CTC recognizer
+ synthetic biển số Việt chất lượng cao
+ curriculum training
+ multi-size/aspect-ratio bucket
+ semantic auxiliary loss
+ EMA checkpoint
+ hard example mining
+ constrained decoder theo luật biển số Việt Nam
+ multi-seed ensemble nếu cần tối đa hóa accuracy
```

Lý do chọn hướng này:

- Biển số là chuỗi ngắn, charset nhỏ, format rõ; CTC rất phù hợp.
- SVTR/SVTRv2-style recognizer là hướng hiện đại cho scene text recognition.
- Train từ đầu trên data hiện tại không phù hợp với TrOCR, PARSeq lớn,
  VLM/OCR foundation model hoặc PaddleOCR pretrained, vì các hướng đó thường
  cần pretrained/data rất lớn.
- Synthetic data và curriculum training giúp bù lại giới hạn số lượng ảnh thật.
- Decoder theo luật biển số Việt Nam tận dụng cấu trúc bài toán để giảm lỗi
  chữ/số phổ biến.

## Những thứ không chọn làm hướng chính

Không chọn các hướng sau làm thành phần bắt buộc:

```text
EfficientNet pretrained
Hybrid CTC-Attention làm head chính ngay
BiLSTM bắt buộc
Hydra bắt buộc
WER làm metric chính
```

Giải thích:

- `EfficientNet pretrained` vi phạm yêu cầu train từ đầu. EfficientNet random
  init có thể thử như ablation, nhưng không phải lựa chọn chính cho OCR chuỗi.
- `Hybrid CTC-Attention` dễ overfit khi data thật ít. CTC nên là head chính;
  phần ngữ cảnh nên đưa vào semantic auxiliary head hoặc decoder.
- `BiLSTM` không còn bắt buộc khi đã có Transformer encoder.
- `Hydra` mạnh nhưng nặng cho Kaggle workflow; YAML config đơn giản dễ chạy và
  dễ debug hơn.
- `WER` hợp văn bản dài, không hợp làm metric chính cho biển số. Exact match và
  CER quan trọng hơn.

## Tổng quan pipeline

```text
Ảnh + label hiện tại
 -> chuẩn hóa label và charset
 -> sinh synthetic biển số Việt
 -> trộn real + synthetic theo curriculum
 -> resize theo multi-size/aspect-ratio bucket
 -> CNN stem
 -> local feature mixing kiểu SVTR/SVTRv2
 -> Transformer encoder
 -> CTC head
 -> semantic auxiliary head khi training
 -> EMA checkpoint
 -> hard example mining
 -> constrained Vietnamese plate decoder
 -> chọn checkpoint tốt nhất hoặc ensemble nhiều seed
```

## 1. Chuẩn hóa dữ liệu

Input:

```text
ocr_dataset/
  train/
  val/
  test/
  train_labels.txt
  val_labels.txt
  test_labels.txt
```

Output sau chuẩn hóa:

```text
train.csv
val.csv
test.csv
charset.txt
```

Yêu cầu:

- Sửa lỗi encoding cho ký tự `Đ`.
- Giữ khoảng trắng trong biển số.
- Kiểm tra ảnh thiếu, label rỗng, ký tự ngoài charset.
- Không thêm synthetic vào validation/test.
- Báo cáo số lượng ảnh train/val/test và charset cuối cùng.

## 2. Synthetic biển số Việt chất lượng cao

Vì không có dataset biển số Việt Nam lớn hơn, synthetic data là bắt buộc nếu
train từ đầu.

Synthetic engine cần bao phủ:

- Biển ô tô một dòng.
- Biển xe máy hai dòng.
- Biển đặc biệt: `LD`, `MĐ`, `TĐ`, `HC`, `NG`, `QT`, `AB`.
- Biển 4 hoặc 5 số cuối.
- Mã tỉnh/thành Việt Nam hợp lệ.
- Ký tự `Đ` đúng encoding.

Hiệu ứng ảnh cần mô phỏng:

- Nền phản quang nhẹ.
- Viền biển, đinh vít, vết bẩn nhỏ.
- Motion blur và out-of-focus blur.
- Nén JPEG.
- Thiếu sáng, lóa nhẹ, thay đổi tương phản.
- Crop lệch, nghiêng, phối cảnh nhẹ.
- Che khuất nhẹ hoặc mờ một phần ký tự.

Nguyên tắc:

- Synthetic dùng để tăng độ đa dạng, không thay thế ảnh thật.
- Synthetic không được quá sạch hoặc quá giả.
- Tỉ lệ synthetic mục tiêu: `4-5x` số ảnh thật.

## 3. Curriculum training

Training chia theo giai đoạn:

```text
epoch 1-40:
  synthetic sạch + real augmentation nhẹ
  mục tiêu: học charset, khoảng trắng, cấu trúc biển số

epoch 41-160:
  synthetic khó hơn + real augmentation vừa
  mục tiêu: chống blur, lệch, ánh sáng xấu, crop chưa chuẩn

epoch 161-280:
  giảm synthetic quá dễ, tăng trọng số ảnh thật
  mục tiêu: tối ưu trên phân phối data thật

epoch 281-340 nếu còn thời gian:
  hard-example phase
  mục tiêu: sửa lỗi 0/O, 1/I, 8/B, mất khoảng trắng, sai Đ
```

Đây vẫn là cùng một quá trình train từ đầu, không phải fine-tune từ pretrained.

## 4. Kiến trúc model

Kiến trúc mục tiêu:

```text
Image
 -> CNN stem
 -> Local feature mixing
 -> Transformer encoder
 -> CTC head
 -> Auxiliary semantic head trong training
 -> OCR text
```

Vai trò:

- `CNN stem`: học nét chữ, cạnh ký tự, texture và viền biển.
- `Local feature mixing`: xử lý biến dạng cục bộ, nghiêng, kéo dãn, crop lệch.
- `Transformer encoder`: học quan hệ giữa các ký tự.
- `CTC head`: head chính để nhận dạng chuỗi.
- `Auxiliary semantic head`: head phụ khi training để tăng khả năng học ngữ cảnh.

## 5. Multi-size/aspect-ratio bucket

Dùng nhiều kích thước ảnh để giảm méo ký tự:

```text
40x160
48x192
56x224
64x256
```

Cấu hình khuyến nghị:

- An toàn trên T4 x2: `40x160,48x192,56x224`.
- Mạnh hơn nếu còn VRAM: thêm `64x256`, giảm batch size nếu cần.

## 6. Training setup

Loss:

```text
CTC loss chính
semantic auxiliary CE/BCE loss phụ
```

Regularization và tối ưu:

- AdamW.
- Warmup + cosine scheduler.
- AMP.
- EMA `0.999-0.9993`.
- Dropout `0.10-0.18`.
- Weight decay `0.05`.
- Gradient clipping.

Không dùng label noise tùy tiện vì OCR biển số cần chính xác tuyệt đối.

## 7. Decoder theo luật biển số Việt Nam

Constrained decoder chuẩn hóa kết quả theo cấu trúc biển số Việt Nam:

- Hai ký tự đầu thường là số tỉnh/thành.
- Phần giữa là chữ hoặc cụm đặc biệt.
- Phần cuối là 4-5 số.
- Tự khôi phục khoảng trắng trước cụm số cuối.
- Sửa nhầm lẫn theo vị trí:
  - vùng số: `O/Q/D -> 0`, `I/L -> 1`, `B -> 8`, `S -> 5`
  - vùng chữ: chỉ sửa khi format yêu cầu chữ

Bản tốt hơn nên dùng confidence từ CTC để chọn candidate hợp lệ có xác suất cao
nhất, thay vì chỉ dùng heuristic cứng.

### Bảng ký tự dễ nhầm cần xử lý

Không được sửa ký tự một cách mù quáng trên toàn chuỗi. Mọi sửa lỗi phải dựa
theo vùng của biển số:

```text
[mã tỉnh/thành số] [cụm chữ/loại xe] [cụm số cuối]
```

Ví dụ:

```text
59A1 00128
30F 78286
50TĐ 03027
60MĐ1 01604
```

#### Nhóm chữ dễ bị đọc thành số

Các ký tự này thường cần sửa khi chúng nằm ở vùng số:

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

#### Nhóm số dễ bị đọc thành chữ

Các ký tự này thường cần sửa khi chúng nằm ở vùng chữ:

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

#### Nhóm chữ dễ nhầm với chữ

Các cặp này không nên tự sửa cứng. Chỉ dùng để tạo candidate và chọn bằng
confidence/format:

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

#### Nhóm số dễ nhầm với số

Các cặp này cũng không nên sửa cứng. Chỉ dùng trong phân tích lỗi hoặc candidate
ranking:

```text
0 <-> 8
1 <-> 7
2 <-> 7
3 <-> 8
5 <-> 6
6 <-> 8
8 <-> 9
```

### Nguyên tắc áp dụng ký tự dễ nhầm

Decoder nên tạo nhiều candidate hợp lệ rồi chọn candidate có điểm cao nhất:

```text
raw prediction
 -> tách vùng theo format biển số Việt Nam
 -> sinh candidate bằng bảng ký tự dễ nhầm
 -> loại candidate sai format
 -> chấm điểm bằng CTC confidence / edit distance / luật biển số
 -> chọn candidate tốt nhất
```

Không nên áp dụng kiểu:

```text
thấy O là đổi thành 0 ở mọi vị trí
thấy 1 là đổi thành I ở mọi vị trí
```

vì sẽ làm hỏng các biển đúng như:

```text
30F 78286
59A1 00128
60MĐ1 01604
```

### Vùng ưu tiên sửa lỗi

Vùng mã tỉnh/thành:

```text
chỉ nên là số
ưu tiên sửa O/Q/D/Đ/I/L/Z/S/G/B/A/T về số nếu tạo được mã hợp lệ
```

Vùng cụm chữ/loại xe:

```text
có thể là chữ, chữ + số, hoặc cụm đặc biệt
ưu tiên sửa 0/1/2/4/5/6/7/8 về chữ nếu format yêu cầu
```

Vùng số cuối:

```text
chỉ nên là 4 hoặc 5 chữ số
ưu tiên sửa O/Q/D/Đ/I/L/Z/S/G/B/A/T về số
```

Vùng ký tự đặc biệt:

```text
LD, MĐ, TĐ, HC, NG, QT, AB
```

Các cụm này cần được bảo vệ bằng format rule để tránh sửa nhầm `Đ` thành `0`
hoặc `D`.

## 8. Metric đánh giá

Metric chính:

```text
Exact match accuracy raw
Exact match accuracy sau rule decoder
CER
```

Metric phụ:

```text
Accuracy riêng cho biển có Đ
Accuracy riêng cho biển hai dòng
Top lỗi thường gặp
Danh sách prediction sai để kiểm tra thủ công
```

Không dùng WER làm metric chính vì biển số không phải văn bản dài nhiều từ.

## 9. Multi-seed ensemble

Để tối đa hóa accuracy trên T4 x2, chạy nhiều seed:

```text
seed 42
seed 123
seed 777
```

Sau đó chọn checkpoint tốt nhất hoặc ensemble bằng majority vote/confidence vote.

## 10. Cấu hình chốt

Bản an toàn:

```text
epochs: 280
batch-size: 128
synthetic-ratio: 4
sizes: 40x160,48x192,56x224
model-dim: 256
layers: 8
heads: 8
EMA: 0.999
AMP: bật
```

Bản mạnh hơn:

```text
epochs: 340
batch-size: 96
synthetic-ratio: 5
sizes: 40x160,48x192,56x224,64x256
model-dim: 320
layers: 10
heads: 8
EMA: 0.9992
AMP: bật
```

Tóm tắt một dòng:

```text
SVTRv2-inspired CTC + synthetic biển số Việt mạnh + curriculum + Albumentations
+ EMA + hard mining + constrained decoder + multi-seed
```
