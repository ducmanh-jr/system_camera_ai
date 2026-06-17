# OCR biển số Việt Nam train từ đầu trên Kaggle T4 x2

Thư mục này có 2 tài liệu chính:

1. [01_Y_TUONG_CHINH.md](01_Y_TUONG_CHINH.md)  
   Chốt phương pháp tốt nhất trong giới hạn: **Kaggle T4 x2 + data hiện tại + train từ đầu 100%**.

2. [02_HUONG_DAN_KAGGLE_IPYNB.md](02_HUONG_DAN_KAGGLE_IPYNB.md)  
   Hướng dẫn upload dataset/code lên Kaggle, chạy notebook `.ipynb`, train, inference và tải kết quả về bằng code.

Notebook mẫu:

- [kaggle_train_t4x2.ipynb](kaggle_train_t4x2.ipynb)

Script hiện có:

- `prepare_dataset.py`: chuẩn hóa label, tạo CSV và charset.
- `train_best.py`: train OCR từ đầu theo pipeline tối ưu T4 x2.
- `train_scratch.py`: bản train cũ, giữ lại để dự phòng.
- `infer.py`: chạy inference từ checkpoint.

Config:

- `configs/t4x2_safe.yaml`: cấu hình ổn định cho Kaggle T4 x2.
- `configs/t4x2_strong.yaml`: cấu hình mạnh hơn nếu còn VRAM/thời gian.

Nguyên tắc bắt buộc:

- Không nạp checkpoint bên ngoài.
- Không dùng pretrained OCR.
- Không dùng ImageNet pretrained.
- Không fine-tune.
- Chỉ dùng `--resume` để tiếp tục chính run scratch của bạn.
