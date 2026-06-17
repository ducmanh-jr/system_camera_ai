# Báo Cáo Nghiên Cứu Kỹ Thuật: Hệ Thống Giám Sát Xe Và Trích Xuất Vùng Biển Số Tối Ưu Bằng YOLOv8 Và ByteTrack

**Tác giả:** [Tên của bạn]  
**Đơn vị:** [Tên trường/khoa]  
**Lĩnh vực:** Thị giác máy tính (Computer Vision) & Xử lý ảnh (Image Processing)  
**Tệp thực thi chính:** [main.py](./main.py)

---

## 1. Tóm Tắt (Abstract)
Báo cáo này trình bày giải pháp phát hiện, theo dõi phương tiện giao thông và tự động trích xuất vùng ảnh biển số xe có chất lượng cao nhất từ luồng video. Bằng cách kết hợp mô hình học sâu **YOLOv8** để phát hiện đối tượng, giải thuật **ByteTrack** để bám vết, và một hàm đánh giá đa tiêu chí (độ nét Laplacian, độ tương phản, kích thước hình học), hệ thống đảm bảo trích xuất một ảnh biển số tối ưu duy nhất cho mỗi phương tiện. Đồng thời, cơ chế khử trùng lặp dựa trên **độ tương đồng đặc trưng Cosine (Cosine Similarity)** giúp loại bỏ các bản ghi trùng lặp khi xảy ra lỗi đứt mạch tracking. Hệ thống đạt độ ổn định cao, tối ưu hóa tài nguyên tính toán nhờ giới hạn vùng quan tâm (ROI).

---

## 2. Kiến Trúc Hệ Thống (System Pipeline)
Quy trình xử lý tuần tự của hệ thống đối với luồng video đầu vào được mô tả qua sơ đồ khối dưới đây:

```text
                                Video Đầu Vào
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │      YOLOv8 + ByteTrack       │
                      │  (Phát hiện & bám vết xe)     │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                            Tâm xe nằm trong ROI?
                            ├── [Không] ──► Bỏ qua frame
                            └── [Có]
                                  │
                                  ▼
                      ┌───────────────────────────────┐
                      │        Cắt ảnh xe sạch        │
                      │  (Trích xuất từ frame gốc)    │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Plate Detector (YOLOv8)    │
                      │   (Phát hiện biển số trong xe)│
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                             Tìm thấy biển số?
                             ├── [Không] ──► Bỏ qua frame
                             └── [Có]
                                   │
                                   ▼
                      ┌───────────────────────────────┐
                      │   Đánh giá điểm chất lượng    │
                      │  (Sharpness, Contrast, Size)  │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ Cập nhật ảnh tốt nhất mỗi xe │
                      │  (Lưu tạm thời theo Track ID) │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                                Kết thúc video?
                                ├── [Không] ──► Tiếp tục đọc frame
                                └── [Có]
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Khử trùng lặp biển số      │
                      │  (So sánh Cosine Similarity)  │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                            Xuất Kết Quả Đầu Ra
                       (SQLite DB, CSV, Review Images)
```

---

## 3. Cơ Sở Lý Thuyết & Chi Tiết Giải Thuật

### 3.1. Phát Hiện & Bám Vết Phương Tiện (Object Detection & Tracking)
Hệ thống sử dụng mô hình YOLOv8 (`yolov8m.pt`) kết hợp với giải thuật **ByteTrack** để thực hiện bám vết phương tiện.
*   **YOLOv8:** Đóng vai trò nhận diện các lớp phương tiện tại dòng [main.py: Dòng 848](./main.py#L848).
*   **ByteTrack:** Khác biệt với các bộ bám vết thông thường (chỉ giữ các đối tượng có điểm tin cậy cao), ByteTrack tận dụng cả các hộp bao có điểm tin cậy thấp nhưng có độ tương đồng IOU cao với vết cũ. Điều này giúp giảm thiểu việc mất dấu vết khi phương tiện bị che khuất một phần hoặc mờ nhòe.

### 3.2. Giới Hạn Vùng Quan Tâm (Region of Interest - ROI)
Để tăng hiệu năng hệ thống (FPS) và loại bỏ các phương tiện ở quá xa camera, hệ thống giới hạn vùng xử lý thông qua hàm kiểm tra tọa độ tại [main.py: Dòng 416](./main.py#L416) và áp dụng trong vòng lặp chính tại [main.py: Dòng 878](./main.py#L878). Chỉ những xe có tâm nằm trong ROI mới được đưa vào phân tích biển số.

### 3.3. Đánh Giá Chất Lượng Vùng Biển Số (Multi-criteria Quality Scoring)
Mỗi ứng viên biển số tìm thấy sẽ được chấm điểm chất lượng tự động thông qua hàm [main.py: Dòng 229](./main.py#L229). Điểm số này ($S$) được tính toán dựa trên tổ hợp tuyến tính của 5 thành phần sau:

$$S = 0.35 \cdot C + 0.25 \cdot S_{sharp} + 0.20 \cdot S_{size} + 0.10 \cdot S_{area\_ratio} + 0.10 \cdot S_{contrast}$$

Trong đó:
1.  **Confidence ($C$):** Độ tin cậy trả về từ mô hình phát hiện biển số [main.py: Dòng 312](./main.py#L312).
2.  **Sharpness Score ($S_{sharp}$):** Tính toán độ sắc nét bằng phương sai của toán tử Laplacian [main.py: Dòng 215](./main.py#L215). Công thức tính Laplacian của ảnh xám $I$:
    $$\Delta I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$
    $$S_{sharp} = \min\left(\frac{\text{Var}(\Delta I)}{\text{SHARPNESS\_NORMALIZER}}, 1.0\right)$$
3.  **Size Score ($S_{size}$):** Đánh giá kích thước vật lý của biển số [main.py: Dòng 239](./main.py#L239). Ưu tiên các ảnh biển số có kích thước rộng và cao lớn để đảm bảo độ phân giải ký tự tốt.
4.  **Contrast Score ($S_{contrast}$):** Độ tương phản đo bằng độ lệch chuẩn của thang xám [main.py: Dòng 222](./main.py#L222).
5.  **Area Ratio Score ($S_{area\_ratio}$):** Tỉ lệ diện tích biển số chiếm trên diện tích xe [main.py: Dòng 237](./main.py#L237).

### 3.4. Khử Trùng Lặp Ảnh Biển Số (De-duplication)
Khi quá trình tracking bị lỗi (ví dụ phương tiện bị xe khác che khuất tạm thời), ByteTrack sẽ gán một ID mới cho phương tiện đó khi nó xuất hiện trở lại. Để tránh việc ghi nhận một xe thành nhiều vi phạm khác nhau, hệ thống thực hiện thuật toán khử trùng lặp [main.py: Dòng 612](./main.py#L612) ở cuối video:
1.  **Trích xuất đặc trưng ảnh (Signature Extraction):** Chuyển ảnh biển số về ảnh xám, resize về kích thước chuẩn $96 \times 32$, thực hiện cân bằng lược đồ xám (Histogram Equalization) để triệt tiêu ảnh hưởng ánh sáng, sau đó chuẩn hóa vector L2 [main.py: Dòng 579](./main.py#L579).
2.  **Độ tương đồng Cosine (Cosine Similarity):** Tính góc giữa hai vector đặc trưng $\mathbf{A}$ và $\mathbf{B}$:
    $$\text{Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
3.  **Kết hợp hình học và thời gian:** Ngoài Cosine Similarity, hệ thống kết hợp thêm khoảng cách vị trí pixel giữa hai biển số [main.py: Dòng 599](./main.py#L599) và khoảng cách số khung hình xuất hiện (Frame Gap) để đưa ra quyết định gộp vết chính xác.

---

## 4. Thiết Lập Tham Số Thực Nghiệm (Hyperparameters)
Hệ thống cung cấp khả năng tinh chỉnh linh hoạt thông qua dòng lệnh [main.py: Dòng 984](./main.py#L984). Các tham số thực nghiệm mặc định tối ưu bao gồm:

| Tham số | Giá trị | Mô tả | Vị trí trong code |
| :--- | :--- | :--- | :--- |
| `DEFAULT_ROI` | `"0.0,0.40,1.0,1.0"` | Vùng xử lý (nửa dưới khung hình) | [main.py: Dòng 60](./main.py#L60) |
| `DEFAULT_PLATE_CONF` | `0.25` | Ngưỡng phát hiện biển số | [main.py: Dòng 57](./main.py#L57) |
| `DEFAULT_MIN_PLATE_SCORE` | `0.46` | Ngưỡng chất lượng tối thiểu để ghi nhận | [main.py: Dòng 72](./main.py#L72) |
| `DEFAULT_MIN_PLATE_SHARPNESS` | `1800.0` | Ngưỡng lọc ảnh mờ (Laplacian Variance) | [main.py: Dòng 71](./main.py#L71) |
| `DEFAULT_DUPLICATE_PLATE_SIMILARITY` | `0.70` | Ngưỡng gộp biển trùng (Cosine Similarity) | [main.py: Dòng 74](./main.py#L74) |

---

## 5. Kết Quả Lưu Trữ (Data Storage & Output Structure)
Dữ liệu kết xuất sau khi kết thúc quá trình quét video được tổ chức khoa học:

1.  **SQLite Database (`outputs/violations.db`):** Lưu trữ quan hệ phục vụ truy vấn [main.py: Dòng 138](./main.py#L138).
2.  **File báo cáo Excel (`outputs/violators.csv`):** Thống kê chi tiết các vi phạm gồm ID, loại xe, thời gian, điểm chất lượng và đường dẫn ảnh [main.py: Dòng 718](./main.py#L718).
3.  **Ảnh Review (`outputs/reviews/`):** Ảnh canvas ghép tự động [main.py: Dòng 485](./main.py#L485) bao gồm ảnh xe rộng bên trái và ảnh cận cảnh biển số cùng thông số kỹ thuật bên phải phục vụ giám sát thủ công.

---

## 6. Hướng Phát Triển Tiếp Theo (Future Work)
1.  **Nhận diện ký tự (OCR):** Tích hợp thêm các bộ thư viện OCR chuyên dụng như EasyOCR hoặc PaddleOCR trên ảnh biển số chất lượng cao đã trích xuất được từ giai đoạn này.
2.  **Tối ưu hóa phần cứng:** Chuyển đổi mô hình YOLOv8 sang định dạng TensorRT hoặc ONNX để tăng tốc độ xử lý thời gian thực trên các thiết bị nhúng.
