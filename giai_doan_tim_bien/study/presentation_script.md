# Kịch Bản Thuyết Trình Dự Án (Thời lượng: ~3 đến 5 phút)

Kịch bản này giúp bạn trình bày dự án một cách tự tin, ngắn gọn nhưng đầy đủ các ý chính về mặt kỹ thuật trước khi bước vào phần vấn đáp. Khi bạn click vào các liên kết màu xanh dưới đây, biên dịch viên (Editor) sẽ tự động di chuyển đến dòng code tương ứng trong file [main.py](./main.py).

---

## 🎙️ PHẦN THUYẾT TRÌNH CHI TIẾT

### 1. Lời Mở Đầu & Giới Thiệu Đề Tài (~30 giây)
> **Nói:**
> "Em xin kính chào Thầy/Cô. Hôm nay em xin phép được trình bày dự án báo cáo của mình với đề tài: **'Nghiên cứu và phát hiện, theo dõi phương tiện giao thông và trích xuất tối ưu vùng ảnh biển số xe từ luồng video.'**
> 
> Dự án này đóng vai trò là **Giai đoạn 1** trong một hệ thống giám sát giao thông thông minh. Mục tiêu cốt lõi của giai đoạn này là phát hiện phương tiện, bám vết chúng và lọc ra bức ảnh vùng biển số xe có chất lượng cao nhất, rõ nét nhất từ video để làm đầu vào chuẩn hóa cho giai đoạn nhận diện ký tự (OCR) tiếp theo."
> 
> **Hành động (nếu có slide/màn hình):** Show slide tiêu đề hoặc màn hình mã nguồn dự án.

---

### 2. Đặt Vấn Đề & Giải Pháp Đề Xuất (~45 giây)
> **Nói:**
> "Trong thực tế, các hệ thống camera giám sát giao thông thường trả về hình ảnh xe ở nhiều cự ly và điều kiện ánh sáng khác nhau. Nếu chúng ta chạy nhận diện OCR liên tục trên mọi khung hình, hệ thống sẽ rất chậm và kết quả OCR sẽ bị nhiễu do ảnh xe ở xa bị mờ hoặc nhỏ.
> 
> Để giải quyết vấn đề này, giải pháp của em tập trung vào 3 điểm cải tiến chính:
> 1. **Chỉ quét vùng ROI (Region of Interest) gần camera:** Giúp tiết kiệm tài nguyên xử lý và đảm bảo thu được ảnh biển số lớn nhất.
>    * Cấu hình mặc định: [main.py: Dòng 60](./main.py#L60)
>    * Định nghĩa hàm kiểm tra: [main.py: Dòng 416](./main.py#L416)
> 2. **Xây dựng hàm chấm điểm chất lượng ảnh tự động:** Không chỉ tin vào độ tin cậy của AI, hệ thống của em còn đánh giá độ sắc nét vật lý (Sharpness), độ tương phản (Contrast) và kích thước pixel của biển số để tự động giữ lại bức ảnh tốt nhất.
>    * Hàm chấm điểm: [main.py: Dòng 229](./main.py#L229)
>    * Đo độ nét Laplacian: [main.py: Dòng 215](./main.py#L215)
>    * Đo độ tương phản: [main.py: Dòng 222](./main.py#L222)
>    * Lọc kích thước vật lý: [main.py: Dòng 249](./main.py#L249)
> 3. **Cơ chế khử trùng lặp (De-duplication):** Đảm bảo mỗi phương tiện chỉ xuất hiện duy nhất một lần trong danh sách ghi nhận, ngay cả khi quá trình bám vết (tracking) bị đứt quãng.
>    * Hàm khử trùng: [main.py: Dòng 612](./main.py#L612)
> 
> **Hành động:** Trỏ chuột vào định nghĩa đường dẫn model: [main.py: Dòng 27](./main.py#L27) và [main.py: Dòng 28](./main.py#L28)."

---

### 3. Quy Trình Hoạt ĐỘng Của Hệ Thống (~1 phút)
> **Sơ đồ quy trình (Pipeline Workflow Diagram):**
> 
> ```text
> ┌───────────────────┐
> │   Video đầu vào   │
> └─────────┬─────────┘
>           │
>           ▼
> ┌─────────┴─────────┐
> │ YOLOv8 + ByteTrack│ ──► (Bám vết và gán ID xe)
> └─────────┬─────────┘
>           │
>           ▼
> ┌─────────┴─────────┐
> │ Tâm xe trong ROI? │ ──► [Không] ──► (Bỏ qua frame này)
> └─────────┬─────────┘
>           │ [Có]
>           ▼
> ┌─────────┴─────────┐
> │  Cắt ảnh xe sạch  │ ──► (Cắt từ frame gốc không vẽ debug)
> └─────────┬─────────┘
>           │
>           ▼
> ┌─────────┴─────────┐
> │  Detect biển số   │
> └─────────┬─────────┘
>           │
>           ▼
> ┌─────────┴─────────┐
> │   Có biển số?     │ ──► [Không] ──► (Bỏ qua frame này)
> └─────────┬─────────┘
>           │ [Có]
>           ▼
> ┌─────────┴─────────┐
> │ Chấm điểm & Cập   │ ──► (Giữ ảnh biển số tốt nhất của xe)
> │ nhật ảnh tốt nhất │
> └─────────┬─────────┘
>           │
>           ▼
> ┌─────────┴─────────┐
> │   Khử trùng lặp   │ ──► (So sánh Cosine Similarity cuối video)
> └─────────┬─────────┘
>           │
>           ▼
> ┌─────────┴─────────┐
> │   Xuất kết quả    │ ──► (SQLite, CSV, Review Image, Video)
> └───────────────────┘
> ```
> 
> > **Nói:**
> "Quy trình xử lý của hệ thống đối với mỗi khung hình trong video diễn ra như sau:
> *   **Bước 1:** Sử dụng model YOLOv8 kết hợp thuật toán **ByteTrack** để phát hiện và gán ID bám vết cho từng phương tiện.
>     * Gọi hàm tracking: [main.py: Dòng 848](./main.py#L848)
> *   **Bước 2:** Hệ thống lọc bỏ các xe ở xa, chỉ giữ lại các xe có tâm nằm trong vùng ROI giám sát.
>     * Câu lệnh lọc: [main.py: Dòng 878](./main.py#L878)
> *   **Bước 3:** Cắt ảnh vùng xe từ frame gốc 'sạch' (không chứa nét vẽ debug) để làm đầu vào cho model phát hiện biển số thứ hai nằm bên trong thân xe.
>     * Sao chép ảnh sạch: [main.py: Dòng 846](./main.py#L846)
>     * Hàm cắt ảnh xe sạch: [main.py: Dòng 464](./main.py#L464)
>     * Chạy phát hiện biển số trên ảnh crop xe: [main.py: Dòng 281](./main.py#L281)
> *   **Bước 4:** Khi phát hiện thấy biển số, hệ thống sẽ tính điểm chất lượng dựa trên độ nét Laplacian, độ tương phản và kích thước. Mỗi ID xe sẽ liên tục cập nhật và chỉ lưu lại ảnh biển số có điểm chất lượng cao nhất xuyên suốt quá trình xe di chuyển.
>     * Cập nhật ảnh tốt nhất: [main.py: Dòng 904](./main.py#L904)
> *   **Bước 5:** Cuối cùng, khi video kết thúc, hệ thống chạy thuật toán khử trùng lặp bằng cách so sánh độ tương đồng đặc trưng ảnh (Cosine Similarity) giữa các xe để gộp các kết quả trùng lặp, đảm bảo tính chính xác cao nhất.
>     * Trích xuất vector đặc trưng: [main.py: Dòng 579](./main.py#L579)
>     * Đối sánh gom cụm: [main.py: Dòng 655](./main.py#L655)
> 
> **Hành động:** Di chuột qua các hàm toán học chính như [main.py: Dòng 229](./main.py#L229) và [main.py: Dòng 612](./main.py#L612) để thể hiện sự chuẩn bị kỹ càng."

---

### 4. Kết Quả Thực Nghiệm & Lưu Trữ (~1 phút)
> **Nói:**
> "Sau khi chạy thử nghiệm trên video giao thông thực tế, hệ thống đã trích xuất và xuất ra các kết quả rất trực quan trong thư mục `outputs/`:
> *   **Ảnh Review (`outputs/reviews/`):** Đây là ảnh ghép tự động giúp người vận hành dễ dàng quan sát, bao gồm ảnh xe rộng bên trái và ảnh cận cảnh biển số kèm bảng điểm chất lượng chi tiết bên phải.
>     * Hàm vẽ ảnh review: [main.py: Dòng 485](./main.py#L485)
> *   **Dữ liệu dạng bảng:** Toàn bộ lịch sử di chuyển và thông tin xe được lưu trữ có cấu trúc vào SQLite database (`violations.db`) và xuất ra các file Excel dạng CSV (`violators.csv`, `tracks.csv`) phục vụ báo cáo.
>     * Đường dẫn file DB: [main.py: Dòng 22](./main.py#L22)
>     * Khởi tạo SQLite DB: [main.py: Dòng 138](./main.py#L138)
>     * Ghi CSV kết quả: [main.py: Dòng 718](./main.py#L718)
>     * Ghi CSV theo vết tracking: [main.py: Dòng 756](./main.py#L756)
> *   **Video đầu ra (`output.mp4`):** Thể hiện trực quan toàn bộ quá trình bám vết và ghi nhận thời gian thực để phục vụ giám sát trực tiếp.
> 
> Bây giờ, em xin phép được chạy demo trực tiếp hệ thống để Thầy/Cô cùng quan sát."
> 
> **Hành động:** Mở Terminal và gõ lệnh chạy demo trực quan:
> `python main.py --show --speed 0.5`
> 
> *(Giải thích nhanh lúc video đang chạy)*: "Như Thầy/Cô có thể thấy trên màn hình, vùng ROI màu xanh cyan đang giám sát làn đường. Các xe đi qua được gán ID bám vết rất mượt mà. Khi xe chuyển sang khung màu đỏ, nghĩa là hệ thống đã trích xuất thành công biển số của xe đó với chất lượng cao nhất."

---

### 5. Lời Kết (~15 giây)
> **Nói:**
> "Dự án của em đã hoàn thành tốt giai đoạn tiền xử lý ảnh và bám vết đối tượng, tạo tiền đề vững chắc cho việc tích hợp bộ nhận diện ký tự OCR ở giai đoạn tiếp theo. 
> 
> Em xin chân thành cảm ơn Thầy/Cô đã lắng nghe phần trình bày của em. Sau đây, em xin phép được tiếp thu ý kiến đóng góp và trả lời các câu hỏi phản biện từ Thầy/Cô."

---

### 6. CÁC THUẬT NGỮ VÀ KHÁI NIỆM KỸ THUẬT TRONG DỰ ÁN
Để giúp bạn tự tin trả lời bất kỳ câu hỏi định nghĩa nào từ giáo viên, dưới đây là bảng tra cứu các công nghệ cốt lõi được sử dụng trong mã nguồn:

### 1. YOLOv8 (You Only Look Once version 8)
*   **Là gì?** Là một mạng thần kinh nhân tạo sâu (Deep Learning) thuộc nhóm SOTA (State-of-the-art) chuyên dùng để nhận diện đối tượng trong ảnh/video với tốc độ cực nhanh và độ chính xác cao.
*   **Làm gì trong dự án?** 
    *   Lần 1: Nhận diện và phân loại phương tiện giao thông (Car, Truck, Bus, Motorcycle) từ khung hình video: [main.py: Dòng 848](./main.py#L848) (sử dụng model [yolov8m.pt](./models/yolov8m.pt)).
    *   Lần 2: Nhận diện vùng biển số xe nằm bên trong ảnh cắt phương tiện: [main.py: Dòng 281](./main.py#L281) (sử dụng model [license_plate_detector.pt](./models/license_plate_detector.pt)).

### 2. ByteTrack (Multi-Object Tracker)
*   **Là gì?** Là thuật toán theo dõi đa đối tượng (Multi-Object Tracking) trong video. Nó giúp liên kết các hộp nhận diện (bounding box) của cùng một đối tượng qua các frame liên tiếp để tạo thành một đường đi (trajectory).
*   **Làm gì trong dự án?** Bám vết các xe và gán cho mỗi xe một `track_id` duy nhất [main.py: Dòng 848](./main.py#L848). Thuật toán này giúp hệ thống biết được xe nào đã đi qua và đang ở đâu, tránh việc nhận diện lặp đi lặp lại cùng một xe ở mỗi khung hình.

### 3. ROI (Region of Interest - Vùng quan tâm)
*   **Là gì?** Là một phân vùng không gian xác định trên khung hình mà chúng ta lựa chọn để chạy các bộ lọc hoặc thuật toán nhận diện.
*   **Làm gì trong dự án?** Được thiết lập ở nửa dưới video [main.py: Dòng 60](./main.py#L60). Hệ thống chỉ thực hiện tìm biển số khi xe đi vào vùng này [main.py: Dòng 878](./main.py#L878) nhằm giảm tải CPU/GPU (không quét xe ở quá xa) và đảm bảo thu được ảnh biển số lớn, rõ ràng nhất khi xe ở gần camera.

### 4. Variance of Laplacian (Độ sắc nét Laplacian)
*   **Là gì?** Là phương pháp toán học dùng để tính toán mức độ sắc nét (focus) hoặc mờ nhòe (blur) của một bức ảnh bằng cách tính phương sai của ảnh sau khi đi qua toán tử vi phân Laplacian.
*   **Làm gì trong dự án?** Đo độ sắc nét của vùng biển số xe được cắt ra [main.py: Dòng 215](./main.py#L215). Nếu ảnh biển số bị mờ (phương sai thấp hơn ngưỡng tối thiểu), hệ thống sẽ loại bỏ để tránh đưa ảnh xấu vào OCR.

### 5. Cosine Similarity (Độ tương đồng Cosine)
*   **Là gì?** Là công thức toán học đo góc giữa hai vector trong không gian đa chiều để đánh giá mức độ tương đồng giữa chúng (kết quả từ -1 đến 1, càng gần 1 thì càng giống nhau).
*   **Làm gì trong dự án?** Dùng trong hàm khử trùng lặp [main.py: Dòng 612](./main.py#L612). Hệ thống chuyển ảnh biển số về dạng vector đặc trưng [main.py: Dòng 579](./main.py#L579) rồi so sánh Cosine để phát hiện xem hai ID xe khác nhau có thực chất là cùng một xe hay không (do đứt mạch tracking).

### 6. Clean Crop (Cắt ảnh sạch)
*   **Là gì?** Là kỹ thuật cắt vùng ảnh đối tượng từ một bản sao sạch của khung hình trước khi vẽ bất kỳ hộp bao, nét vẽ debug hoặc chữ viết đè lên.
*   **Làm gì trong dự án?** Hệ thống tạo bản sao sạch `clean_frame` [main.py: Dòng 846](./main.py#L846) để trích xuất ảnh xe [main.py: Dòng 904](./main.py#L904) và biển số lưu vào các thư mục kết quả. Điều này giúp ảnh đầu ra không bị lem luốc bởi các nét vẽ debug của hệ thống.

### 7. SQLite & CSV
*   **Là gì?** SQLite là một hệ quản trị cơ sở dữ liệu quan hệ nhỏ gọn lưu trực tiếp dưới dạng file. CSV là định dạng lưu trữ bảng biểu phân tách bằng dấu phẩy, mở được bằng Excel.
*   **Làm gì trong dự án?** SQLite lưu trữ lịch sử vi phạm có cấu trúc [main.py: Dòng 138](./main.py#L138) phục vụ truy vấn lâu dài. CSV ghi nhận báo cáo [main.py: Dòng 718](./main.py#L718) phục vụ xuất file báo cáo nhanh.

---

## 💡 LỜI KHUYÊN KHI NÓI
1. **Nói chậm rãi, rõ chữ:** Đừng vội vàng, hãy giữ phong thái tự tin. Giáo viên đánh giá cao sự hiểu sâu về thuật toán hơn là việc học thuộc lòng.
2. **Nhấn mạnh vào ByteTrack & Laplacian Sharpness:** Đây là hai điểm cộng kỹ thuật cực lớn trong bài báo cáo của bạn. Hãy nói rõ bạn dùng chúng để giải quyết bài toán thực tế (mất dấu xe và ảnh mờ).
3. **Thao tác Demo mượt mà:** Hãy mở sẵn thư mục `outputs/` trước khi trình bày để khi giới thiệu kết quả, bạn có thể click mở ảnh review ngay lập tức.
