# MaskLab — IoT Lab 3: CNN nhận dạng khẩu trang trên cloud

Hệ thống nhận dạng **nhiều khuôn mặt trong cùng một ảnh**, mỗi khuôn mặt thuộc một trong ba lớp: **đeo đúng**, **không đeo**, **đeo sai**. Dùng camera laptop hoặc tải ảnh. Backend FastAPI nhận ảnh, phát hiện khuôn mặt bằng CNN YuNet, phân loại từng khuôn mặt bằng CNN MobileNetV3-Small và trả về tọa độ, xác suất, số lượng cùng thời gian suy luận.

**Bản cloud đang chạy:** [Mở MaskLab trên Render](https://iot-mask-cnn-lab3.onrender.com). [Thử API](https://iot-mask-cnn-lab3.onrender.com/docs) · [Tải một file model](https://iot-mask-cnn-lab3.onrender.com/api/model/download).

Mở trang cloud, bấm **Bật camera laptop** và cho phép camera, hoặc **Ảnh mẫu 3 lớp**. Nút **Đo tự động 30 mẫu** thu số đo thật từ trình duyệt đến cloud; giữ tab hiển thị khi đo, sau đó **Xuất CSV**. Render Free có thể cần khởi động lại sau thời gian không dùng.

## Chạy bản đã có model

Windows, Python 3.11–3.13:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\run.bat
```

Mở **http://localhost:8000**, bấm **Bật camera laptop** và cho phép trình duyệt truy cập camera. Hoặc bấm **Chọn ảnh**. Nút Dừng tắt camera. Nếu mở giao diện từ cloud, địa chỉ HTTPS cho phép trình duyệt dùng camera laptop của người xem.

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Không cần cài PyTorch/CUDA để chạy backend. File duy nhất cần triển khai là `models/mask_detector.bundle` cùng mã ứng dụng. Bundle là file ZIP đóng gói ONNX CNN, YuNet detector, metadata, nhãn và SHA256; backend nạp trực tiếp trong bộ nhớ, không cần giải nén thủ công. Checkpoint để train tiếp được giữ riêng trên máy, không phải đầu vào của backend.

## Triển khai Render Free

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/vandat-snk/Lab2_IoT)

Nút trên đọc `render.yaml`, khai báo duy nhất một web service gói Free. Kiểm tra cấu hình rồi triển khai trong tài khoản của bạn.

1. Đưa dự án và `models/mask_detector.bundle` lên repository GitHub.
2. Trong [Render Dashboard](https://dashboard.render.com), chọn **New → Web Service**, chọn repository `vandat-snk/Lab2_IoT` hoặc dùng **Public Git Repository** với URL repository.
3. Chọn **Docker**, region **Singapore**, instance **Free**. Không chọn gói trả phí. Dockerfile nằm ngay thư mục gốc. Port được đọc từ biến `PORT` của Render.
4. Health check: `/api/health`. Bấm **Deploy Web Service**. Có thể dùng `render.yaml` bằng **New → Blueprint**.
5. Khi Live, mở URL `https://…onrender.com`. Kiểm tra `/api/health` trả `status: ok` và đúng ba nhãn. Giao diện được phục vụ bởi cùng backend.
6. Trên máy local cũng có thể mở `http://localhost:8000`, nhập URL Render vào **Địa chỉ backend**, bấm Kết nối rồi bật camera. CORS đã cho phép localhost cổng 8000 và 7860.

Render Free ngủ sau 15 phút không có yêu cầu và mất khoảng một phút để khởi động lại. Bấm Kết nối và chờ backend sẵn sàng trước khi demo. Giữ **lần đầu** tách khỏi thống kê ổn định. Giới hạn miễn phí có thể thay đổi: [Render Free](https://render.com/docs/free). Ngày kiểm tra chính sách: 26/10/2026. Không cấu hình database trả phí trong blueprint.

## Train lại bằng GPU

Môi trường đã kiểm tra: RTX 2050 4 GB. Backend cloud dùng CPU; GPU được ưu tiên cho huấn luyện local. Cài CUDA build của PyTorch, không cài gói TensorFlow GPU cũ cho Windows.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/train.py --device cuda --epochs 25 --batch-size 32
.\.venv\Scripts\python.exe scripts/export_model.py
```

Script train báo lỗi nếu yêu cầu CUDA mà không có GPU, tránh vô tình train bằng CPU. Với VRAM ít hơn, giảm batch size. `download_cuda_wheel.py` là phương án tải tiếp từng phần cho **Python 3.13 Windows x64**, đã ghim SHA256 của wheel chính thức, chỉ cần dùng khi tải pip thông thường bị gián đoạn.

MobileNetV3-Small là mạng CNN nhẹ sử dụng transfer learning ImageNet, fine-tune toàn mạng với đầu phân loại 3 lớp. Đầu vào RGB 160×160, chuẩn hóa ImageNet. AdamW, cosine LR, augmentation chỉ ở train, cân bằng lớp bằng WeightedRandomSampler. Chọn checkpoint bằng **macro F1 trên validation**. Test được đánh giá sau khi chọn checkpoint. Mã trong `scripts/train.py`.

Dataset FMD có nhãn ba lớp, tải từ mirror Hugging Face có ghim revision; nguồn gốc và SHA256 nằm trong `data/raw/`. Chia theo **nhóm ảnh gốc trước khi crop**, gom trùng RGB và ảnh gần trùng bằng dHash, để tránh cùng ảnh xuất hiện ở nhiều tập. Không đảm bảo loại hết trùng danh tính vì bộ dữ liệu không có person ID. Những khuôn mặt nhỏ hơn 12 pixel bị loại khỏi dữ liệu crop để tránh học ảnh mất thông tin. Số mẫu thực dùng nằm trong `artifacts/dataset_report.json`.

## Đo ba loại độ trễ đúng yêu cầu

| Số đo | Điểm bắt đầu và kết thúc | Nơi đo |
|---|---|---|
| Suy luận backend/cloud | Tổng thời gian YuNet `detect()` và ONNX CNN `run()` trên các batch khuôn mặt | `time.perf_counter()` ở server |
| Phản hồi local ↔ backend/cloud | Ngay trước `fetch()` gửi ảnh đến khi đọc xong JSON phản hồi | `performance.now()` ở trình duyệt |
| Toàn trình | Trước khi lấy khung hình/copy ảnh nguồn, qua resize, encode JPEG, gửi và nhận, vẽ bounding box đến cơ hội trình duyệt paint tiếp theo | `performance.now()` ở trình duyệt |

Toàn trình bắt đầu ở thời điểm ứng dụng lấy khung hình từ video, không tính thời gian phơi sáng/cảm biến hoặc ISP bên trong webcam. Với ảnh tải lên, bắt đầu sau bước giải mã file đầu vào và trước resize. Hai `requestAnimationFrame` là phép xấp xỉ thời điểm kết quả sẵn sàng hiển thị, không phải phép đo quang học màn hình.

`inference_ms = face_detection_ms + cnn_ms`. `server_processing_ms` gồm đọc/giải mã ảnh, chờ khóa suy luận, tiền xử lý, suy luận và hậu xử lý trong handler; không gồm thời gian tải upload trước khi handler bắt đầu hay gửi response. `RTT - server_processing_ms` chỉ là phần dư vận chuyển/HTTP/ngoài handler, **không phải độ trễ mạng một chiều**. Không trừ timestamp giữa hai máy khác nhau. Mỗi lần chỉ có một khung hình đang chờ, tránh tích hàng đợi camera.

Mẫu đầu sau kết nối hoặc bật camera được đánh dấu `warmup=true`, vẫn có trong CSV nhưng loại khỏi mean/P95. Mỗi CSV có URL backend, SHA256 model, request ID, thời gian UTC, số khuôn mặt và số lượng từng lớp. Nếu backend là localhost thì phải gọi kết quả **local–local**. Chỉ gọi **local–cloud** sau khi thực sự gửi đến URL cloud.

Nút **Đo tự động 30 mẫu** trong giao diện chạy 31 lượt trên ảnh nhóm mẫu (1 warm-up + 30 mẫu ổn định), lấy thời gian thật của trình duyệt và backend, chờ vẽ kết quả ở mỗi lượt. Sau đó bấm **Xuất CSV**. Số đo này dùng ảnh mẫu cố định; vẫn nên đo thêm camera thật trong buổi demo. Bấm Dừng để ngắt phép đo.

Benchmark 30 lượt API (không thay thế số đo toàn trình trong trình duyệt):

```powershell
.\.venv\Scripts\python.exe scripts/benchmark.py --url https://TEN-UNG-DUNG.onrender.com --image web/samples/three_classes.jpg --requests 30 --output artifacts/cloud_latency
```

Script CLI cố ý để `end_to_end_ms=null` vì không lấy ảnh webcam và vẽ kết quả trên trình duyệt. Đo toàn trình bằng giao diện và nút **Xuất CSV**.

## PostgreSQL tùy chọn

Cloud đã có PostgreSQL riêng `masklab-measurements` (database `masklab`), cùng vùng Singapore với backend. Gói Free hết hạn **25/11/2026**. Kết nối nội bộ được lưu trong biến môi trường `DATABASE_URL` của Render; không commit mật khẩu hoặc `.env`. Xem cách demo và giới hạn tại [docs/POSTGRESQL.md](docs/POSTGRESQL.md).

Chọn **Gửi số đo vào PostgreSQL** sau khi giao diện báo kết nối thành công. Backend tạo bảng `lab_measurements` khi ghi lần đầu và chỉ trả `saved: true`, `record_id` sau khi commit. Giao diện hiện số bản ghi đã lưu trong phiên và ID mới nhất. Lỗi ghi trả HTTP 503; nhận dạng và xuất CSV vẫn hoạt động. Chỉ lưu ID, SHA256 model, ba độ trễ và số lượng từng lớp; không lưu ảnh/video. Để kiểm tra bằng SQL:

```sql
SELECT created_at, inference_ms, round_trip_ms, end_to_end_ms, counts
FROM lab_measurements ORDER BY id DESC LIMIT 20;
```

## API

- `GET /api/health`: trạng thái model, ba nhãn, SHA256, trạng thái cấu hình PostgreSQL.
- `GET /api/model-info`: kiến trúc, tiền xử lý, thông tin train/test thật.
- `GET /api/model/download`: tải một file model bundle từ backend/cloud để đối chiếu SHA256 hoặc lưu lại.
- `POST /api/predict`: multipart `file`, tùy chọn `detection_threshold` và `classification_threshold`. Trả tọa độ `[x1,y1,x2,y2]` theo ảnh gửi lên, nhãn của **từng** khuôn mặt, xác suất 3 lớp, counts và timing. Nhãn `uncertain` nếu confidence dưới ngưỡng; không ép đoán chắc chắn khi model thiếu tự tin.
- `POST /api/measurements`: gửi số đo trình duyệt vào PostgreSQL nếu có cấu hình.
- `GET /docs`: Swagger UI để thử API.

Ảnh tối đa 8 MB, JPEG/PNG/WebP, tối đa 4096 pixel mỗi chiều và 16 megapixel. Giao diện gửi ảnh tối đa 1280×720. YuNet resize giữ tỉ lệ, cạnh dài 640 pixel, ngưỡng phát hiện 0,7 chọn qua validation, NMS loại khung trùng. Bộ phát hiện trả tối đa 100 khuôn mặt/khung hình, không lấy duy nhất khuôn mặt lớn nhất. CNN xử lý tối đa 16 crop/batch để giới hạn RAM, rồi gộp mọi kết quả của cùng khung hình vào một phản hồi API. YuNet và tất cả batch CNN đều nằm trong số đo suy luận.

## Kiểm tra và minh chứng

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`artifacts/evaluation.json`: Accuracy, Precision, Recall, F1 từng lớp, macro/weighted F1, confusion matrix trên test chưa dùng chọn checkpoint. `artifacts/history.json`: loss và accuracy/F1 theo epoch. `artifacts/training_results.png`: biểu đồ thật. `artifacts/model_export.json`: dung lượng, SHA256 bundle và sai số đối chiếu PyTorch–ONNX. `artifacts/pipeline_evaluation.json`: đánh giá cả phát hiện + phân loại nếu đã chạy script tương ứng.

Xem [đối chiếu yêu cầu](docs/REQUIREMENTS.md), [báo cáo kỹ thuật](docs/REPORT.md), [kịch bản demo](docs/DEMO.md) và [nguồn tham khảo](docs/SOURCES.md). Không sử dụng con số 99,94% từ bài mẫu làm kết quả của dự án này. Khả năng camera thực tế phụ thuộc ánh sáng, góc mặt, khoảng cách và dữ liệu. Ảnh có ba lớp được hỗ trợ trong cùng lượt suy luận; độ chính xác thực tế cần kiểm tra bằng ảnh/nhóm người mới.
