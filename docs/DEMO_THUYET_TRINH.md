# Kịch bản demo và trả lời thầy: CNN nhận dạng khẩu trang, IoT và cloud

Đối chiếu trực tiếp toàn bộ bốn slide trong `E:\UsersData\3- Phat trien he thong IoT LAB 3-2.pptx`.
Số liệu dưới đây lấy từ model, dataset và kết quả huấn luyện thực của dự án.

## 1. Chuẩn bị trước buổi demo

Mở sẵn các cửa sổ theo thứ tự này:

1. Ứng dụng: https://iot-mask-cnn-lab3.onrender.com/
2. File Explorer tại `E:\VanDat_\IoTUD\Lab2.3`.
3. PyCharm mở dự án, chuẩn bị `artifacts/evaluation.json`, `scripts/export_model.py`, `Dockerfile`, `backend/main.py`.
4. Render: https://dashboard.render.com/web/srv-dbff2ue7bikc73dac4v0
5. API: https://iot-mask-cnn-lab3.onrender.com/docs và https://iot-mask-cnn-lab3.onrender.com/api/health
6. pgAdmin: server **Render MaskLab - Cloud**, database **masklab**, Query Tool đã đọc `public.lab_measurements`.

Mở app trước vài phút để dịch vụ miễn phí khởi động. Cắm sạc laptop. Kiểm tra Wi-Fi, camera, ánh sáng và trạng thái PostgreSQL.
Chuẩn bị ba người trong cùng khung hình: một đeo đúng, một không đeo, một đeo dưới mũi. Dùng ảnh mẫu ba lớp để kiểm tra trước.
Nếu đổi mạng và pgAdmin không kết nối được, cập nhật đúng IP mới trong Render. Giữ SSL, chỉ cho phép IP cần thiết.
Không trình chiếu trang Environment có mật khẩu đang hiện.

## 2. Thứ tự demo khoảng 10–12 phút

### Bước 1: Giới thiệu mục tiêu và luồng xử lý, khoảng 40 giây

**Mở:** trang ứng dụng cloud.

**Nói:**

> Em thực hiện Lab 3 về nhận dạng ba trạng thái khẩu trang: đeo đúng, không đeo và đeo sai. Camera laptop lấy khung hình và gửi qua HTTPS tới backend trên Render. Backend tìm từng khuôn mặt bằng YuNet, cắt vùng khuôn mặt rồi chạy CNN MobileNetV3-Small để phân loại. Kết quả trả về giao diện gồm vị trí, nhãn, độ tin cậy, số người từng loại và các độ trễ.

**Chỉ trên giao diện:** nguồn ảnh, ba bộ đếm, địa chỉ backend từ xa và ba ô độ trễ.
Model xử lý trên server. Trình duyệt đảm nhiệm camera, gửi ảnh, vẽ kết quả và đo thời gian phía người dùng.

### Bước 2: Mở dataset thật, khoảng 1 phút

**Mở trong File Explorer:**

- Ảnh gốc: `E:\VanDat_\IoTUD\Lab2.3\data\raw\images`
- Nhãn XML: `E:\VanDat_\IoTUD\Lab2.3\data\raw\annotations`
- Ảnh dùng để train: `E:\VanDat_\IoTUD\Lab2.3\data\processed\train`

Mở một ảnh gốc và tệp XML cùng tên. Trong XML, chỉ tên nhãn và tọa độ bounding box.
Sau đó mở ba thư mục nhãn trong `train` để thầy thấy các ảnh khuôn mặt đã cắt.

**Nói:**

> Em dùng bộ Face Mask Detection của Andrewmvd trên Kaggle, tải từ bản mirror Hugging Face có lưu phiên bản nguồn. Bộ gốc gồm 853 ảnh và 853 tệp nhãn XML. Sau lọc, em dùng 851 ảnh nguồn hợp lệ và tạo 3.479 ảnh khuôn mặt kích thước 160 × 160. Em chia theo nhóm ảnh nguồn trước khi cắt để giảm việc các khuôn mặt từ cùng ảnh lọt sang cả train và test.

| Nhãn | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| Đeo đúng (`with_mask`) | 1.935 | 416 | 443 | 2.794 |
| Không đeo (`without_mask`) | 401 | 83 | 86 | 570 |
| Đeo sai (`mask_weared_incorrect`) | 81 | 17 | 17 | 115 |
| Tổng | 2.417 | 516 | 546 | 3.479 |

**Nói thêm nếu thầy hỏi 20 mẫu:**

> Slide yêu cầu mỗi nhãn 20 mẫu khác nhau. Tập train của em có 1.935, 401 và 81 mẫu tương ứng, đều vượt mức này. Ảnh tăng cường không được tính thành mẫu gốc độc lập.

**Nguồn và minh chứng:**

- Kaggle: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- Mirror: https://huggingface.co/datasets/hmnshudhmn24/face-mask-detection
- Nguồn đã tải: `E:\VanDat_\IoTUD\Lab2.3\data\raw\provenance.json`
- Thống kê: `E:\VanDat_\IoTUD\Lab2.3\artifacts\dataset_report.json`
- Mã tải và chuẩn bị: `scripts/download_data.py`, `scripts/prepare_data.py` trong thư mục dự án.

Dataset nằm trên laptop để train, được loại khỏi Git để tránh đưa toàn bộ ảnh lên repository. Backend cloud chỉ cần model đã xuất để nhận dạng.

### Bước 3: Chứng minh train bằng gì và trong bao lâu, khoảng 1 phút

**Mở:** `E:\VanDat_\IoTUD\Lab2.3\artifacts\evaluation.json`.

Chỉ các trường `architecture`, `device`, `torch_version`, `cuda_version`, `epochs`, `batch_size`, `best_epoch`, `training_seconds`.
Sau đó mở `E:\VanDat_\IoTUD\Lab2.3\artifacts\training_results.png`.

**Nói:**

> Em dùng Python, PyTorch 2.9.1 và CUDA 12.8 để fine-tune CNN MobileNetV3-Small trên GPU NVIDIA GeForce RTX 2050 của laptop. Model khởi đầu từ trọng số pretrained ImageNet và em huấn luyện lại cho ba nhãn khẩu trang. Em chạy 25 epoch, batch size 32. Thời gian vòng train và đánh giá cuối là 1.097,78 giây, khoảng 18 phút 18 giây. Checkpoint tốt nhất là epoch 13, chọn theo macro F1 trên validation.

**Nếu thầy hỏi file train:** mở `E:\VanDat_\IoTUD\Lab2.3\scripts\train.py`.
Chỉ `make_model()`, lớp đầu ra ba nhãn, `--device cuda`, `model.to(device)`, vòng lặp epoch và `torch.save()`.

**Nếu thầy hỏi kết quả:**

> Accuracy trên tập test gồm 546 ảnh khuôn mặt đã cắt là 97,80%, macro F1 là 0,8898. Đây là kết quả classifier trên ảnh crop. Khi chạy camera, kết quả còn phụ thuộc bộ phát hiện khuôn mặt và điều kiện ánh sáng. Lớp đeo sai có ít dữ liệu nên còn hạn chế.

Thời gian 18 phút 18 giây không gồm tải dữ liệu, cài thư viện, chuẩn bị ảnh hoặc đưa ứng dụng lên cloud.
YuNet là bộ phát hiện pretrained được sử dụng lại, không phải bộ phát hiện do em tự train trong lần chạy này.
Không chạy lại toàn bộ train trong phần demo thường lệ. Trình bày kết quả lần train đã hoàn tất và mã có thể tái chạy.

### Bước 4: Mở file model duy nhất và phần đóng gói, khoảng 1 phút

**Mở File Explorer:** `E:\VanDat_\IoTUD\Lab2.3\models`.
Chọn `mask_detector.bundle`, mở Properties nếu cần xem kích thước.

**Nói:**

> File dùng để triển khai là `mask_detector.bundle`, khoảng 5,59 MiB. File này chứa CNN phân loại đã xuất sang ONNX, YuNet phát hiện khuôn mặt và metadata. Backend chỉ nhận một đường dẫn file bundle để nạp toàn bộ pipeline.

**Khi thầy hỏi “file nào gom model?”**, phân biệt:

| Vai trò | File |
|---|---|
| Định nghĩa CNN và chạy huấn luyện | `E:\VanDat_\IoTUD\Lab2.3\scripts\train.py` |
| Trọng số PyTorch sau train | `E:\VanDat_\IoTUD\Lab2.3\artifacts\checkpoint.pt` |
| Mã xuất ONNX và gom các thành phần | `E:\VanDat_\IoTUD\Lab2.3\scripts\export_model.py` |
| Một file model đưa lên cloud | `E:\VanDat_\IoTUD\Lab2.3\models\mask_detector.bundle` |
| Mã mở bundle và chạy suy luận | `E:\VanDat_\IoTUD\Lab2.3\backend\inference.py` |

**Mở:** `scripts/export_model.py`, chỉ `torch.onnx.export`, biến `files`, `metadata` và đoạn `zipfile.ZipFile`.

Các thành phần chính bên trong bundle:

Có thể mở file `.bundle` bằng WinRAR để xem nội dung vì định dạng đóng gói là ZIP. Không cần giải nén để backend chạy.

```text
classifier.onnx     CNN phân loại ba trạng thái
face_detector.onnx YuNet phát hiện khuôn mặt
metadata.json      Nhãn, chuẩn hóa đầu vào, thông tin train và checksum
NOTICE.txt         Thông tin nguồn
licenses/          Giấy phép các thành phần
```

Không sửa hoặc xuất lại model trong buổi demo.

### Bước 5: Chứng minh model và backend nằm trên cloud, khoảng 1 phút

**Mở:** Render Dashboard, service **iot-mask-cnn-lab3**, chỉ trạng thái đang chạy và repository liên kết.

**Nói:**

> Mã nguồn backend nằm trong thư mục `backend` của repository GitHub Lab2_IoT. Render lấy code, build Docker image và chạy FastAPI/Uvicorn ở Singapore. Trong container, backend nằm tại `/app/backend`, giao diện tại `/app/web` và file model tại `/app/models/mask_detector.bundle`.

**Mở:** `E:\VanDat_\IoTUD\Lab2.3\Dockerfile`.
Chỉ `WORKDIR /app`, `COPY backend`, `COPY web`, `COPY models/mask_detector.bundle` và lệnh chạy `uvicorn backend.main:app`.

**Chứng minh file cloud có thật:** mở https://iot-mask-cnn-lab3.onrender.com/api/model/download để tải `mask_detector.bundle` từ chính server.
Mở https://iot-mask-cnn-lab3.onrender.com/api/health để chỉ ba nhãn, `CPUExecutionProvider` và SHA256 model.
SHA256 trong health khớp file local theo `artifacts/model_export.json`.

**Nói rõ:**

> GPU dùng cho huấn luyện trên laptop. Bản Render Free hiện chạy suy luận bằng CPU qua ONNX Runtime. Model không train lại mỗi khi bật camera.

**API:** mở https://iot-mask-cnn-lab3.onrender.com/docs.
Chỉ `POST /api/predict` nhận file ảnh và trả `detections`, `counts`, `timing`; có thể dùng Try it out với ảnh mẫu nếu thầy muốn xem JSON.
`GET /api/model-info` trả thông tin model, `POST /api/measurements` nhận số đo để ghi PostgreSQL.

### Bước 6: Demo nhận dạng đồng thời ba loại, khoảng 1–2 phút

1. Trên app cloud, bấm **Ảnh mẫu 3 lớp** để kiểm tra.
2. Chỉ cùng một khung hình có nhiều khuôn mặt, box riêng, nhãn riêng và ba bộ đếm. Ảnh mẫu hiện trả 15 khuôn mặt, gồm 13 đeo đúng, một không đeo, một đeo sai ở ngưỡng mặc định 50%.
3. Bật **camera laptop**, cho ba người thuộc ba trạng thái cùng xuất hiện.
4. Đổi vị trí hoặc đổi trạng thái khẩu trang, quan sát nhãn thay đổi.

**Nói:**

> Một ảnh có nhiều người nên em tìm từng khuôn mặt trước, sau đó phân loại riêng từng vùng. Vì vậy trong một khung hình có thể trả cả ba nhãn. Em không gán một nhãn chung cho toàn ảnh.

Không chứng minh yêu cầu đồng thời chỉ bằng cách lần lượt đưa ba ảnh một người vào camera.
Nếu buổi demo chỉ có một người, ảnh nhóm công khai chứng minh nhiều người đồng thời; cần nói đúng nguồn ảnh. Demo người thật đủ ba trạng thái cùng khung hình là bằng chứng tốt hơn khi có điều kiện.
`Chưa chắc chắn` là trạng thái khi confidence thấp, không phải lớp thứ tư được train.

### Bước 7: Giải thích và đo đủ ba độ trễ, khoảng 1–2 phút

**Chỉ ba ô trên app và nói:**

| Số đo | Bắt đầu và kết thúc | Thành phần trong bản hiện tại |
|---|---|---|
| Suy luận trên cloud | Đồng hồ server đo các lần chạy detector và classifier | Thời gian YuNet detect cộng các lần ONNX CNN run. Các bước giải mã ảnh, resize/crop/chuẩn hóa và chờ khóa được theo dõi ngoài chỉ số này |
| Phản hồi local ↔ cloud, RTT | Trình duyệt bắt đầu gửi ảnh đến khi đọc và giải mã JSON kết quả xong | Truyền ảnh, mạng, chờ và xử lý toàn bộ trên backend, trả kết quả và đọc JSON |
| Toàn trình | Ứng dụng bắt đầu lấy khung hình đến khi vẽ kết quả và chờ lịch hiển thị xong | Lấy frame, resize, mã hóa JPEG, RTT, vẽ và chờ paint |

**Nói:**

> Thời gian suy luận nằm bên trong thời gian phản hồi. Thời gian phản hồi nằm bên trong toàn trình. Vì vậy em không cộng ba con số lại. RTT cũng không phải chỉ riêng độ trễ mạng. Hai phía dùng đồng hồ riêng để đo khoảng thời gian, không lấy giờ server trừ giờ laptop.

Thời gian toàn trình không đo thời gian cảm biến camera trước khi ứng dụng lấy frame. Ghi PostgreSQL diễn ra sau khi kết thúc phép đo toàn trình, nên không nằm trong ba số này.

**Thao tác:** bấm **Đo tự động 30 mẫu**, giữ tab hiển thị, đợi chạy xong, bấm **Xuất CSV**.
Ứng dụng chạy 1 mẫu warm-up và 30 mẫu ổn định trên ảnh nhóm cố định. Mẫu đầu không được tính vào mean/P95 ổn định.
Nếu đo camera thật thì thu CSV riêng, trình bày đúng nguồn ảnh.

**Số đo cloud đã thu trước đây, để tham khảo và đối chiếu:**

| Chỉ số | Trung bình | P95 |
|---|---:|---:|
| Suy luận cloud | 230,69 ms | 292,19 ms |
| Phản hồi local–cloud | 428,59 ms | 492,80 ms |
| Toàn trình | 461,07 ms | 528,30 ms |

Đây là 30 mẫu ổn định trên ảnh nhóm cố định, tab hiển thị, file `E:\VanDat_\IoTUD\Lab2.3\artifacts\browser_cloud_latency.csv`.
Khi demo dùng số đo vừa thu. Kết quả thay đổi theo số khuôn mặt, mạng và tải server; không khẳng định lần nào cũng giống bảng này.
P95 nghĩa là khoảng 95% số đo trong tập nằm dưới hoặc bằng giá trị đó.

### Bước 8: Chứng minh dữ liệu được ghi vào PostgreSQL, khoảng 1 phút

1. Bật **Gửi số đo vào PostgreSQL** trên giao diện.
2. Chạy một ảnh hoặc camera vài giây, chỉ thông báo **Đã lưu … số đo … Bản ghi mới nhất #…**.
3. Chuyển pgAdmin, tab **masklab/masklab_user@Render MaskLab - Cloud**.
4. Nhấn **F5** / **Execute script** để đọc lại dữ liệu. Chỉ các ID mới và thời gian theo giờ Việt Nam.

**Đường dẫn lưu:**

**Servers → Render MaskLab - Cloud → Databases → masklab → Schemas → public → Tables → lab_measurements**.

**Nói:**

> Database PostgreSQL cũng nằm trên Render ở Singapore, tách thành service masklab-measurements. Backend ghi bảng public.lab_measurements của database masklab. pgAdmin trên laptop kết nối bằng SSL để xem dữ liệu cloud, không phải bảng của PostgreSQL local đang có sẵn.

| Cột trong database | Ý nghĩa |
|---|---|
| `id` | Số thứ tự bản ghi |
| `created_at` | Thời điểm backend ghi, lưu UTC |
| `request_id` | UUID của lần nhận dạng |
| `model_sha256` | Mã xác định phiên bản file model |
| `inference_ms` | Thời gian suy luận trên server |
| `round_trip_ms` | Thời gian phản hồi local–cloud |
| `end_to_end_ms` | Độ trễ toàn trình |
| `counts` | JSONB: số đeo đúng, không đeo, đeo sai, chưa chắc chắn |

**Nói thêm:**

> Database chỉ lưu kết quả đo và số lượng. Ảnh camera, video, dataset và trọng số model không lưu trong PostgreSQL. Backend chỉ báo ghi thành công sau khi database commit.

Mẫu warm-up cũng có thể nằm trong database. Cột `warmup` hiện có ở CSV, chưa có trong bảng SQL. Khi tính thống kê ổn định để báo cáo, dùng CSV đã lọc warm-up thay vì lấy AVG tất cả bản ghi trong database.
Gói PostgreSQL Free hiện hết hạn ngày **25/11/2026**. Giữ CSV cho báo cáo.

### Bước 9: Kết thúc, khoảng 20 giây

> Em đã trình bày dữ liệu ba nhãn, huấn luyện CNN nhẹ bằng GPU, xuất một file model và đưa lên cloud, API/giao diện dùng camera, nhận dạng nhiều người, ba độ trễ và bản ghi PostgreSQL thực tế. Hạn chế còn lại là dữ liệu đeo sai ít, khuôn mặt nhỏ/nghiêng có thể bị bỏ sót và cloud miễn phí có cold start.

## 3. Những câu thầy dễ hỏi

**“Teachable Machine đâu?”** Slide ghi ví dụ hỗ trợ Teachable Machine. Em dùng PyTorch để fine-tune CNN nhẹ bằng GPU và triển khai ONNX, vẫn thực hiện luồng huấn luyện, xuất file và prediction theo yêu cầu.

**“Arduino đâu?”** Slide 1 cho phép camera thường hoặc camera IoT. Bản em chọn camera laptop và backend cloud. Slide 2 có ví dụ backend/mạch Arduino; em chưa triển khai phần cứng Arduino trong bản này. Nếu thầy yêu cầu phần cứng riêng để chấm, cần bổ sung phần đó, không gọi laptop là Arduino.

**“Có phải tự train từ đầu?”** Em dùng transfer learning: pretrained ImageNet, thay đầu ra ba lớp và fine-tune CNN. YuNet sử dụng pretrained. Đây không phải huấn luyện tất cả trọng số từ khởi tạo ngẫu nhiên.

**“Một file nhưng trong đó hai model?”** Đây là một file triển khai đóng gói toàn bộ pipeline. CNN phân loại và detector có vai trò khác nhau. Backend chỉ nạp một đường dẫn bundle; em có thể chỉ nội dung file và mã export.

**“CNN có nhận dạng nhiều người được không?”** Detector tìm nhiều vùng khuôn mặt, classifier xử lý các crop theo batch và giữ ánh xạ kết quả về từng box. Do đó nhiều người và nhiều nhãn cùng tồn tại trong một lần nhận dạng.

**“Tại sao có 97,80% nhưng camera vẫn sai?”** Accuracy này đo trên ảnh crop có nhãn chuẩn. Camera còn bị ảnh hưởng bởi phát hiện mặt, miền dữ liệu, ánh sáng và góc nhìn. Không dùng accuracy crop để cam kết độ chính xác camera.

**“Sao cùng chương trình mà hôm nay độ trễ khác?”** Mạng, số người, tải CPU, cold start và lịch vẽ trình duyệt thay đổi. Cần cùng điều kiện, nhiều mẫu, tách warm-up và báo cáo mean/P95.

**“Sao PostgreSQL local không có bảng?”** Bản cloud dùng database masklab trên Render. Cần chọn server Render MaskLab - Cloud trong pgAdmin.

## 4. Bản trả lời nhanh cần nhớ

> Model CNN là MobileNetV3-Small, fine-tune bằng PyTorch/CUDA trên RTX 2050, chạy 25 epoch trong khoảng 18 phút 18 giây. Dataset Face Mask Detection lấy từ Kaggle qua mirror Hugging Face, nằm trong data/raw và data/processed trên laptop. File export_model.py đóng gói model thành mask_detector.bundle. File bundle nằm ở models trên laptop và /app/models trên Render. Backend FastAPI nằm ở backend trong repository và /app/backend trong container Render. Cloud chạy ONNX CPU. PostgreSQL Render lưu bảng masklab.public.lab_measurements với thời gian, UUID, SHA model, ba độ trễ và counts. Ba độ trễ là suy luận, phản hồi local–cloud và toàn trình.
