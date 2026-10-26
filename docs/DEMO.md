# Kịch bản demo và nộp bài

Kịch bản chi tiết theo từng thao tác và lời nói với thầy: [DEMO_THUYET_TRINH.md](DEMO_THUYET_TRINH.md).

## Chuẩn bị

- Mở URL cloud và `/api/health` trước buổi demo, chờ dịch vụ khởi động xong.
- Cắm sạc laptop, camera đủ sáng, người cách camera vừa đủ để thấy rõ mũi và miệng.
- Chuẩn bị ba người: một người đeo khẩu trang che kín mũi/miệng, một người không đeo, một người đeo dưới mũi. Có thể dùng ảnh nhóm có sẵn để kiểm tra trước.
- Không đưa test accuracy của bài nghiên cứu mẫu vào phần kết quả của mình. Dùng bảng và biểu đồ trong `artifacts/`.

## Demo 5–7 phút

1. Giải thích yêu cầu ba nhãn và kiến trúc: camera laptop → HTTPS API → YuNet tìm các khuôn mặt → CNN phân loại các crop → JSON → giao diện vẽ kết quả.
2. Mở thông tin model, chỉ ra train bằng GPU và có đúng ba nhãn. Trình bày file model duy nhất `mask_detector.bundle` và SHA256.
3. Bật camera, cho cả ba người xuất hiện **trong cùng khung hình**. Chỉ rõ từng bounding box và ba bộ đếm. Đổi vị trí và trạng thái khẩu trang để chứng minh không dựa vào vị trí cố định.
4. Chỉ ra từng độ trễ. Suy luận server là một phần của RTT, RTT là một phần của toàn trình. Server và trình duyệt đo bằng đồng hồ riêng, không cần đồng bộ giờ.
5. Thu ít nhất 30 khung hình camera ổn định, Xuất CSV. Có thể bấm **Đo tự động 30 mẫu** để đo có kiểm soát trên ảnh nhóm cố định, rồi xuất CSV riêng. Trình bày mean, median/P95 và việc loại warm-up. Ghi URL backend trong CSV để xác nhận cloud thật.
6. Mở biểu đồ learning curves/confusion matrix và báo cáo Precision/Recall/F1 của ba lớp. Nêu giới hạn mẫu đeo sai ít và khuôn mặt nhỏ/che khuất.
7. Nếu có PostgreSQL, bật gửi số đo và truy vấn 5–20 bản ghi mới nhất. Nếu không cần database thì CSV đã đủ yêu cầu slide.

## Các tệp nộp

- Mã nguồn đầy đủ, README và tài liệu đối chiếu.
- `models/mask_detector.bundle`.
- `artifacts/dataset_report.json`, `evaluation.json`, `history.json`, `training_results.png`, `model_export.json`.
- Minh chứng ảnh nhiều người đủ ba trạng thái và số đo CSV từ cloud. Nếu là ảnh ghép minh họa phải ghi rõ là ảnh ghép, không gọi là ảnh nhóm tự nhiên.
- URL cloud đang chạy và URL repository.

## Câu trả lời ngắn khi bảo vệ

**Vì sao không phân loại cả khung hình một lần?** Một ảnh có thể có nhiều người thuộc các lớp khác nhau. Cần tìm box từng khuôn mặt rồi phân loại mỗi crop, trả về nhiều kết quả.

**Có phải CNN không?** Có. MobileNetV3-Small là một kiến trúc CNN dùng tích chập depthwise/pointwise và các khối inverted residual. Transfer learning vẫn là huấn luyện CNN.

**Vì sao model có một file mà bên trong có nhiều thành phần?** Pipeline cần CNN và detector. File bundle đóng gói cả hai cùng metadata, nạp bằng một đường dẫn duy nhất. ONNX của CNN có thể lấy ra từ bundle khi cần, nhưng backend không yêu cầu thao tác đó.

**Accuracy crop có bảo đảm camera cũng như vậy không?** Không. Pipeline còn có thể bỏ sót khuôn mặt và camera khác tập dữ liệu. Cần báo cáo crop classifier và detector/pipeline riêng.

**Vì sao toàn trình cao hơn suy luận?** Gồm thêm lấy frame, resize, mã hóa JPEG, upload/download, chờ server và vẽ kết quả. Render cold start có thể lớn hơn nhiều so với thời gian chạy model.
