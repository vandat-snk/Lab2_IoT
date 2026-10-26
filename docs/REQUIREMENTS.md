# Đối chiếu yêu cầu thầy

Đã đọc toàn bộ 4 slide trong `3- Phat trien he thong IoT LAB 3-2.pptx` và 12 trang bài báo `Phát hiện khẩu trang khuôn mặt bằng phương pháp học sâu` của Đàm Minh Lịnh và cộng sự (2023). Slide cuối là hình cảm ơn. Ảnh chụp người dùng gửi khớp phần B slide 1.

| Yêu cầu trong slide | Cách thực hiện | Minh chứng cần kiểm tra |
|---|---|---|
| Dataset ảnh từ lab 2 hoặc khác | FMD có sẵn ba nhãn đeo đúng/không đeo/đeo sai | `scripts/download_data.py`, provenance, thống kê dataset |
| Mỗi nhãn ít nhất 20 mẫu khác nhau (slide 2) | Dùng toàn bộ dữ liệu hợp lệ, trên 20 mẫu gốc mỗi lớp | `artifacts/dataset_report.json`, không lấy augmentation làm mẫu độc lập |
| Huấn luyện model học sâu nhẹ | CNN MobileNetV3-Small, fine-tune 3 lớp, GPU CUDA | `scripts/train.py`, history và evaluation |
| Xuất model ra một file, lưu Cloud | Một `models/mask_detector.bundle` chứa toàn bộ model CNN và bộ phát hiện | `scripts/export_model.py`, SHA256, file trong bản build cloud |
| Camera thường hoặc camera IoT | Camera laptop bằng trình duyệt; API cũng nhận ảnh JPEG từ thiết bị IoT | Giao diện camera và `/api/predict` |
| Nhận dạng cùng lúc ba loại nhãn | Phát hiện mọi khuôn mặt, đưa các crop vào cùng batch CNN ba lớp | Bounding box riêng, counts riêng; kiểm thử batch nhiều khuôn mặt |
| Viết giao diện hiển thị | Giao diện tiếng Việt: camera, upload ảnh, đếm lớp, nhãn/confidence, ba số đo | `web/` |
| Cung cấp API | FastAPI, Swagger `/docs`, health check và predict | `backend/` |
| Suy luận trên Cloud | Đo YuNet detect + CNN run tại server bằng đồng hồ đơn điệu | `timing.inference_ms` |
| Phản hồi local ↔ Cloud | Đo từ gửi request đến đọc xong kết quả tại trình duyệt | RTT trên giao diện và CSV |
| Độ trễ toàn trình | Từ lấy frame đến kết quả vẽ xong/chờ paint, gồm encode và request | `end_to_end_ms` trên giao diện và CSV |

## Quan hệ với bài báo mẫu

Bài báo dùng CNN và pipeline camera → phát hiện khuôn mặt → crop → phân loại, đánh giá Accuracy/Precision/Recall/F1 và nhiều khuôn mặt. Dự án giữ cấu trúc xử lý đó, mở rộng **2 lớp thành 3 lớp**, chọn CNN nhẹ để phù hợp laptop 4 GB VRAM và cloud miễn phí, thêm API và phép đo local–cloud.

Không sao chép số liệu bài báo. Hai CNN trong bài báo có khoảng 17,17 triệu và 68,66 triệu tham số, đầu vào 256×256, đầu ra 2 nhãn. Dự án này dùng MobileNetV3-Small, đầu vào 160×160, đầu ra 3 nhãn. Đây là lựa chọn triển khai cho yêu cầu model nhẹ, không phải bản tái lập nguyên xi hai thí nghiệm trong bài báo. Có thể giữ 16 epoch/batch 32 khi cần thử lại cấu hình gần bài mẫu, nhưng checkpoint vẫn chọn bằng validation của dự án.

Slide 2 có mô tả ví dụ backend/mạch Arduino. Theo lựa chọn trực tiếp của người dùng, bài này dùng laptop làm thiết bị camera, server cloud làm backend và CNN. Không có phần giả lập rằng model CNN đang chạy trên Arduino.

## Điều kiện nghiệm thu thực tế

1. Health check backend cloud thành công, SHA256 giống model đã xuất local.
2. Một ảnh hoặc một nhóm người có đủ ba trạng thái, hiển thị bounding box/nhãn riêng và counts.
3. Camera laptop mở được, phản hồi lặp nhiều lần không chồng yêu cầu và không gắn box vào sai khung hình.
4. Thu ít nhất 30 mẫu ổn định local–cloud, xuất CSV có đủ ba độ trễ. Giữ riêng warm-up/cold start.
5. Phân biệt metric phân loại trên crop với chất lượng cả pipeline. Nêu số lượng test của lớp đeo sai để tránh suy diễn từ tập quá nhỏ.
6. Nếu chọn PostgreSQL, kiểm tra bản ghi thật trong database; chỉ có cấu hình kết nối chưa phải minh chứng ghi thành công.
