# Một file model triển khai

`mask_detector.bundle` là file duy nhất backend cần nạp.

Trong file:

- `classifier.onnx`: CNN MobileNetV3-Small fine-tune ba lớp.
- `face_detector.onnx`: CNN YuNet phát hiện các khuôn mặt.
- `metadata.json`: thứ tự nhãn, kích thước ảnh, chuẩn hóa, crop padding, kết quả đánh giá và SHA256 từng thành phần.
- `NOTICE.txt`: nguồn và ghi công.

Backend đọc trực tiếp bundle, kiểm tra SHA256 trước khi chạy. Không cần PyTorch trên cloud. SHA256 toàn bundle nằm trong `artifacts/model_export.json` và API `/api/health`.
