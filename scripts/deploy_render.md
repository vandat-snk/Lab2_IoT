# Triển khai sau khi code đã push

Repository: https://github.com/vandat-snk/Lab2_IoT

Mở https://dashboard.render.com/select-repo?type=web và chọn **Public Git Repository** nếu chưa cấp quyền tích hợp GitHub. Dán URL trên, chọn Docker, Singapore và Free. Không cần tạo API key, database hay gói trả phí cho nhận dạng.

Nếu chọn Blueprint, Render sẽ đọc `render.yaml` ở thư mục gốc. Đảm bảo nhánh chứa `models/mask_detector.bundle`. Sau khi Deploy thành Live, gửi URL ứng dụng để kiểm tra API và đo độ trễ cloud thực tế.

Đăng nhập Render/GitHub trong trình duyệt không tự cấp quyền truy cập API/CLI cho ứng dụng khác. Không đưa mật khẩu/token vào mã nguồn hoặc chat.
