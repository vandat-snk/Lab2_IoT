# PostgreSQL cho MaskLab

## Triển khai cloud

- Backend: `iot-mask-cnn-lab3` trên Render, Singapore.
- PostgreSQL: `masklab-measurements`, database `masklab`, PostgreSQL 18, cùng vùng Singapore.
- Gói Free: 1 GB, hết hạn **25/11/2026**. Xuất CSV trước khi hết hạn; gói miễn phí không có backup tự động.
- Backend đọc `DATABASE_URL` từ Environment của Render, dùng địa chỉ nội bộ. Không đưa chuỗi kết nối hoặc mật khẩu vào mã nguồn, ảnh minh chứng hay báo cáo.
- Database chặn kết nối từ Internet; backend kết nối qua mạng nội bộ Render.
- PostgreSQL Docker `smartdoc-search-db-1` trên laptop thuộc dự án SmartDoc; ứng dụng cloud không sử dụng database đó.

## Dữ liệu được lưu

Bảng `lab_measurements` gồm `id`, `created_at` (UTC), `request_id`, `model_sha256`,
`inference_ms`, `round_trip_ms`, `end_to_end_ms`, `counts` (JSONB chứa số người theo bốn nhãn, gồm chưa chắc chắn).
Không lưu ảnh camera, video, ảnh crop, dataset hoặc file model vào PostgreSQL.

Số đo do trình duyệt gửi phục vụ demo; đây không phải hệ thống kiểm toán chống giả mạo số đo.

## Cách demo

1. Mở trang cloud, chờ kết nối backend thành công.
2. Kiểm tra dòng **PostgreSQL đã kết nối** rồi chọn **Gửi số đo vào PostgreSQL**.
3. Bấm **Ảnh mẫu 3 lớp**, bật camera, hoặc **Đo tự động 30 mẫu**.
4. Giao diện hiện **Đã lưu … số đo … Bản ghi mới nhất: #…** sau khi database xác nhận ghi.
5. Xuất CSV để giữ số đo phục vụ báo cáo.

`/api/health` trả `database_configured` khi có biến môi trường và `database_available` sau khi kiểm tra kết nối thật (cache 15 giây).
`POST /api/measurements` chỉ trả `saved: true` và `record_id` sau khi transaction commit; lỗi kết nối hoặc ghi trả HTTP 503.
Việc ghi PostgreSQL diễn ra sau khi đo toàn trình, nên thời gian ghi database không bị cộng vào ba số đo nhận dạng.
CSV vẫn được giữ trong trình duyệt khi PostgreSQL lỗi. Mẫu warm-up cũng có thể được lưu trong bảng; CSV có cột `warmup` để lọc khi tính thống kê.

## SQL minh họa khi đã kết nối bằng công cụ quản trị

```sql
SELECT id, created_at, request_id, inference_ms, round_trip_ms, end_to_end_ms, counts
FROM lab_measurements
ORDER BY id DESC
LIMIT 20;

SELECT COUNT(*) AS samples,
       AVG(inference_ms) AS mean_inference_ms,
       AVG(round_trip_ms) AS mean_round_trip_ms,
       AVG(end_to_end_ms) AS mean_end_to_end_ms
FROM lab_measurements;
```

Truy vấn AVG này gồm tất cả bản ghi, kể cả warm-up; dùng CSV để báo cáo kết quả ổn định theo phiên.
Không mở truy cập database công khai chỉ để trình chiếu SQL.

## Kiểm tra trước buổi báo cáo

Mở app trước buổi báo cáo để Render khởi động; kiểm tra trạng thái PostgreSQL và thử lưu một ảnh mẫu.
Nếu ô bị khóa, bấm kết nối lại và kiểm tra biến môi trường/database có còn hoạt động không.
Tài liệu giới hạn gói miễn phí: https://render.com/docs/free
