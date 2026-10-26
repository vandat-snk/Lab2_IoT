"""Write the lab report using actual run artifacts only."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'artifacts/dataset_report.json').read_text())
    evaluation=json.loads((ROOT/'artifacts/evaluation.json').read_text())
    export=json.loads((ROOT/'artifacts/model_export.json').read_text())
    classes=data['classes'];report=evaluation['classification_report']
    lines=['# Báo cáo Lab 3 — CNN nhận dạng ba trạng thái khẩu trang',
        '', '## 1. Mục tiêu và kiến trúc', '',
        'Phát hiện đồng thời nhiều khuôn mặt, phân loại đeo đúng, không đeo và đeo sai. Laptop lấy ảnh từ webcam, gửi JPEG qua HTTP(S) đến FastAPI. Server dùng CNN YuNet xác định các khuôn mặt, NMS loại khung trùng, crop có padding 10%, chuẩn hóa RGB và đưa vào cùng batch CNN MobileNetV3-Small. Server trả JSON, trình duyệt vẽ nhãn và đo độ trễ.',
        '', '## 2. Dữ liệu và phương pháp', '',
        f"Số ảnh gốc hợp lệ: {data['source_images']}, số nhóm ảnh độc lập sau gom trùng: {data['source_groups']}. Chia train/validation/test theo nhóm trước khi crop. Nhóm chứa ảnh RGB giống hệt hoặc dHash khác tối đa 4 bit. Bộ dữ liệu không có mã danh tính nên vẫn có khả năng cùng người xuất hiện ở các ảnh khác nhau.",
        '', '| Lớp | Train | Validation | Test |', '|---|---:|---:|---:|']
    for label in classes:lines.append('| '+label+' | '+' | '.join(str(data['counts'][s].get(label,0)) for s in ['train','val','test'])+' |')
    lines += ['', 'Loại khuôn mặt có cạnh dưới 12 pixel. Augmentation chỉ ở train: lật ngang, affine nhẹ, thay đổi màu/độ sáng. WeightedRandomSampler cân bằng lớp đeo sai ít mẫu. Không tính các ảnh augmentation là mẫu gốc độc lập.',
        '', '## 3. CNN và huấn luyện GPU', '',
        f"Kiến trúc: {evaluation['architecture']}. Số tham số: {evaluation['parameter_count']:,}. Đầu vào RGB 160×160, đầu ra 3 logits. Trọng số đầu vào pretrained ImageNet, fine-tune toàn bộ CNN.",
        '', f"Thiết bị thực train: **{evaluation['device']}**. PyTorch {evaluation['torch_version']}, CUDA {evaluation['cuda_version']}. Batch size {evaluation['batch_size']}, {evaluation['epochs']} epoch, seed {evaluation['seed']}. Tổng thời gian train và đánh giá: {evaluation['training_seconds']:.1f} giây. Chọn checkpoint epoch {evaluation['best_epoch']} bằng validation macro F1, không dùng test chọn checkpoint.",
        '', 'Optimizer AdamW, LR ban đầu 0.0003, weight decay 0.001, cosine decay, cross entropy label smoothing 0.05. Mixed precision khi chạy CUDA.',
        '', '## 4. Kết quả phân loại trên ảnh crop test', '',
        f"**Accuracy: {report['accuracy']*100:.2f}%**. **Macro F1: {report['macro avg']['f1-score']:.4f}**. Đây là chất lượng CNN trên ảnh cắt bằng nhãn chuẩn, chưa gồm lỗi bộ phát hiện.",
        '', '| Lớp | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    for label in classes:
        r=report[label];lines.append(f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
    lines+=['','Confusion matrix theo thứ tự lớp: '+', '.join(classes)+'. Hàng = nhãn thật, cột = dự đoán.','', '```text',*[' '.join(f'{n:5}' for n in row) for row in evaluation['confusion_matrix']],'```','', '![Learning curves và confusion matrix](../artifacts/training_results.png)',
            '', 'Lớp đeo sai có ít mẫu test nên ước lượng Recall/F1 còn không chắc chắn. Tăng dữ liệu người thật/góc mặt/ánh sáng của lớp này trước khi sử dụng ngoài demo.',
            '', '## 5. Xuất model một file', '',
            f"File `models/mask_detector.bundle`: {export['size_bytes']/1024/1024:.2f} MiB. Chứa CNN ONNX, YuNet detector và metadata, backend nạp bằng một đường dẫn duy nhất. Đối chiếu PyTorch–ONNX trên batch thử, sai số tuyệt đối tối đa {export['onnx_max_abs_error']:.8f}; kiểm tra batch động 1, 3 và 8 khuôn mặt.",
            '', f"SHA256: `{export['sha256']}`.",
            '', '## 6. Đánh giá cả pipeline trên ảnh gốc test', '']
    path=ROOT/'artifacts/pipeline_evaluation.json'
    if path.exists():
        pipe=json.loads(path.read_text())
        lines += [f"Đã chạy trên {pipe['source_images']} ảnh test gốc. Ghép dự đoán–ground truth bằng IoU ≥ 0,5. Detector precision {pipe['detector_precision']:.4f}, detector recall {pipe['detector_recall']:.4f}. Macro F1 cả pipeline {pipe['macro_f1']:.4f}. Đây không phải mAP.",
            '', '| Lớp | Precision pipeline | Recall pipeline | F1 pipeline |', '|---|---:|---:|---:|']
        for label in classes:
            r=pipe['per_class_end_to_end'][label];lines.append(f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
        lines+=['',f"Số ảnh tự nhiên test được tìm thấy có dự đoán đúng ít nhất một khuôn mặt của cả ba lớp ở cùng một lần chạy: {pipe['three_class_natural_frames_found']}. Ảnh được chọn trình diễn không được dùng thay cho đánh giá toàn tập.",
                '', 'YuNet giữ tỉ lệ ảnh với cạnh dài 640 pixel, ngưỡng 0,7 được chọn qua validation. Pipeline được cải tiến sau khi xem kết quả SSD baseline trên tập test này, vì vậy số đo pipeline mang tính đánh giá kỹ thuật khám phá, không phải benchmark hoàn toàn độc lập. Việc chọn checkpoint CNN vẫn chỉ dùng validation; số đo crop classifier test được tách riêng.']
    else:lines+=['Chưa có kết quả pipeline thực. Không suy diễn từ accuracy crop.']
    lines+=['','## 7. Đo độ trễ và cloud','',
        'Giao diện đo suy luận server (YuNet detect + CNN run), RTT từ gửi ảnh đến đọc JSON và toàn trình từ lấy khung hình đến kết quả vẽ xong/chờ paint. Mẫu warm-up được gắn cờ riêng, mean/P95 chỉ tính mẫu ổn định. Đồng hồ server và trình duyệt tách biệt. Toàn trình không gồm thời gian cảm biến camera trước khi ứng dụng lấy frame.',
        '', 'Render Free chạy backend ONNX CPU, giao diện và file model đóng gói trong cùng Docker image. Camera là camera laptop của người truy cập trang. PostgreSQL là tùy chọn, không lưu ảnh.',
        '', 'Bảng số đo cloud phải lấy từ CSV thực sau khi deploy và chạy camera/ảnh. Nếu chưa có URL cloud Live hoặc CSV browser, phần nghiệm thu local–cloud/toàn trình chưa hoàn tất. Không gọi benchmark localhost là cloud.',
        '', '## 8. Giới hạn và hướng cải thiện','',
        'Khuôn mặt nhỏ, nghiêng, thiếu sáng hoặc che khuất có thể bị YuNet bỏ sót; CNN không thể phân loại khuôn mặt chưa được phát hiện. Dữ liệu đeo sai ít và miền dữ liệu camera thực khác ảnh Internet. Ngưỡng confidence không phải thước đo độ chắc chắn đã hiệu chỉnh. Cloud miễn phí có cold start và giới hạn tài nguyên. Cần demo nhóm người thật đủ ba trạng thái để kiểm tra thêm.',
        '', '## 9. Tài liệu tham khảo','',
        'Slide thầy cung cấp và bài báo Đàm Minh Lịnh và cộng sự (2023), DOI 10.54607/hcmue.js.20.11.3933(2023). Nguồn dữ liệu/model/tài liệu triển khai chi tiết trong [SOURCES.md](SOURCES.md). Kết quả ở báo cáo này được tạo từ artifact thực, không sao chép số liệu 99,94% của bài mẫu.']
    verification=ROOT/'artifacts/verification.json'
    latency=ROOT/'artifacts/docker_local_latency.json'
    if verification.exists() and latency.exists():
        check=json.loads(verification.read_text());timing=json.loads(latency.read_text())
        lines+=['','## 10. Kiểm tra triển khai local Docker','',
            f"Đã đạt {check['tests']['total']} kiểm tra, gồm ảnh nhóm thật nén JPEG chất lượng 88 với đủ ba lớp đúng theo IoU ≥ 0,5, batch 3/20 khuôn mặt, file model tải xuống và ảnh lỗi. Docker build thành công. Backend chạy với giới hạn {check['docker_memory_limit_bytes']/1024/1024:.0f} MiB RAM, đỉnh bộ nhớ đo được {check['docker_peak_memory_bytes']/1024/1024:.1f} MiB, không bị OOM.",
            '',f"Đo 30 lượt ổn định và 1 warm-up trên **local Docker**, ảnh mẫu có {json.loads((ROOT/'artifacts/demo_sample.json').read_text())['result']['face_count']} khuôn mặt. Mỗi lượt đều trả về ít nhất một khuôn mặt thuộc cả ba lớp. Suy luận trung bình {timing['inference_ms']['mean']:.2f} ms (P95 {timing['inference_ms']['p95']:.2f} ms); RTT trung bình {timing['round_trip_ms']['mean']:.2f} ms (P95 {timing['round_trip_ms']['p95']:.2f} ms).",
            '', 'Các số trên là local Docker, chưa phải local–cloud. CLI không đo phần lấy ảnh/vẽ trên trình duyệt nên không gán số giả cho toàn trình. Phải dùng nút Đo tự động 30 mẫu hoặc camera trên giao diện sau khi cloud Live để lấy đủ ba độ trễ. Camera vật lý chưa được xác nhận.']
    browser_latency=ROOT/'artifacts/browser_local_latency.json'
    if browser_latency.exists():
        browser=json.loads(browser_latency.read_text())
        lines+=['','## 11. Kiểm tra giao diện và độ trễ toàn trình local','',
            'Đã chạy nút Đo tự động 30 mẫu trong trình duyệt trên ảnh nhóm công khai, gồm 1 warm-up và 30 mẫu ổn định. Cả 31 lần đều nhận dạng đủ ba lớp. Kiểm tra được hiển thị box, bộ đếm, ba số đo và nút xuất CSV. Đây là ảnh mẫu cố định, không phải webcam vật lý và không phải cloud.',
            '', '| Số đo trình duyệt local | Trung bình (ms) | P95 (ms) |','|---|---:|---:|']
        for key,label in [('inference_ms','Suy luận backend'),('round_trip_ms','Phản hồi local–local'),('end_to_end_ms','Toàn trình trình duyệt')]:
            lines.append(f"| {label} | {browser[key]['mean']:.2f} | {browser[key]['p95']:.2f} |")
        lines+=['','CSV gốc: `artifacts/browser_local_latency.csv`.','', '![Giao diện và kết quả đo thực](../artifacts/browser_local_verified.jpg)']
    (ROOT/'docs/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
