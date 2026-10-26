-- Chay tren ket noi Render MaskLab, database masklab.
-- Cac cau lenh chi doc du lieu, khong thay doi bang.

SELECT current_database() AS database_name,
       current_schema() AS schema_name,
       version() AS postgresql_version;

-- 50 so do moi nhat; hien thi theo gio Viet Nam.
SELECT id,
       created_at AT TIME ZONE 'Asia/Ho_Chi_Minh' AS thoi_gian_viet_nam,
       ROUND(inference_ms::numeric, 1) AS suy_luan_ms,
       ROUND(round_trip_ms::numeric, 1) AS phan_hoi_ms,
       ROUND(end_to_end_ms::numeric, 1) AS toan_trinh_ms,
       (counts->>'with_mask')::int AS deo_dung,
       (counts->>'without_mask')::int AS khong_deo,
       (counts->>'mask_weared_incorrect')::int AS deo_sai,
       (counts->>'uncertain')::int AS chua_chac_chan,
       request_id,
       model_sha256
FROM public.lab_measurements
ORDER BY id DESC
LIMIT 50;

-- Tong so ban ghi, bao gom warm-up va tat ca cac phien.
SELECT COUNT(*) AS tong_so_ban_ghi,
       MIN(created_at) AT TIME ZONE 'Asia/Ho_Chi_Minh' AS lan_dau,
       MAX(created_at) AT TIME ZONE 'Asia/Ho_Chi_Minh' AS moi_nhat
FROM public.lab_measurements;
