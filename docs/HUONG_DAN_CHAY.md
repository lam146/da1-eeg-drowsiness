# Hướng dẫn chạy phần mềm N16

## 1. Chuẩn bị lần đầu

Mở PowerShell tại `D:\dh\da1` và chạy:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

Script tạo `.venv`, cài gói dự án và thư viện kiểm thử. Không cần cài scikit-learn hay SciPy.

## 2. Demo khi chưa có thiết bị

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1
```

Quy trình tự động:

1. Tạo 6 người mô phỏng, mỗi người 2 phiên.
2. Chia cửa sổ 4 giây, bước 1 giây và trích đặc trưng.
3. Đánh giá LOSO theo subject.
4. Huấn luyện mô hình cuối và lưu JSON.
5. Phát lại một phiên, ghi cảnh báo và độ trễ.
6. Kiểm tra đầu vào rồi mở giao diện.

Trong giao diện, nhấn **Bắt đầu**. Trạng thái `TINH`, `NGUY_CO`, `CANH_BAO` được cập nhật theo từng cửa sổ. Nút **Chọn CSV** cho phép chọn một phiên khác có cùng schema.

## 3. Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m n16_eeg.gui --check
```

## 4. Schema CSV raw

Các cột tối thiểu:

- `timestamp_ms`: thời gian mẫu tính bằng mili giây.
- `subject_id`, `session_id`: mã người và phiên.
- `raw_eeg`: mẫu EEG thô.
- `poor_signal`: 0 là tốt; giá trị cao là tiếp xúc kém.
- `kss`: nhãn 1–9; KSS 1–5 là tỉnh, 7–9 là buồn ngủ, 6 bị loại khỏi baseline nhị phân.
- `event`: ghi mốc thay đổi trạng thái hoặc sự kiện.

Các cột `attention`, `meditation` có thể để trống và chỉ dùng tham khảo.

## 5. Quy trình với dữ liệu thật

1. Ghép đôi MindWave Mobile 2 qua Bluetooth.
2. Cài extra `hardware` và tìm cổng bằng `python -m n16_eeg.acquire --list-ports`.
3. Thu từng người vào một CSV riêng.
4. Gán KSS đúng thời điểm theo protocol; không dùng dữ liệu mô phỏng để thay thế.
5. Chạy `n16-featurize` cho từng phiên rồi ghép các bảng feature.
6. Chạy `n16-train` trên ít nhất 3 người; nên đạt kế hoạch tối thiểu 8 người × 2 phiên.
7. Kiểm tra metrics từng fold, lỗi chớp mắt so với nhãn tay và p95 độ trễ.

Chạy cảnh báo trực tiếp sau khi đã có mô hình:

```powershell
python -m n16_eeg.live --port COM5 --model output/model/n16_logistic_model.json
```

File nhãn tay cho chớp mắt chỉ cần cột `timestamp_ms`. Chấm với dung sai ±200 ms:

```powershell
python -m n16_eeg.blink_eval data/raw/S01_afternoon_01.csv `
  data/annotations/S01_blinks.csv
```

## 6. Giới hạn hiện tại

- Ngưỡng chớp mắt cần hiệu chỉnh bằng nhãn tay của chính thiết bị và vị trí đeo thực tế.
- Mô hình mô phỏng dễ phân tách hơn dữ liệu thật; điểm synthetic không chứng minh hiệu năng thực nghiệm.
- Cần bổ sung dữ liệu thật, bảng đối chiếu nhãn tay và biểu đồ kết quả trước mốc nghiệm thu.
