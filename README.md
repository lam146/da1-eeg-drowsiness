# N16 — Phát hiện buồn ngủ từ EEG đơn kênh NeuroSky

Đồ án 1, học kỳ 1 năm học 2026–2027, nhóm N16. Kho mã phục vụ theo dõi tiến độ và kiểm tra các mô-đun phần mềm của hệ thống phát hiện buồn ngủ từ EEG đơn kênh.

Mục tiêu là kết hợp **đặc trưng phổ EEG và đặc trưng chớp mắt từ raw**, xây dựng mô hình phân loại tỉnh/buồn ngủ và triển khai cảnh báo âm thanh. Nhãn dữ liệu tự thu dựa trên KSS theo quy trình thí nghiệm. Thiết bị sử dụng là NeuroSky với raw danh định 512 Hz.

## Phạm vi triển khai

| Thành phần | Nội dung đã có trong mã nguồn | Phạm vi đã kiểm tra |
|---|---|---|
| Thu nhận | Parser ThinkGear, kiểm tra checksum, thu qua COM; nhập CSV OpenViBE | Kiểm thử parser; đã đọc các phiên pilot thu qua OpenViBE. Chưa coi đó là kiểm chứng parser riêng trên thiết bị |
| Tiền xử lý | Nhánh FFT 0,5–45 Hz; công cụ lọc SOS ngoại tuyến và sàng lọc vùng lỗi | Kiểm thử tín hiệu giả lập và xử lý các bản ghi pilot; giữ nguyên raw gốc |
| Đặc trưng | Công suất băng tần, tỷ số công suất, entropy; thuật toán nền ước lượng chớp mắt | Kiểm thử phần mềm; chưa xác nhận độ chính xác chớp mắt bằng nguồn đối chiếu độc lập |
| Học máy | Logistic Regression, SVM và Random Forest trong Pipeline scikit-learn; chuẩn hóa trong mỗi fold | Đã kiểm thử quy trình huấn luyện, lưu mô hình và dự đoán ngoài fold bằng fixture giả lập |
| Đánh giá | Leave-One-Group-Out với nhóm là người; confusion matrix, macro-F1, balanced accuracy và AUC khi xác định được | Chưa có kết quả phân loại thực nghiệm được xác nhận trên dữ liệu tự thu hoặc SEED-VIG |
| Giao diện | Nạp raw, kiểm tra CSV, tạo đặc trưng theo khoảng nhãn, huấn luyện và nhập SEED-VIG | Giao diện Tkinter bốn tab đã qua kiểm tra khởi tạo |
| Runtime C++17 | COM, ThinkGear, DSP, suy luận theo mô hình JSON và cảnh báo | Đã biên dịch và chạy self-test; chưa chứng minh tiêu chí độ trễ cảnh báo trên thiết bị |

## Kiến trúc phần mềm

- **Python:** chuẩn bị dữ liệu, xử lý ngoại tuyến, huấn luyện và đánh giá. Nhánh scikit-learn lưu Pipeline dưới dạng `joblib`.
- **C++17:** mô-đun thu nhận và suy luận trực tiếp; hiện dùng định dạng JSON từ nhánh logistic riêng.
- **Giao diện ML:** hỗ trợ kiểm tra file và tổ chức các bước chuẩn bị dữ liệu; không tự tạo nhãn KSS từ tên file.

Hai nhánh mô hình cần được phân biệt: Pipeline scikit-learn **chưa tự xuất thành mô hình JSON của runtime C++**. Bộ lọc SOS hai chiều ngoại tuyến cũng khác nhánh FFT trong ứng dụng. Khi chuyển mô hình sang chạy trực tiếp, cần thống nhất bộ lọc, thứ tự đặc trưng, đơn vị, scaler và kiểm tra đầu ra giữa các bản triển khai.

## Trạng thái dữ liệu và thực nghiệm

Cập nhật ngày **03/11/2026**, theo hồ sơ xử lý hiện có:

- Đã thu sáu phiên pilot của người S01. Sau sàng lọc kỹ thuật và tách vùng biên lọc, còn **1.291 cửa sổ ứng viên 4 giây**, tương đương **86 phút 4 giây**.
- Số cửa sổ trên chưa phải số mẫu huấn luyện sạch có nhãn đã xác nhận. Còn căn mốc KSS, xem các vùng nghi nhiễu và thống nhất quy tắc khoảng nhãn; phiên 06 có chênh lệch thời lượng raw và nhật ký chưa giải thích được.
- Đã kiểm tra file đặc trưng và nhãn PERCLOS của **23 phiên SEED-VIG** trên máy cục bộ. Chưa xác nhận đầy đủ ánh xạ người của bản phát hành và chưa chốt ngưỡng phân loại; chưa chạy đánh giá thực nghiệm trên bộ này.
- Chưa có dữ liệu người thứ hai được xác nhận trong hồ sơ hiện tại. Sáu phiên S01 không thay thế sáu người và chưa đủ để làm LOSO theo người trên dữ liệu tự thu.

### Nội dung cần hoàn thiện

1. Hoàn tất quy tắc nhãn và căn thời gian, tạo tập đặc trưng có nhãn từ dữ liệu đạt chất lượng.
2. Bổ sung dữ liệu theo người và thực hiện đánh giá chéo theo người; báo cáo kết quả từng fold và các giới hạn.
3. Đối chiếu từng sự kiện chớp mắt với nguồn tham chiếu độc lập để kiểm tra tiêu chí **≥90% so với đếm tay**.
4. Đối chiếu mô hình Python–C++, kiểm tra suy luận trực tiếp và đo **độ trễ cảnh báo <3 giây** theo mốc đo đã thống nhất.

Các chỉ số từ dữ liệu giả lập chỉ xác nhận hoạt động phần mềm, không dùng làm accuracy/F1 của báo cáo thực nghiệm. Chưa đánh dấu các tiêu chí chớp mắt hoặc độ trễ là đã đạt.

## Kiểm tra nhanh mã nguồn

### Yêu cầu môi trường

Python 3.10 trở lên có Tkinter, PowerShell trên Windows; nếu biên dịch C++ bằng script thì cần `g++` hỗ trợ C++17. Các gói Python được khai báo trong `pyproject.toml`.

Trong PowerShell tại thư mục kho mã:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m n16_eeg.ml_gui --check
```

Mở ứng dụng nạp dữ liệu:

```powershell
.\Mo_UngDung_ML.cmd
```

Kiểm tra toàn luồng bằng dữ liệu **giả lập**, không cần dữ liệu cá nhân hoặc SEED-VIG:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1 -NoGui
```

Biên dịch và kiểm tra runtime C++:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_cpp.ps1
```

Script build gọi self-test sau khi biên dịch. Để chạy thêm phát lại C++ trên dữ liệu giả lập:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_cpp_demo.ps1
```

Lần kiểm tra ngày **03/11/2026**: **66 kiểm thử Python đạt**, `pip check` không phát hiện xung đột, giao diện bốn tab khởi tạo được, C++ biên dịch và self-test đạt. Đây là kết quả kiểm tra phần mềm, không phải kết quả đánh giá phát hiện buồn ngủ trên người.

## Cấu trúc kho mã

| Đường dẫn | Vai trò |
|---|---|
| `src/n16_eeg/thinkgear.py`, `acquire.py` | Giải mã và thu dữ liệu ThinkGear |
| `src/n16_eeg/preprocess.py`, `signal_features.py`, `blink.py` | Xử lý tín hiệu và tính đặc trưng |
| `src/n16_eeg/ml_gui.py`, `ml_workbench.py` | Giao diện và chuẩn bị dữ liệu có kiểm tra nguồn |
| `src/n16_eeg/seed_vig.py`, `openvibe_import.py` | Bộ nhập theo định dạng nguồn |
| `src/n16_eeg/sklearn_train.py` | Pipeline scikit-learn và đánh giá theo người |
| `src/n16_eeg/model.py`, `train.py`, `workflow.py` | Nhánh logistic/JSON và quy trình demo |
| `cpp/` | Runtime C++17 và cấu hình CMake |
| `scripts/` | Cài đặt, build, demo và xử lý dữ liệu cục bộ |
| `tests/` | Kiểm thử đơn vị và tích hợp với fixture giả lập |
| `configs/seed_vig_manifest.example.json` | Cấu hình mẫu có placeholder; chưa phải cấu hình dữ liệu dùng ngay |

Các script xử lý pilot tham chiếu file đo trên máy nghiên cứu; muốn chạy lại cần cung cấp dữ liệu và metadata tương ứng. Kho mã không chứa các file đó. Quy trình demo giả lập ở trên có thể dùng để kiểm tra phần mềm riêng.

## Nguồn kỹ thuật

- [ThinkGear Communications Protocol](https://developer.neurosky.com/docs/doku.php?id=thinkgear_communications_protocol): định dạng raw và giải mã dữ liệu.
- [SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html): dữ liệu tham khảo mức tỉnh táo; xử lý riêng với nhãn KSS của NeuroSky.
- [scikit-learn](https://scikit-learn.org/stable/): Pipeline, các mô hình nền và đánh giá theo nhóm.
