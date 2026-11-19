# N16 — Phát hiện buồn ngủ từ EEG đơn kênh NeuroSky

Đồ án 1, học kỳ 1 năm học 2026–2027, nhóm N16. Kho mã phục vụ theo dõi tiến độ và kiểm tra các mô-đun phần mềm của hệ thống phát hiện buồn ngủ từ EEG đơn kênh.

Mục tiêu là kết hợp **đặc trưng phổ EEG và đặc trưng chớp mắt từ raw**, xây dựng mô hình phân loại tỉnh/buồn ngủ và triển khai cảnh báo âm thanh. Nhãn dữ liệu tự thu dựa trên KSS theo quy trình thí nghiệm. Thiết bị sử dụng là NeuroSky với raw danh định 512 Hz.

## Bản Windows và mô hình cập nhật ngày 17/11/2026

Tải ứng dụng trong `releases/N16_KSS_20261117/N16_NhanBietBuonNgu_KSS_20261117.exe`. Mở ứng dụng, chọn cổng COM và nhấn **Bắt đầu**, hoặc dùng **Chọn CSV để kiểm tra** để phát lại raw OpenViBE.

Bản này dùng SVM RBF với 9 đặc trưng phổ, cửa sổ 3 giây và bước 0,25 giây. Ngưỡng chính là 0,60, kèm 6 lần liên tiếp đạt từ 0,575 và 2 lần liên tiếp đạt từ 0,60 để xác nhận cảnh báo. Tập phát triển gồm 4.129 cửa sổ của 5 người. LOSO lồng nhau đạt 60,51% balanced accuracy. Cấu hình được chọn trên tập phát triển đạt 63,26%. Nhãn cũ còn giả định thời gian danh định và dữ liệu hiện tại cần được bổ sung để đánh giá độc lập.

Ứng dụng đã được kiểm tra khởi động và đối chiếu phát lại CSV với Python. Trên fixture giả lập của bản này, lệnh báo đầu tiên xuất hiện sau 4.998 ms từ mốc chuyển trạng thái. Phép đo Bluetooth đến âm thanh thực tế và kiểm tra chạy bền còn cần hoàn thiện. Cảnh báo nhầm ở các phiên tỉnh vẫn xuất hiện.

Sau khi cài các gói trong `pyproject.toml`, có thể kiểm tra bản chạy từ mã nguồn bằng PowerShell tại thư mục kho mã.

```powershell
.\.venv\Scripts\python.exe scripts/n16_live_app.py --release-dir releases/N16_KSS_20261117 --check
```

Đóng gói lại bằng Python có PyInstaller và các thư viện của dự án.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_live_exe.ps1 -ReleaseDirectory releases/N16_KSS_20261117 -Name N16_NhanBietBuonNgu_KSS_20261117
```

Thư mục phát hành chứa EXE, mô hình JSON, cấu hình và kết quả kiểm tra phần mềm. Các script xử lý dữ liệu nghiên cứu dùng các bản ghi và hồ sơ gán nhãn lưu cục bộ. Kiểm thử mã nguồn ngày 19/11/2026 đạt 83 bài, bỏ qua 1 bài.

## Phương án demo ngày 10/11/2026

Python chạy từ thu tín hiệu đến cảnh báo. Mô hình chính là **Logistic Regression dùng phổ FFT và blink từ raw**, huấn luyện trên cửa sổ 3 giây. Hệ cập nhật mỗi 0,25 giây, báo khi điểm nguy cơ đạt **0,50 trong ba lần liên tiếp** và kết thúc đợt báo ở lần dự đoán dưới ngưỡng. Cửa sổ tín hiệu kém xóa trạng thái xác nhận. Bộ đếm blink mặc định cung cấp đặc trưng cho mô hình này.

Chạy gói cục bộ đã chốt từ PowerShell tại thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_n16.ps1 -Sound
```

Gói `output/N16_Chot_20261110` lưu mô hình, cấu hình, fixture giả lập và SHA-256. Thêm `-Port COM4 -Sound` để dùng NeuroSky, thay COM4 bằng cổng thực tế. `-Variant Spectral` mở bản đối chứng chỉ phổ. `-Variant LowRepeat` mở bản thử giảm số đợt báo lặp. Hai biến thể dùng cùng cửa sổ 3 giây. Trên bản sao mới của kho mã, chạy `scripts/run_demo.ps1 -NoGui` để kiểm tra phần mềm bằng dữ liệu giả lập tự sinh.

### Ứng dụng Windows nhận biết trực tiếp

Mở `output/N16_Live_Windows/N16_NhanBietBuonNgu.exe`, chọn cổng COM của NeuroSky và nhấn **Bắt đầu**. Ứng dụng đóng gói sẵn mô hình Main và thư viện chạy. Khi dùng cùng tai nghe, đóng OpenViBE hoặc bộ ghi đang giữ cổng COM trước khi mở kết nối. Nút **Thử loa** kiểm tra âm thanh, còn **Mở thư mục log** mở các tệp kết quả đã lưu.

Bản `N16_NhanBietBuonNgu_CSV.exe` có nút **Chọn CSV để kiểm tra**. Chọn một phiên raw OpenViBE (`Time:512Hz`, `Electrode`) hoặc N16 (`raw_eeg` và trục thời gian 512 Hz). Chế độ CSV dự đoán ngoại tuyến và lưu điểm, trạng thái và cảnh báo của từng cửa sổ. Với OpenViBE, dùng cùng cấu hình xuất raw như dữ liệu huấn luyện. Các file thiếu `poor_signal` vẫn chạy được với thông báo chất lượng chưa xác minh. Kết quả hiển thị là dự đoán của mô hình, cần nhãn tham chiếu để đánh giá độ chính xác.

Bản `N16_NhanBietBuonNgu_CSV_KiemTinHieu.exe` thêm sàng lọc ngoại tuyến cho raw count. Cửa sổ có mẫu chạm hoặc vượt biên dự kiến `[-2048, 2047]`, hoặc có đoạn đứng giá trị liên tục từ 0,25 giây, được gắn `raw_artifact` và xóa xác nhận cảnh báo. Lọc này chỉ áp dụng cho chế độ nạp CSV.

Bản `N16_NhanBietBuonNgu_054.exe` áp dụng cấu hình LowRepeat cho cả kết nối trực tiếp và nạp CSV. Cảnh báo bật sau ba dự đoán liên tiếp từ 0,54 và tắt sau ba dự đoán liên tiếp dưới ngưỡng hoặc khi tín hiệu bị loại. Bản này vẫn có sàng lọc raw khi nạp CSV.

Để đóng gói lại từ Windows, dùng môi trường `.venv` có NumPy, PyInstaller và pyserial. Script lấy manifest và mô hình LowRepeat từ thư mục bản chốt cục bộ.

```powershell
.\.venv\Scripts\python.exe -m pip install "pyinstaller>=6,<7" "pyserial==3.5"
powershell -ExecutionPolicy Bypass -File .\scripts\build_live_exe.ps1
```

## Phạm vi triển khai

| Thành phần | Nội dung đã có trong mã nguồn | Phạm vi đã kiểm tra |
|---|---|---|
| Thu nhận | ThinkGear qua COM, nhập CSV OpenViBE, chế độ ghi ba dấu video–EEG | Parser và gắn dấu đã qua thử nghiệm phần mềm. Dấu cùng loa và video cần thử với thiết bị thật |
| Tiền xử lý | Nhánh FFT 0,5–45 Hz; công cụ lọc SOS ngoại tuyến và sàng lọc vùng lỗi | Kiểm thử tín hiệu giả lập và xử lý các bản ghi pilot; giữ nguyên raw gốc |
| Đặc trưng | Công suất băng tần, tỷ số công suất, entropy; thuật toán nền ước lượng chớp mắt | Kiểm thử phần mềm; chưa xác nhận độ chính xác chớp mắt bằng nguồn đối chiếu độc lập |
| Học máy | Logistic Regression, SVM và Random Forest trong Pipeline scikit-learn; chuẩn hóa trong mỗi fold | Đã chạy trên 3.339 cửa sổ EEG của S01 và S02 với nhãn KSS tạm |
| Đánh giá | Leave-One-Group-Out theo người; ma trận nhầm lẫn, macro-F1, balanced accuracy và AUC | Hai lượt LOSO NeuroSky. SEED-VIG có thực nghiệm riêng chia theo tiền tố tệp, cần xác nhận mã người |
| Giao diện | Nạp raw, kiểm tra CSV, tạo đặc trưng theo khoảng nhãn, huấn luyện và nhập SEED-VIG | Giao diện Tkinter bốn tab đã qua kiểm tra khởi tạo |
| Runtime | Python cho bản chính 3 giây, C++17 cho bản đối chứng 4 giây | Đã kiểm tra phát lại. Phép đo Bluetooth đến âm thanh thật vẫn cần thực hiện |

## Kiến trúc phần mềm

- **Python:** chuẩn bị dữ liệu, huấn luyện, đánh giá và chạy demo trực tiếp. Logistic của nhánh FFT xuất trọng số, scaler và ngưỡng sang JSON. Các Pipeline nghiên cứu khác lưu `joblib`.
- **C++17:** nhánh triển khai đối chứng dùng mô hình FFT 4 giây. Phát lại trên cùng cấu hình 4 giây đã được dùng để kiểm tra kết quả suy luận giữa Python và C++.
- **Giao diện ML:** hỗ trợ kiểm tra file và tổ chức các bước chuẩn bị dữ liệu; không tự tạo nhãn KSS từ tên file.

Khi chạy một tệp mô hình, dùng đúng độ dài cửa sổ và cách tính đặc trưng lúc huấn luyện. Phép đối chiếu Python–C++ đã thực hiện trên nhánh FFT 4 giây. Mô hình FFT 3 giây trong gói chốt dùng launcher Python.

## Trạng thái dữ liệu và thực nghiệm

Cập nhật ngày **12/11/2026**, theo hồ sơ xử lý hiện có.

- Bản chính dùng **3.339 cửa sổ 3 giây**, cùng mốc kết thúc và nhãn tạm của S01/S02. Đạt **57,29% balanced accuracy** qua hai lượt LOSO. Bản chỉ phổ đạt 57,65%. S03 được dành cho kiểm tra người mới khi đủ cả hai trạng thái.
- Bộ blink mặc định đạt **74,07% và 82,61% F1** trên hai phiên S02 theo cách căn đỉnh EEG đầu phiên. Ứng viên học từ dữ liệu công khai đạt **72,73% và 88,37%** với cùng phép căn ước lượng. Phiên BLINK_03 dùng bộ ghi cũ, với trục mẫu EEG ngắn hơn video khoảng 9 giây và độ lệch thay đổi giữa phiên. Ảnh mốc của thiết bị hỗ trợ rà soát, còn phép đối chiếu 90% cần video có dấu đồng bộ độc lập.
- Bản demo ngưỡng 0,50 báo dưới 3 giây ở 24/24 ca giả lập đã giữ để kiểm tra. Phát lại có chờ theo nhịp thật trên laptop mất **2,500 giây** đến lệnh báo ở ca gốc. Ứng viên ngưỡng 0,54 có một ca giả lập đo theo đồng hồ đạt **2,750 giây**. Các số này đo khi bộ đệm đã đủ mẫu.
- Đã chạy thăm dò riêng trên **23 phiên SEED-VIG** với nhãn PERCLOS. Phân loại hai vùng PERCLOS cực đạt **70,67%** với EEG vùng trán và **93,79%** khi thêm EOG, tính balanced accuracy trên 16 nhóm tiền tố có đủ hai lớp. Ánh xạ tiền tố sang người cần xác nhận.
- Nhánh FFT 4 giây đối chứng đã chạy 21 quyết định C++ khớp Python trên dữ liệu giả lập. Điểm LOSO thăm dò của bản C++ có trọng số cố định là **57,19%**.

### Kiểm tra gián đoạn với bộ ghi Python

Khi máy tính không nhận raw EEG trong ít nhất 0,5 giây, bản chạy trực tiếp ghi `MAT_TIN_HIEU` và khởi tạo lại cửa sổ cùng bộ đếm xác nhận. Bộ ghi lưu thời điểm gián đoạn vào `.capture.json` và đánh dấu đoạn trong cột `raw_segment`. Ứng dụng ML bỏ các cửa sổ bắc qua hai đoạn. Đối chiếu mốc KSS với trục mẫu trước khi tạo đặc trưng, vì `signal_time_ms` tính theo chỉ số mẫu nhận được và không cộng thời gian gián đoạn.

Để thử thu một phiên blink mới trên Windows, thay `COM4` bằng cổng thực tế và chọn tên file mới.

```powershell
.\.venv\Scripts\python.exe -m n16_eeg.acquire --port COM4 --subject S02 --session BLINK_04 --output .\raw\S02_BLINK_04.csv --duration-min 5 --sync-markers
```

Bật video có tiếng, đợi EEG chạy rồi nhấn `M` một lần ở đầu, giữa và cuối phần cần chấm. Nhấn `Q` để lưu CSV và `.capture.json`. Ghi thời điểm ba tiếng dấu vào CSV `event,video_timestamp_ms` với mã `SYNC_01` đến `SYNC_03`. Ghi các lần chớp mắt quan sát được vào CSV `timestamp_ms` theo đồng hồ video. Cả hai tệp dùng mili giây.

```powershell
.\.venv\Scripts\python.exe .\scripts\align_blink_sync_markers.py .\raw\S02_BLINK_04.csv .\raw\video_markers.csv .\raw\video_blinks.csv .\output\blink04_check
```

Công cụ căn theo hai dấu ngoài và kiểm tra dấu giữa. Nếu có gián đoạn raw trong khoảng đó hoặc sai số dấu vượt ngưỡng, kết quả để trống điểm blink. Giám sát này dựa trên thời điểm máy tính nhận mẫu, nên một khoảng ngắt ngắn hoặc mất mẫu rải rác vẫn có thể nằm dưới ngưỡng phát hiện.

### Nội dung cần hoàn thiện

1. Hoàn tất quy tắc nhãn và căn thời gian, tạo tập đặc trưng có nhãn từ dữ liệu đạt chất lượng.
2. Kiểm tra mô hình đã chốt trên một người mới có dữ liệu hai trạng thái và mốc KSS đối chiếu được.
3. Đối chiếu từng sự kiện chớp mắt với nguồn tham chiếu độc lập để kiểm tra tiêu chí **≥90% so với đếm tay**.
4. Đo từ mốc tham chiếu đã thống nhất đến lúc loa thực sự phát âm khi dùng NeuroSky qua Bluetooth, để kiểm tra **độ trễ cảnh báo <3 giây**.

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
