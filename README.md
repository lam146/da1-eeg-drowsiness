# N16 – Phát hiện buồn ngủ khi lái xe từ EEG NeuroSky

## Tài liệu học ML

- [Học ML từ con số 0 cho người mới — PDF](docs/learning/Hoc_ML_Tu_Con_So_0_Cho_Nguoi_Moi.pdf): thuật ngữ, ví dụ đời thường và 12 bài tập có đáp án.
- [Tự học ML cho đồ án EEG N16 — PDF](docs/learning/N16_TuHoc_ML_Cho_DoAn_EEG.pdf): đặc trưng EEG, KSS, LOSO, mô hình và cách đọc kết quả.

Kho mã không kèm dữ liệu EEG cá nhân, file SEED-VIG, dữ liệu dẫn xuất, mô hình đã huấn luyện hoặc các bộ cài/tài liệu tham khảo tải về. Tải dữ liệu theo quyền sử dụng của nguồn và đặt tại `data/external/` trên máy. Các fixture trong kiểm thử là dữ liệu giả lập.

## Xử lý dữ liệu ngoại tuyến

`scripts/filter_pilot_offline.py` tạo bản lọc riêng và đánh dấu vùng lỗi, không ghi đè raw. `scripts/process_s01_pilots.py` xử lý sáu phiên S01 theo các đường dẫn cục bộ được khai báo trong script; các file đo/phiếu không nằm trong kho Git. Đặc trưng ứng viên chưa có nhãn không được coi là tập huấn luyện hoàn chỉnh. Bộ lọc hai chiều ngoại tuyến khác với nhánh FFT trong ứng dụng và không phải bằng chứng về độ trễ cảnh báo live.

```powershell
.\.venv\Scripts\python.exe scripts/filter_pilot_offline.py raw/ten_phien.csv output/ten_lan_xu_ly_moi
```

Các bài kiểm tra phần mềm không thay thế thực nghiệm phân loại, đối chiếu chớp mắt hoặc đo độ trễ trên thiết bị.

## Ứng dụng nạp CSV vào ML

Nhấp đúp **`Mo_UngDung_ML.cmd`** trong thư mục dự án để mở giao diện tiếng Việt.
Ứng dụng cho phép xem CSV/đồ thị raw, tạo đặc trưng từ khoảng nhãn đã kiểm tra,
và chọn Logistic Regression, SVM hoặc Random Forest để huấn luyện/đánh giá LOSO.
Xem [hướng dẫn từng bước](docs/HUONG_DAN_UNG_DUNG_ML.md).
File raw chưa có nhãn vẫn xem được; file thử chớp mắt không tự trở thành dữ liệu buồn ngủ.
Tab **4. SEED-VIG** cho phép xem biến MAT và nhập theo manifest sau khi được cấp quyền.
Đã kiểm tra dữ liệu MAT thật của 23 phiên ngày 28/10/2026; tab 4 hiển thị hồ sơ dữ liệu đã tải. Xem [dữ liệu thật và cấu hình cần hoàn thiện](docs/SEED_VIG_DU_LIEU_THAT_20261028.md). Chưa có kết quả huấn luyện trên dữ liệu này.
Xem [cách áp dụng đầy đủ ba nguồn GVHD](docs/SEED_VIG_VA_BA_NGUON_GVHD.md).

## Cập nhật theo tài liệu GVHD 23/10/2026

Đã bổ sung nhập CSV OpenViBE, nhập đặc trưng MAT SEED-VIG theo manifest,
và scikit-learn LOSO theo người (Logistic Regression, SVM, Random Forest).
Hướng dẫn, lệnh chạy, quy tắc nhãn và giới hạn nằm tại
[Áp dụng tài liệu GVHD](docs/AP_DUNG_TAI_LIEU_GVHD.md).
SEED-VIG/PERCLOS được đánh giá riêng, không coi là raw NeuroSky/KSS;
pipeline joblib mới chỉ dùng ngoại tuyến, không thay JSON runtime C++.

Mã nguồn MVP cho Đồ án 1 nhóm N16, HK1 2026–2027. Kiến trúc dùng Python cho huấn luyện/đánh giá ngoại tuyến và C++17 cho hệ thống chạy thời gian thực, phù hợp định hướng Kỹ thuật máy tính.

## Chức năng đã có

- Đọc gói ThinkGear qua Bluetooth COM, kiểm tra checksum và lưu raw EEG 512 Hz.
- Tiền xử lý 0,5–45 Hz, loại nhiễu điện lưới 50 Hz bằng FFT.
- Trích đặc trưng phổ delta/theta/alpha/beta/gamma, các tỷ số và spectral entropy.
- Phát hiện chớp mắt/EOG, độ rộng chớp và chớp kéo dài.
- Huấn luyện logistic regression không cần scikit-learn; lưu/đọc mô hình JSON.
- Đánh giá Leave-One-Subject-Out, đặt scaler bên trong từng fold để tránh rò rỉ dữ liệu.
- Phát lại gần thời gian thực, cảnh báo liên tiếp, ghi risk và độ trễ.
- Chạy trực tiếp từ luồng NeuroSky khi có thiết bị và phát cảnh báo âm thanh.
- Chấm precision/recall/F1 và sai số đếm chớp mắt so với nhãn tay.
- Giao diện Tkinter hiển thị trạng thái, nguy cơ, theta/alpha, số chớp và đồ thị.
- Workflow một lệnh để tạo dữ liệu mô phỏng, huấn luyện, đánh giá và kiểm tra demo.
- Runtime C++17 độc lập thực hiện COM, ThinkGear, DSP, suy luận và cảnh báo.

> Dữ liệu và chỉ số mô phỏng chỉ dùng kiểm thử phần mềm. Không dùng chúng làm kết quả accuracy/F1 của báo cáo thực nghiệm.

## Chạy nhanh khi chưa có thiết bị

Trong PowerShell tại thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1
```

Lệnh thứ hai tự tạo cohort giả lập, huấn luyện mô hình, kiểm tra toàn bộ pipeline rồi mở giao diện. Dùng `-NoGui` nếu chỉ muốn chạy kiểm tra dòng lệnh:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1 -NoGui
```

Build và chạy sản phẩm C++:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_cpp.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_cpp_demo.ps1
```

## Chạy từng bước

```powershell
.\.venv\Scripts\Activate.ps1
python -m n16_eeg.workflow --root .
python -m n16_eeg.gui
python -m pytest -q
```

Các tệp đầu ra chính:

- `output/model/n16_logistic_model.json`: mô hình đã lưu.
- `output/model/loso_metrics.json`: đánh giá LOSO, có cảnh báo dữ liệu synthetic.
- `output/demo/replay_model_log.csv`: nhật ký phát lại.
- `output/demo/workflow_summary.json`: tóm tắt kiểm tra toàn luồng.

## Khi có thiết bị NeuroSky

Cài thêm driver đọc cổng nối tiếp:

```powershell
python -m pip install -e .[hardware]
python -m n16_eeg.acquire --list-ports
```

Sau khi ghép đôi Bluetooth và biết cổng COM:

```powershell
python -m n16_eeg.acquire --port COM5 --subject S01 --session afternoon_01 `
  --duration-min 30 --output data/raw/S01_afternoon_01.csv
```

Bổ sung nhãn KSS vào cột `kss`, sau đó:

```powershell
python -m n16_eeg.featurize data/raw/S01_afternoon_01.csv data/processed/S01_features.csv
python -m n16_eeg.train data/processed/all_subjects_features.csv
python -m n16_eeg.demo data/raw/S01_afternoon_01.csv `
  --model output/model/n16_logistic_model.json --speed 1 --sound
python -m n16_eeg.live --port COM5 --model output/model/n16_logistic_model.json
```

Khi dùng runtime C++ cho sản phẩm cuối:

```powershell
.\cpp\build\n16_runtime.exe live --port COM5 `
  --model output\model\n16_logistic_model.json --duration-min 30
```

Chi tiết vận hành nằm trong `docs/HUONG_DAN_CHAY.md`; quy trình thu và an toàn nằm trong `docs/PROTOCOL_THU_DU_LIEU.md`.

## Cấu trúc chính

- `src/n16_eeg/acquire.py`: thu dữ liệu ThinkGear.
- `src/n16_eeg/preprocess.py`: lọc tín hiệu.
- `src/n16_eeg/blink.py`: phát hiện chớp mắt.
- `src/n16_eeg/signal_features.py`: đặc trưng phổ.
- `src/n16_eeg/model.py`: mô hình và LOSO.
- `src/n16_eeg/workflow.py`: pipeline tự động.
- `src/n16_eeg/gui.py`: giao diện demo.
- `src/n16_eeg/live.py`: suy luận trực tiếp từ thiết bị.
- `src/n16_eeg/blink_eval.py`: chấm phát hiện chớp mắt với nhãn tay.
- `tests/test_core.py`: kiểm thử đơn vị và tích hợp.
- `cpp/`: runtime C++17 cho triển khai thời gian thực.
- `docs/KIEN_TRUC_PYTHON_CPP.md`: cách build và giải thích giao tiếp Python–C++.

Không thử nghiệm khi đang lái xe thật. Chỉ dùng mô phỏng lái xe, có người giám sát và dừng ngay khi người tham gia khó chịu hoặc quá buồn ngủ.
