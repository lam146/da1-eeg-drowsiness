# N16 – Phát hiện buồn ngủ khi lái xe từ EEG NeuroSky

Mã nguồn MVP cho Đồ án 1 nhóm N16, HK1 2026–2027. Phần mềm có thể chạy hoàn chỉnh khi chưa có thiết bị bằng dữ liệu mô phỏng; khi có NeuroSky MindWave Mobile 2 chỉ cần chuyển sang bước thu dữ liệu thật.

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

Không thử nghiệm khi đang lái xe thật. Chỉ dùng mô phỏng lái xe, có người giám sát và dừng ngay khi người tham gia khó chịu hoặc quá buồn ngủ.
