# Kiến trúc Python – C++ của đồ án N16

## 1. Phân chia trách nhiệm

### Python: pipeline ngoại tuyến

- Tạo hoặc nhập dữ liệu EEG.
- Gán nhãn KSS và kiểm soát chất lượng.
- Trích đặc trưng hàng loạt.
- Huấn luyện logistic regression.
- Đánh giá Leave-One-Subject-Out.
- Xuất mô hình sang `output/model/n16_logistic_model.json`.
- Vẽ biểu đồ và chuẩn bị kết quả báo cáo.

### C++17: runtime thời gian thực

- Đọc Bluetooth COM trên Windows.
- Giải mã và kiểm tra checksum gói ThinkGear.
- Duy trì cửa sổ vòng 2.048 mẫu.
- Lọc 0,5–45 Hz bằng FFT.
- Tính công suất delta/theta/alpha/beta/gamma và các tỷ số.
- Phát hiện chớp mắt, độ rộng và chớp kéo dài.
- Đọc scaler, trọng số và ngưỡng từ JSON do Python xuất.
- Suy luận xác suất buồn ngủ mỗi giây.
- Ghi log và phát âm thanh sau hai cửa sổ liên tiếp vượt ngưỡng.

```text
DỮ LIỆU THÍ NGHIỆM
        │
        ▼
Python: chuẩn bị dữ liệu → huấn luyện → LOSO
        │
        ▼
              model.json
                  │
NeuroSky ─COM─► C++ runtime ─► DSP ─► inference ─► cảnh báo
```

## 2. Cấu trúc C++

```text
cpp/
├── CMakeLists.txt
├── include/n16/
│   ├── blink.hpp
│   ├── fft.hpp
│   ├── model.hpp
│   ├── runtime.hpp
│   ├── serial_port.hpp
│   ├── signal.hpp
│   ├── thinkgear.hpp
│   └── types.hpp
└── src/
    ├── blink.cpp
    ├── fft.cpp
    ├── main.cpp
    ├── model.cpp
    ├── runtime.cpp
    ├── serial_port.cpp
    ├── signal.cpp
    └── thinkgear.cpp
```

## 3. Build bằng MinGW-w64

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_cpp.ps1
```

Chương trình được tạo tại:

```text
cpp\build\n16_runtime.exe
```

Chạy demo bằng dữ liệu mô phỏng và mô hình Python:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_cpp_demo.ps1
```

## 4. Build bằng CMake

```powershell
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release --output-on-failure
```

Với Visual Studio, chương trình thường nằm tại `cpp/build/Release/n16_runtime.exe`.

## 5. Các chế độ chương trình

Kiểm tra FFT và parser:

```powershell
.\cpp\build\n16_runtime.exe self-test
```

Phát lại CSV:

```powershell
.\cpp\build\n16_runtime.exe replay `
  --input data\synthetic\demo_session.csv `
  --model output\model\n16_logistic_model.json `
  --log output\demo\cpp_replay_log.csv
```

Chạy trực tiếp với NeuroSky:

```powershell
.\cpp\build\n16_runtime.exe live `
  --port COM5 `
  --model output\model\n16_logistic_model.json `
  --duration-min 30
```

## 6. Định dạng chuyển giao mô hình

Python xuất JSON gồm:

- `feature_names`: thứ tự đặc trưng.
- `mean`: trung bình dùng chuẩn hóa.
- `scale`: độ lệch chuẩn dùng chuẩn hóa.
- `weights`: trọng số logistic regression.
- `bias`: hệ số chặn.
- `threshold`: ngưỡng quyết định.

C++ không huấn luyện lại. Nó chuẩn hóa đặc trưng bằng `mean`, `scale`, sau đó tính:

```text
z = Σ(((xᵢ - meanᵢ) / scaleᵢ) × weightᵢ) + bias
risk = 1 / (1 + exp(-z))
```

## 7. Trạng thái kiểm thử

Bản C++ đã được biên dịch bằng g++ và kiểm thử trên cùng phiên synthetic với Python. Hai runtime tạo 21 cửa sổ, cùng 10 cửa sổ cảnh báo và giá trị risk gần như trùng khớp. Đây chỉ là kiểm thử tương đương phần mềm, không phải kết quả thực nghiệm trên người.
