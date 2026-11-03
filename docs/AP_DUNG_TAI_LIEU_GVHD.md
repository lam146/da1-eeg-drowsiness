# Áp dụng tài liệu GVHD cho N16

Cập nhật bổ sung 28/10/2026: đã có và kiểm tra dữ liệu SEED-VIG thật của 23 phiên tại `data/external/seed-vig`; chưa huấn luyện. Xem [hồ sơ dữ liệu hiện tại](SEED_VIG_DU_LIEU_THAT_20261028.md). Nội dung ngày 23/10 bên dưới là lịch sử triển khai; các câu “chưa có dữ liệu” chỉ phản ánh thời điểm đó. Với đường OpenViBE NeuroSky 3.6.0 đã kiểm tra, Electrode được giữ ở raw count; không mặc định µV theo tài liệu Word.

Cập nhật 23/10/2026. Giữ kiến trúc Python huấn luyện và C++17 runtime, bổ sung nhánh scikit-learn ngoại tuyến và nhập dữ liệu ngoài. Không tự thay báo cáo M1 đã nộp. File Word gốc giữ nguyên; không cài hay chạy các bộ cài được dẫn trong file.

## 1. Nguồn mới làm thay đổi gì

| Nguồn | Áp dụng vào dự án | Phạm vi kiểm chứng |
|---|---|---|
| HDSD_NeuroSky_MindWave.docx của GVHD | Bổ sung đường USB Adapter / ThinkGear Connector / OpenViBE; chọn Electrode, kiểm tra cấu hình 512 Hz | Đã đọc văn bản và các ảnh giao diện/CSV liên quan; cần CSV và thiết bị thực để xác nhận |
| [ThinkGear Serial Stream Guide](https://developer.neurosky.com/docs/doku.php?id=thinkgear_communications_protocol) | Giải mã signed big-endian, checksum, kiểm tra payload, log lỗi gói và raw count | Test gói phân mảnh, sai checksum, payload thiếu, biên signed và EXCODE |
| [SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html) | Nhập MAT đặc trưng, giữ person/session rõ ràng, đánh giá riêng dữ liệu lái mô phỏng | Hiện kiểm tra bằng MAT giả lập; chưa có bản dữ liệu được cấp quyền |
| [scikit-learn](https://scikit-learn.org/stable/) | Pipeline StandardScaler và classifier; LeaveOneGroupOut theo subject_id | Logistic Regression, SVM và Random Forest; không chọn mô hình theo test vòng ngoài |

SEED-VIG công bố đặc trưng EEG, EEG trán, EOG và nhãn PERCLOS liên tục trong [0,1]. Trang mô tả có tensor EEG 17×885×5 hoặc ×25; EOG 885×36. Đây không phải raw NeuroSky 512 Hz hay nhãn KSS. Dùng làm benchmark ngoại tuyến, không biến các tensor thành luồng raw.

## 2. Hai đường thu không chạy tranh cổng

**Đường OpenViBE theo hướng dẫn GVHD:** cấu hình USB/driver đúng phiên bản thiết bị, MindWave Manager, ThinkGear Connector, Acquisition Server với NeuroSky Mindset và cấu hình 512 Hz, rồi Designer ghi CSV. Các bước là nội dung hướng dẫn được cung cấp, không phải xác nhận đã chạy thành công trên máy hiện tại. COM4 trong ảnh chỉ là ví dụ, phải chọn COM thực tế.

Ảnh Driver Properties có `Power (8 channels)`: đó là các giá trị công suất do thiết bị/phần mềm cung cấp, không biến tai nghe thành EEG 8 điện cực. CSV có `Electrode`, `Delta`, `Theta` và các cột công suất khác; nhánh DSP của N16 lấy **Electrode**, không dùng Theta làm raw. File hướng dẫn ghi Electrode mặc định là microvolt; cần giữ đơn vị này và kiểm tra đúng cấu hình xuất. Không đổi nhãn µV thành count.

**Đường COM trực tiếp:** đóng ứng dụng đang chiếm COM trước khi dùng chương trình N16. Khi chạy tuyến OpenViBE/Connector, không đồng thời mở cùng COM bằng chương trình thu trực tiếp. Nếu adapter không cung cấp ThinkGear serial tương thích thì dùng đường OpenViBE, không cố áp đặt Bluetooth SPP của phiên bản khác.

Phần mềm chỉ đọc cổng, không tự gửi lệnh thay baud hoặc firmware. Không cần chạy toàn bộ ứng dụng bằng quyền quản trị nếu có thể ghi file trong thư mục dự án; chỉ cấp quyền khi có lý do rõ ràng cho thao tác cụ thể.

## 3. Chuẩn bị môi trường

Trong PowerShell tại `D:\dh\da1`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[hardware,ml-extra,dev]"
.\.venv\Scripts\python.exe -m n16_eeg.acquire --list-ports
```

Các lệnh sau dùng `python -m ...` với interpreter `.venv`, không phụ thuộc việc đã cài entry point hay chưa. Không sửa code C++ trong đợt cập nhật này.

## 4. Thu COM trực tiếp và kiểm tra raw

Thay COM5 bằng cổng thực. Đặt tên mới cho mỗi phiên; chương trình từ chối ghi đè.

```powershell
.\.venv\Scripts\python.exe -m n16_eeg.acquire --port COM5 --subject S01 --session pilot_01 --duration-min 2 --output data/raw/S01_pilot_01.csv
```

CSV giữ các trường cũ và bổ sung `sample_index`, `signal_time_ms`, `signal_unit=count`, `dataset_id=neurosky`, `fs_hz=512`. `timestamp_ms` là thời gian PC nhận mẫu; `signal_time_ms=index/512` chỉ là trục thời gian danh định, không chứng minh không mất mẫu. Sau kết thúc bình thường có `.capture.json`: số mẫu, lỗi checksum/length/payload và ước lượng tốc độ mẫu đến. File chẩn đoán này chưa được bảo đảm xuất khi ngắt đột ngột/Ctrl+C.

Kiểm tra số mẫu, chất lượng tiếp xúc, khoảng gián đoạn và tốc độ đến trong phiên đủ dài. Các mẫu Bluetooth có thể đến theo cụm. Không dùng duy nhất sample_index hoặc tốc độ đến để khẳng định đạt đủ 512 Hz. Parser đã chặt hơn nhưng chưa thử trực tiếp adapter của nhóm.

## 5. Nhập CSV OpenViBE

Mở CSV bằng trình soạn thảo để xem dấu phân cách và tên cột. Theo ảnh GVHD, time thường tên `Time:512Hz`, tín hiệu là `Electrode`; phải dùng tên đúng trong file thực.

```powershell
.\.venv\Scripts\python.exe -m n16_eeg.openvibe_import data/external/openvibe/S01.csv data/raw/S01_openvibe.csv --separator ";" --time-column "Time:512Hz" --electrode-column "Electrode" --time-unit seconds --signal-unit uV --subject S01 --session openvibe_01
```

Bộ nhập chỉ nhận signal CSV một hàng/mẫu, thời gian tăng liên tục gần 1/512 s. Nếu CSV khác cấu trúc, có gián đoạn hoặc header không khớp, chương trình dừng để kiểm tra chứ không tự đoán. Không chuyển đơn vị tín hiệu. Có `.source.json` lưu nguồn và cách ánh xạ.

CSV ví dụ trong hướng dẫn không có cột poorSignal. Vì vậy bộ nhập để `poor_signal` và `kss` trống, **không tự coi tín hiệu sạch hoặc tự gán nhãn tỉnh**. Nếu bản xuất thực có poorSignal đúng thang ThinkGear 0..255 thì truyền `--quality-column "tên cột"`. Không dùng phần trăm noise của giao diện như poorSignal. Nếu không có chất lượng, cần bổ sung ghi/log chất lượng hoặc chốt quy trình gán cờ chất lượng riêng trước bước tạo feature; chưa có đường tự động bỏ qua kiểm tra chất lượng.

Gán nhãn KSS đúng protocol đã được thống nhất. Lưu ý `featurize.py` hiện còn forward-fill KSS tối đa 5 phút; đó **không phải** quy tắc “60 giây trước trả lời” trong đề xuất M1. Chưa tự đổi cách gán nhãn trong đợt này. Trước thực nghiệm phải thống nhất một quy tắc, khóa và thực hiện nhất quán; không diễn giải các nhãn forward-fill như mốc khởi phát tới giây.

```powershell
.\.venv\Scripts\python.exe -m n16_eeg.featurize data/raw/S01_openvibe.csv data/processed/S01_openvibe_features.csv
```

Chỉ chạy bước này sau khi có KSS và chất lượng hợp lệ. Feature giữ đơn vị/nguồn/fs, từ chối nhiều person/session trong một file và sample_index đứt. Bộ kiểm tra này chưa tự nhận mọi khoảng mất dữ liệu Bluetooth vì sample_index do PC gán.

## 6. SEED-VIG: lấy dữ liệu và chuẩn bị manifest

[Trang cấp quyền](https://bcmi.sjtu.edu.cn/home/seed/downloads.html) yêu cầu hoàn tất giấy phép và gửi qua Apply, sau xét duyệt sẽ nhận liên kết/mật khẩu qua email. Người dùng/GVHD thực hiện bước cấp quyền. Code không tự đăng ký, ký giấy phép, tải hoặc công khai dữ liệu. Thư mục `data/external/` đã được bỏ qua bởi Git.

Đặt bản được cấp quyền vào `data/external/seed-vig/`. Xem các biến trước:

```powershell
.\.venv\Scripts\python.exe -m n16_eeg.seed_vig inspect data/external/seed-vig/ten_file.mat
```

`inspect` dùng scipy.io và trả tên biến, kích thước, kiểu. MAT v7.3/HDF5 hoặc struct phức tạp chưa hỗ trợ; cần xem cấu trúc thực rồi bổ sung adapter, không đổi đuôi file.

Sao chép `configs/seed_vig_manifest.example.json` thành manifest làm việc. Ví dụ cố tình chưa thể chạy: phải thay tên file, tên biến, mã người/phiên và xác nhận căn chỉnh. Các đường dẫn tương đối tính từ thư mục manifest, không phải cwd.

- `subject_id`: mã người thật từ hồ sơ dataset; mọi phiên cùng người phải cùng ID. Không mặc định một file là một người.
- `session_id`: mã phiên, cặp người/phiên không trùng.
- `feature_key`: một biến đã kiểm tra, thí dụ `psd_movingAve`; các phiên phải cùng trường/layout.
- `sample_axis`: zero-based; tensor channel×sample×band dùng 1, EOG sample×feature dùng 0.
- `label_key`: xem từ file nhãn thực; không đoán là `perclos` khi chưa inspect.
- `alignment_verified`: chỉ đặt true sau kiểm tra feature và nhãn đúng thứ tự/thời gian. Bằng số dòng chưa đủ chứng minh đồng bộ.

Bộ nhập dời sample axis lên đầu rồi flatten các chiều còn lại thành `seed_<key>_0000...`. Giữ person, session, feature_row và perclos để truy xuất. Mô hình dùng feature_names trong sidecar, không dùng perclos như đầu vào.

## 7. Nhị phân hóa PERCLOS và phân loại

PERCLOS gốc là nhãn liên tục. Nếu chọn phân loại, phải chốt hai ngưỡng **trước** nhìn test. Giá trị ở giữa hai ngưỡng bị loại; không có ngưỡng chuẩn N16 được giả định trong code. Hồi quy trực tiếp PERCLOS là hướng có thể bổ sung, chưa được hiện thực ở module này.

Lệnh minh họa dưới dùng 0,2 và 0,8 **chỉ để mô tả cú pháp**, không phải ngưỡng GVHD hoặc nghiên cứu gốc đã chốt:

```powershell
.\.venv\Scripts\python.exe -m n16_eeg.seed_vig convert configs/seed_vig_manifest.json data/processed/seed_vig_binary.csv --alert-max 0.2 --drowsy-min 0.8
$seedMeta = Get-Content data/processed/seed_vig_binary.manifest.json -Raw | ConvertFrom-Json
$seedFeatures = $seedMeta.feature_names
.\.venv\Scripts\python.exe -m n16_eeg.sklearn_train data/processed/seed_vig_binary.csv --features $seedFeatures --dataset seed-vig --classifier logistic --output-dir output/seed_vig/logistic_run01
```

Đầu ra gồm `pipeline.joblib`, `metrics.json`, `oof_predictions.csv`; sidecar của bước nhập giữ ngưỡng/phiên/nguồn và số dòng loại. Sao lưu sidecar cùng kết quả, không chỉ lấy mỗi metrics.json. Không load joblib từ nguồn không tin cậy vì có thể thực thi mã khi nạp.

Các feature đã làm mượt (moving average/LDS) cần kiểm tra tính nhân quả trước khi nói chạy online. Đợt này chỉ đánh giá ngoại tuyến trên feature được cung cấp. Một pipeline trên chúng không phải bằng chứng runtime C++ NeuroSky có cùng kết quả.

## 8. Scikit-learn trên dữ liệu tự thu

Ghép feature nhiều người có cùng cấu hình, đơn vị và quy tắc nhãn; không trộn SEED-VIG/OpenViBE/count. Chọn cột rõ ràng, ví dụ:

```powershell
.\.venv\Scripts\python.exe -m n16_eeg.sklearn_train data/processed/all_subjects_features.csv --features theta_alpha rel_theta rel_alpha blink_rate_per_min blink_width_mean --dataset neurosky --classifier logistic --output-dir output/sklearn/logistic_run01
```

Với nguồn OpenViBE dùng `--dataset openvibe`. Đối chiếu SVM và Random Forest bằng `--classifier svm` / `--classifier random-forest` và thư mục run mới. Giữ cùng feature, dữ liệu, nhãn và người test. Các cấu hình mặc định cố định; chưa có nested tuning. Không chọn “mô hình thắng” từ LOSO ngoài rồi gọi đó là test độc lập cuối cùng.

Thực hiện [Pipeline để tránh rò rỉ chuẩn hóa](https://scikit-learn.org/stable/common_pitfalls.html) và [LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html) với nhóm là người. Test một lớp vẫn được lưu: confusion matrix, accuracy, macro-F1 với labels=[0,1], zero_division=0; AUC và balanced accuracy đặt null theo quy ước báo cáo. Train một lớp không huấn luyện được thì lưu fold skipped kèm lý do; không âm thầm bỏ khỏi hồ sơ.

Mỗi fold lưu train_subjects và mean/scale để audit. OOF CSV chỉ chứa dự đoán ngoài fold, không dùng prediction của model cuối fit toàn dữ liệu. Model cuối chỉ phục vụ nghiên cứu ngoại tuyến sau đánh giá. Dữ liệu synthetic có cảnh báo; không dùng chỉ số test fixture như kết quả người thật.

## 9. C++ và xuất mô hình

Runtime C++ hiện đọc JSON logistic cũ, không đọc joblib/SVM/Random Forest. Đợt này **không tự xuất pipeline scikit-learn sang C++**. Nhánh NumPy cũ vẫn dùng cho workflow hiện có, nhưng chặn export legacy nếu nguồn đánh dấu OpenViBE/SEED-VIG hoặc đơn vị không phải count.

Trước khi triển khai model sklearn trên thiết bị, cần export scaler/weights/bias/feature order, manifest DSP và kiểm tra Python-C++ trên cùng raw. Đặc trưng SEED-VIG không cùng schema với runtime N16; không chép joblib hoặc trọng số vào JSON để chạy live. Nhánh thu OpenViBE µV và raw count phải được hiệu chuẩn/thống nhất đơn vị hoặc huấn luyện riêng.

## 10. Việc cần làm tiếp

1. Thu pilot theo tuyến thiết bị thực, gửi vài dòng đầu CSV và cấu hình xuất để kiểm tra adapter.
2. Chốt chất lượng và quy tắc gán KSS trước tạo feature thực nghiệm.
3. Xin/nhận bản SEED-VIG được cấp quyền, kiểm tra ID người và biến MAT.
4. Chốt ngưỡng PERCLOS nếu làm phân loại; giữ benchmark riêng với KSS.
5. Chạy baseline LOSO cùng cấu hình và xem từng người, không chỉ mean.
6. Sau khi khóa pipeline mới triển khai/export sang C++ và đo độ trễ âm thanh thật.

Các file báo cáo M1 và giáo trình PDF trước đó không được sửa trong đợt áp dụng này. Tài liệu này là bản cập nhật triển khai; cần bổ sung vào M2 khi có kết quả thật. Không có tuyên bố đã đạt blink >=90% hoặc trễ <3 s từ test phần mềm.

## 11. Kiểm thử ngày 23/10/2026

Đã chạy 38 test và `pip check` thành công trong `.venv`, với scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0. Test dùng gói byte, signal CSV và MAT giả lập; đã thử CLI huấn luyện, lưu/đọc joblib và chặn ghi đè. Chưa chạy adapter vật lý, OpenViBE live hoặc bản SEED-VIG được cấp quyền. Trình render Word không có LibreOffice trong runtime, nên việc đọc tài liệu GVHD dùng văn bản OOXML và các ảnh nhúng liên quan; không thay đổi file Word gốc.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Giữ nguyên các thay đổi có sẵn trong worktree; không tạo commit/push trong đợt này.
