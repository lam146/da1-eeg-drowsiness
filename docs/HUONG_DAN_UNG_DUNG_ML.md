# Hướng dẫn ứng dụng nạp CSV vào ML — N16

Ứng dụng chạy trên máy Windows, xử lý file tại máy. Không cần phần mềm mô phỏng lái xe để sử dụng. Giao diện này phục vụ chuẩn bị dữ liệu và huấn luyện ngoại tuyến.

## 1. Mở ứng dụng

1. Mở File Explorer, vào `D:\dh\da1`.
2. Nhấp đúp `Mo_UngDung_ML.cmd`.
3. Cửa sổ **N16 | Nạp CSV và học máy** xuất hiện với bốn tab; tab 4 dành riêng cho SEED-VIG.
4. Giữ cửa sổ dòng lệnh đang mở cho đến khi dùng xong ứng dụng.

Nếu mở bằng PowerShell tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe -m n16_eeg.ml_gui
```

Máy hiện tại đã có môi trường cần thiết. Khi chuyển dự án sang máy khác có Python và Tkinter, tạo môi trường và cài gói một lần:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml-extra]"
```

## 2. Xem file vừa đo

1. Chọn tab **1. Nạp và xem CSV**.
2. Bấm **Chọn file CSV...** rồi chọn file đo.
3. Đọc thông tin số dòng, cột, ô thiếu và thời lượng.
4. Xem bảng và đồ thị raw ở phía dưới.

Đọc được CSV phân tách bằng dấu phẩy, chấm phẩy hoặc tab. Hai định dạng raw hỗ trợ:

| Loại file | Cột cần có |
| --- | --- |
| OpenViBE 512 Hz | `Time:512Hz`, `Electrode` |
| Raw N16 | `raw_eeg`, `signal_time_ms` hoặc `timestamp_ms` |

Thời gian OpenViBE tính bằng giây, thời gian raw N16 tính bằng mili giây. Dữ liệu cần liên tục ở 512 Hz; ứng dụng báo lỗi nếu có khoảng đứt hoặc giá trị raw không hợp lệ. Giữ nguyên file gốc để đối chiếu.

**Với `raw\test1.csv`:** bạn có thể nạp để xem ngay. Đây là lần thử chớp mắt theo lệnh; không gán cả file là tỉnh hay buồn ngủ dựa vào động tác chớp mắt. Chưa đủ nhãn thì dừng ở bước xem dữ liệu.

## 3. Chuẩn bị nhãn cho một phiên đo thực tế

Chỉ làm bước này sau khi có nhật ký KSS và quy tắc gán nhãn đã thống nhất với GVHD. Ứng dụng không tự suy ra KSS từ EEG.

1. Sang tab **2. Tín hiệu → đặc trưng**.
2. Nhập mã người, ví dụ `S01`. Cùng một người luôn dùng cùng mã ở tất cả ngày đo.
3. Nhập mã phiên, ví dụ `S01_20261024_P01`. Mỗi lần đo dùng mã phiên riêng.
4. Nhập tên/phiên bản quy tắc nhãn đã thống nhất để lưu cùng kết quả.
5. Chọn đơn vị raw đã xác nhận từ cấu hình thu: `count` hoặc `uV`. Không đoán đơn vị từ biên độ đồ thị.
6. Bấm **Tạo mẫu nhãn trống...**, đặt tên như `S01_20261024_P01_nhan.csv`. Mẫu chỉ có tên cột; bạn cần điền dữ liệu thật.
7. Mở mẫu bằng Excel hoặc trình soạn thảo văn bản, điền rồi lưu CSV và đóng file trước khi nhập lại.

File nhãn cần bốn cột:

| Cột | Cách điền |
| --- | --- |
| `start_s` | Giây bắt đầu, tính từ mẫu đầu tiên trong CSV raw = 0 giây |
| `end_s` | Giây kết thúc, lớn hơn bắt đầu và không vượt thời lượng file |
| `kss` | Số nguyên 1–9 theo KSS thực tế và quy tắc căn thời gian đã thống nhất |
| `quality_status` | `usable`: đã xem, dùng được; `exclude`: loại; `unknown`: chưa rõ |

Ví dụ **chỉ minh họa cú pháp**, không phải nhãn cho file của bạn:

```csv
start_s,end_s,kss,quality_status
0,60,3,usable
60,65,,exclude
65,120,,unknown
```

Các khoảng không được chồng nhau. Có thể để trống KSS cho vùng loại/chưa rõ. Số thập phân dùng dấu chấm, ví dụ `12.5`. Vùng `usable` phải có KSS hợp lệ.

Nếu quy tắc đã thống nhất là dùng **60 giây trước thời điểm trả lời KSS**, chỉ điền khoảng đó sau khi đã căn đồng hồ nhật ký và CSV. Ví dụ mốc KSS ở giây 300 tương ứng khoảng 240–300 giây. Không tự áp dụng một điểm KSS cho toàn bộ 5 phút; không mặc định chọn quy tắc này khi chưa được thống nhất.

Xem nhật ký sự kiện và tín hiệu để loại vùng đeo/chỉnh điện cực, mất tiếp xúc và thao tác chủ động. Tách chúng thành khoảng `exclude`; vùng chưa đủ cơ sở dùng `unknown`. Đồ thị chỉ hỗ trợ xem, không tự xác nhận chất lượng. Nếu file không có `poor_signal`, đánh giá bằng tay được ghi rõ là thủ công.

## 4. Tạo bảng đặc trưng

1. Bấm **Chọn file nhãn...**, chọn CSV vừa điền.
2. Kiểm tra lại người, phiên, đơn vị, quy tắc nhãn và các khoảng thời gian.
3. Tích ô xác nhận sau khi đã kiểm tra đúng nội dung.
4. Bấm **Tạo đặc trưng và lưu thư mục mới...**, chọn nơi lưu, ví dụ `D:\dh\da1\output`.
5. Chờ thông báo hoàn tất và ghi lại đường dẫn kết quả.

Ứng dụng dùng cửa sổ **4 giây**, bước **1 giây**. Chỉ giữ cửa sổ nằm hoàn toàn trong một khoảng `usable`: KSS 1–5 thành nhãn 0, KSS 7–9 thành nhãn 1, KSS 6 bỏ khỏi baseline. Cửa sổ qua ranh giới nhãn hoặc vùng loại bị bỏ. Đây là cấu hình baseline của N16, cần được thống nhất khi làm báo cáo.

Nếu có `poor_signal`, ứng dụng kiểm tra thêm: trong cửa sổ có dữ liệu chất lượng thì phải đầy đủ, trong phạm vi 0–255; tỷ lệ mẫu >50 không vượt 20%. Ngưỡng này là cấu hình lọc dữ liệu của phần mềm, không thay thế kiểm tra thực nghiệm. Nếu thiếu toàn bộ `poor_signal`, ứng dụng dựa vào xác nhận thủ công, không tạo giá trị chất lượng giả.

Mỗi lần lưu tạo một thư mục `features_<thời gian>_<mã>` mới:

- `features.csv`: đặc trưng phổ và chớp mắt theo cửa sổ, có mã người/phiên và nhãn.
- `preparation.json`: quy tắc, đơn vị, số cửa sổ giữ/loại, đường dẫn và mã kiểm tra SHA-256 của file nguồn.

Lặp bước 2–4 cho từng phiên của từng người. Các file ghép huấn luyện phải cùng bộ cột, cùng nguồn và đơn vị. Cùng một người không được đổi mã để tạo thêm người cho LOSO.

## 5. Nạp bảng đặc trưng và huấn luyện

1. Sang tab **3. Huấn luyện và kết quả**.
2. Bấm **Chọn các CSV đặc trưng...**. Giữ Ctrl để chọn nhiều `features.csv` từ các lần chuẩn bị; lần chọn mới thay danh sách cũ.
3. Chọn nguồn đúng: `openvibe`, `neurosky` hoặc `seed-vig`. SEED-VIG nhập ở tab 4 hoặc công cụ manifest riêng, không ghép chung với dữ liệu thiết bị của bạn. Xem [hướng dẫn ba nguồn GVHD và SEED-VIG](SEED_VIG_VA_BA_NGUON_GVHD.md).
4. Chọn mô hình: `logistic`, `svm` hoặc `random-forest`.
5. Chọn đặc trưng. Lần đầu có thể dùng **Chỉ phổ** để chạy baseline; sau đó chạy thử nhóm có chớp mắt theo kế hoạch đánh giá đã định trước.
6. Bấm **Huấn luyện + LOSO...**, chọn nơi lưu và chờ hoàn tất.
7. Đọc kết quả từng người rồi bấm **Mở thư mục kết quả**.

Cần ít nhất hai người thật với mã khác nhau và đủ hai lớp trong dữ liệu huấn luyện. LOSO lần lượt giữ một người để kiểm tra, học từ những người còn lại. Fold thiếu lớp để học được ghi là bỏ qua; nếu tất cả fold không đánh giá được, chương trình báo lỗi. Đây là điều kiện phần mềm, không có nghĩa hai người đã đủ cho một nghiên cứu tốt.

Giao diện chỉ đề xuất các đặc trưng N16/SEED-VIG hỗ trợ. Không đưa ID, KSS, nhãn hay cột thời gian vào đặc trưng. File feature có số thiếu/NaN/Inf, trùng cửa sổ, lẫn nguồn hoặc đơn vị sẽ bị chặn. CSV tùy ý có cấu trúc khác cần chuyển đúng schema trước.

Mỗi lần chạy tạo thư mục `training_<thời gian>_<mã>` mới:

| File | Dùng để làm gì |
| --- | --- |
| `pipeline.joblib` | Mô hình ngoại tuyến đã học trên toàn bộ đầu vào sau đánh giá |
| `metrics.json` | Accuracy, macro-F1, balanced accuracy, ROC-AUC và kết quả từng người |
| `oof_predictions.csv` | Nhãn thật và dự đoán cho các mẫu kiểm tra ngoài người học |
| `inputs.json` | Danh sách và SHA-256 của CSV đã dùng |

Kết quả trên giao diện là đánh giá LOSO; mô hình cuối học toàn bộ dữ liệu để lưu. AUC/balanced accuracy có thể không tính được khi người kiểm tra chỉ có một lớp; xem số người có chỉ số hợp lệ. Điểm SVM là `decision_function`, không phải xác suất phần trăm. Dữ liệu mô phỏng được gắn cảnh báo và chỉ dùng kiểm thử phần mềm.

Không chọn mô hình/ngưỡng bằng cách thử nhiều lần trên cùng kết quả kiểm tra rồi coi đó là kết quả độc lập. `pipeline.joblib` chưa phải JSON chạy C++ thời gian thực; GUI này chưa xuất model sang runtime C++. Không nạp joblib từ nguồn không tin cậy.

## 6. Khi gặp lỗi

| Thông báo/tình huống | Cách xử lý |
| --- | --- |
| Không có cửa sổ đủ nhãn/chất lượng | Kiểm tra căn thời gian; cần ít nhất một khoảng dùng được dài 4 giây, KSS khác 6 |
| `LOSO requires at least two identified people` | Bổ sung dữ liệu người khác; một người đo nhiều phiên vẫn chỉ là một người |
| `Final training requires both classes` | Chưa có đủ hai lớp; tiếp tục thu theo kế hoạch, không sửa nhãn để chạy được |
| `No evaluable LOSO fold` | Người còn lại không đủ hai lớp để học; bổ sung dữ liệu phù hợp |
| `Dataset provenance mismatch` | Chọn đúng nguồn, tách các bộ dữ liệu khác nguồn |
| `Require one verified signal unit` | Xác minh đơn vị; không trộn count với uV |
| Mẫu nhãn đã tồn tại | Dùng tên mới; nút tạo mẫu không ghi đè file đang có |
| Chỉ có raw và chưa có KSS | Xem tín hiệu ở tab 1; chuẩn bị nhật ký/nhãn thực tế trước khi huấn luyện |

Với lần đo thử chớp mắt hiện tại, bước phù hợp là mở tab 1 để xem tín hiệu. Các lần đo phục vụ phân loại cần bổ sung KSS và nhật ký sự kiện ngay khi đo để có thể xử lý tiếp ở tab 2 và 3.
