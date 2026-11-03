# N16 sử dụng đầy đủ ba nguồn GVHD

Cập nhật 28/10/2026. Theo yêu cầu của nhóm, cả ThinkGear, SEED-VIG và scikit-learn đều nằm trong phạm vi. **Đã có dữ liệu SEED-VIG thật trên máy**: kiểm tra 23 phiên và giải nén các nhánh 5Bands, EEG trán, EOG, nhãn vào `data/external/seed-vig`. Có dữ liệu chưa đồng nghĩa đã huấn luyện. Chưa có accuracy/F1; còn xác minh ID người và quy tắc nhãn. Xem [hồ sơ dữ liệu thật và cấu hình nháp](SEED_VIG_DU_LIEU_THAT_20261028.md). Các bước xin quyền bên dưới được giữ để tham khảo, không còn là mô tả thiếu file hiện tại.

## 1. Ba nguồn dùng vào ba việc

| Nguồn | Đã có trong mã | Bằng chứng thực nghiệm cần bổ sung |
| --- | --- | --- |
| [ThinkGear](https://developer.neurosky.com/docs/doku.php?id=thinkgear_communications_protocol) | Parser byte stream, checksum, signed raw và chất lượng; thu COM trực tiếp. Đường OpenViBE có bộ đọc/chuyển CSV. | Capture thiết bị thật, cấu hình/đơn vị, lỗi gói nếu thu trực tiếp, chất lượng và file raw. Khi thu qua OpenViBE, không nói parser riêng của N16 đã được dùng nếu thực tế chưa chạy nó. |
| [SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html) | Xem cấu trúc MAT, manifest, chuyển đặc trưng/nhãn PERCLOS, tab nhập riêng; đã kiểm tra MAT thật của 23 phiên. | Đã có file; còn hồ sơ mã người, xác nhận căn chỉnh, quy tắc ngưỡng và kết quả LOSO thật. |
| [scikit-learn](https://scikit-learn.org/stable/) | StandardScaler + Logistic Regression/SVM/Random Forest trong Pipeline; LeaveOneGroupOut theo người, chỉ số và lưu joblib. | Chạy trên SEED-VIG thật sau cấp quyền; chạy riêng trên NeuroSky khi có nhiều người và đủ lớp. Không lấy kiểm thử phần mềm làm kết quả nghiên cứu. |

Không cần triển khai mọi chức năng trong thư viện hoặc mọi lệnh firmware của tài liệu ThinkGear. Cần dùng đúng phần phục vụ bài toán và chỉ rõ bằng chứng. Nhánh cũ NumPy/JSON vẫn riêng; lựa chọn mô hình scikit-learn trên GUI không tự chuyển thành model runtime C++.

## 2. Việc làm ngay khi chưa có SEED-VIG

1. Mở [trang cấp quyền chính thức](https://bcmi.sjtu.edu.cn/home/seed/downloads.html).
2. Tải License Agreement, đọc và điền theo thông tin thật của người/đơn vị có quyền ký. Trao đổi GVHD khi cần xác nhận của đơn vị.
3. Gửi giấy phép qua [Apply](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/), chọn đúng SEED-VIG theo nội dung biểu mẫu hiện hành. Nhóm tự thực hiện bước ký/gửi; ứng dụng không tự đăng ký.
4. Trang chính thức cho biết sau xét duyệt sẽ gửi link tải và mật khẩu qua email; không giả định thời gian duyệt.
5. Trong lúc chờ, tiếp tục mốc thu S01/pilot và chuẩn bị metadata. Phần SEED-VIG trong báo cáo ghi đang xin quyền/chờ dữ liệu theo trạng thái thật.

Không lấy dữ liệu từ nơi không rõ quyền, không đưa mật khẩu hoặc file dữ liệu hạn chế lên Git công khai.

## 3. Khi đã nhận dữ liệu

1. Giữ cấu trúc thư mục gốc và metadata từ nhà cung cấp. Có thể đặt dưới `D:\dh\da1\data\external\seed-vig`. Thư mục external đã được bỏ qua bởi Git, nhưng vẫn kiểm tra trước khi đưa dữ liệu lên kho.
2. Nhấp đúp `Mo_UngDung_ML.cmd`, vào **tab 4. SEED-VIG**.
3. Bấm **Xem biến trong file MAT...**, chọn file đặc trưng và file nhãn để xem tên biến, kích thước và kiểu. Công cụ dùng scipy.io; MAT v7.3/HDF5 hoặc cấu trúc không hỗ trợ cần adapter khác sau khi kiểm tra file thật.
4. Bấm **Tạo manifest mẫu trống...**, lưu trong thư mục làm việc. Mở JSON bằng trình soạn thảo và thay mọi `REPLACE` bằng thông tin thật.
5. Điền một mục sessions cho mỗi cặp người/phiên. Không suy ra người từ tên file; dùng metadata. Cùng người qua nhiều phiên giữ cùng subject_id.

| Trường | Cách điền |
| --- | --- |
| `subject_id`, `session_id` | Mã người thật và phiên theo hồ sơ; không thay đổi để tạo thêm người |
| `feature_file`, `label_file` | Đường dẫn tuyệt đối hoặc tương đối từ thư mục JSON |
| `feature_key`, `label_key` | Tên biến đã xem và xác minh, không mặc định tên nhãn là perclos |
| `sample_axis` | Trục mẫu đếm từ 0; channel × sample × band dùng 1 nếu đúng layout thực |
| `alignment_verified` | Chỉ true sau khi kiểm tra hàng đặc trưng khớp nhãn/thời gian; cùng số hàng chưa đủ |
| `source_is_synthetic` | false cho dữ liệu thật; true cho fixture mô phỏng dùng kiểm thử. Không dùng fixture báo cáo kết quả thật |

Trang SEED-VIG mô tả đặc trưng EEG, EEG trán, EOG và nhãn PERCLOS liên tục 0–1. Đây không phải raw NeuroSky 512 Hz hoặc KSS. Bộ nhập hiện nhận một trường đặc trưng cùng layout cho các phiên trong mỗi manifest; không tự kết hợp EEG/EOG nhiều trường khác nhau. Phải kiểm tra cách làm trơn đặc trưng trước khi diễn giải kết quả thời gian thực.

## 4. Nhập và huấn luyện bằng GUI

1. Chọn JSON đã điền trong tab 4.
2. Ghi hai ngưỡng PERCLOS: tỉnh khi ≤ ngưỡng thấp, buồn ngủ khi ≥ ngưỡng cao. Cần `0 ≤ thấp < cao ≤ 1`; vùng giữa bị loại. **Không có ngưỡng mặc định**; chốt căn cứ/quy tắc trước khi xem điểm test, trao đổi GVHD để chọn bài toán phù hợp. Bộ nhập hiện hỗ trợ phân loại nhị phân, chưa hiện thực hồi quy PERCLOS.
3. Điền tên quy tắc và căn cứ; chỉ tích xác nhận quyền/căn chỉnh khi đã kiểm tra.
4. Bấm **Nhập MAT thành CSV đặc trưng...**, chọn nơi lưu. Mỗi lần tạo thư mục mới `seed_vig_<thời gian>_<mã>`.
5. Giữ `features.csv` và `seed_import.json`. JSON chứa ngưỡng, snapshot manifest, số hàng/người, đường dẫn và SHA-256 của file nguồn. Snapshot dùng để kiểm tra nguồn, không phải manifest đã tự sửa đường dẫn để chạy lại ở thư mục mới.
6. Sau nhập, tab 3 thay danh sách đầu vào bằng CSV SEED-VIG vừa nhập và chọn nguồn seed-vig; ứng dụng chưa tự huấn luyện. Nếu cần giữ lựa chọn cũ, ghi lại trước.
7. Vào tab 3, giữ các đặc trưng SEED-VIG hoặc bấm **Đặc trưng SEED-VIG**; chọn mô hình. Không dùng nút Chỉ phổ của N16 để suy ra nhóm feature SEED-VIG.
8. Khi đủ nhiều người và hai lớp trong các tập học, chạy **Huấn luyện + LOSO...**. Giữ metrics.json, oof_predictions.csv, pipeline.joblib và inputs.json. Fold thiếu lớp có thể bị bỏ qua; báo đúng số fold/người hợp lệ.

SEED-VIG và NeuroSky chạy **hai kết quả riêng**. Không trộn PERCLOS/KSS thành nhãn chung, không huấn luyện trên đặc trưng nhiều kênh rồi gắn thẳng vào NeuroSky đơn kênh. Joblib ngoại tuyến không tự đáp ứng tiêu chí cảnh báo live hoặc xuất JSON cho C++.

## 5. Cách ghi vào báo cáo M2

- **ThinkGear:** mô tả đúng đường thu đã chạy, giao thức và schema; parser riêng có kiểm thử nhưng chỉ nói đã thử trên phần cứng khi có capture/log tương ứng.
- **SEED-VIG chưa nhận:** ghi đang xin quyền/chờ dữ liệu; đã chuẩn bị bộ nhập và quy trình đánh giá, kiểm thử bằng fixture. Không ghi đã huấn luyện trên SEED-VIG thật.
- **SEED-VIG đã nhận và chạy:** ghi nguồn/phiên bản, người/phiên, trường đặc trưng, ngưỡng, cách split và số liệu từng người có nguồn.
- **scikit-learn:** trình bày Pipeline/scaler trong mỗi fold và thuật toán thực dùng. Chỉ một người tự thu thì chưa có LOSO NeuroSky; kết quả SEED-VIG không thay phần đánh giá thiết bị tự thu.

Hoàn tất yêu cầu ba nguồn cần đủ cả mã, dữ liệu được phép dùng và bằng chứng chạy. Việc được cấp dữ liệu là phụ thuộc còn mở; kế hoạch M2 phải báo đúng trạng thái, không đánh dấu xong SEED-VIG chỉ vì có tab nhập.
