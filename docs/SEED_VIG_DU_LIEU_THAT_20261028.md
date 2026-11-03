# Dữ liệu SEED VIG đã kiểm tra ngày 28 tháng 10 năm 2026

Đã tìm thấy bộ dữ liệu trong `D:\dh\SEED-VIG (2).zip`, không phải thư mục con sẵn có trong `da1`. Đã đọc README và giải nén chọn lọc 95 file, 76.045.788 byte, vào `D:\dh\da1\data\external\seed-vig`. Không cần giải nén Raw_Data hoặc EEG_Feature_2Hz để dùng các nhánh đặc trưng hiện có. Hai ZIP tên `SEED-VIG.zip` và `SEED-VIG (1).zip` ở thư mục cha chỉ có 38 byte và không đọc được như ZIP; không dùng chúng.

## Kết quả kiểm tra dữ liệu thật

| Nhánh | Phiên | Kích thước một file | Biến đầu vào đã kiểm tra | Trục hàng | Số đặc trưng khi trải phẳng |
|---|---:|---|---|---:|---:|
| EEG_Feature_5Bands | 23 | 17 × 885 × 5 | psd_movingAve | 1 | 85 |
| Forehead_EEG/EEG_Feature_5Bands | 23 | 4 × 885 × 5 | psd_movingAve | 1 | 20 |
| EOG_Feature | 23 | 885 × 36 | features_table_ica | 0 | 36 |
| perclos_labels | 23 | 885 × 1 | perclos | Không áp dụng | Nhãn liên tục |

Mỗi nhánh đặc trưng có 20.355 hàng trước nhị phân hóa. Ba nhánh mô tả cùng 23 phiên, không cộng thành 61.065 quan sát độc lập. Đã kiểm tra mọi biến dữ liệu trong các file đã chọn: đúng kích thước và không NaN/Inf. Mọi vector perclos có 885 số hữu hạn trong [0,1]. Ghép theo cùng tên file gốc, không ghép theo thứ tự liệt kê thư mục.

Hai biến psd_LDS và de_LDS cùng các biến movingAve cũng có trong file EEG. Không khẳng định các đặc trưng đã làm trơn là nhân quả hoặc tương đương luồng thời gian thực. Bộ nhập hiện hỗ trợ chọn một trường đặc trưng mỗi cấu hình; chưa có cấu hình ghép EEG+EOG tự động.

## Các điều đã biết và còn cần xác minh

- Đã có dữ liệu MAT thật trên máy; trạng thái cũ “chỉ có fixture/chưa có dữ liệu” không còn đúng với lần kiểm tra này.
- Có 23 phiên nhưng chỉ 21 tiền tố số khác nhau trong tên file; tiền tố 4 và 5 mỗi tiền tố có hai phiên. Không coi 23 file là 23 người. README không cung cấp bảng ánh xạ người rõ ràng; bài gốc cũng không đủ để xác nhận ánh xạ cụ thể tên file của bản phát hành này. Các nhóm tiền tố chỉ là ứng viên, chưa phải ID người đã kiểm chứng.
- Đã kiểm tra cùng tên phiên, số hàng và mô tả thứ tự trong README. Chưa có timestamp độc lập để kiểm chứng đồng bộ tới giây; cấu hình nháp giữ alignment_verified=false để người thực hiện rà soát trước khi nhập.
- Chưa chọn hai ngưỡng PERCLOS. Không lấy KSS 1–5/7–9 áp lên PERCLOS. Nếu muốn đối chiếu nghiên cứu gốc, README mô tả hồi quy với COR và RMSE và chia theo thời gian trong mỗi thí nghiệm; phân loại LOSO của dự án là một thiết kế đánh giá khác.
- Chưa chạy huấn luyện hoặc tạo accuracy/F1 trên bộ dữ liệu này. Không báo cáo việc kiểm tra MAT như kết quả ML.

## Các cấu hình đã tạo

Trong `D:\dh\da1\output\seed_vig_local_20261028`:

- `manifest_eeg_5bands.draft.json`: nhánh EEG 85 đặc trưng.
- `manifest_forehead_5bands.draft.json`: nhánh EEG trán 20 đặc trưng.
- `manifest_eog_ica.draft.json`: nhánh mắt 36 đặc trưng.
- `inventory.json`: kích thước, biến, min/max nhãn và SHA-256 từng cặp file.

Đường dẫn MAT và tên biến/trục mẫu đã điền từ kiểm tra thật. Các trường subject_id vẫn là REPLACE theo nhóm ứng viên và alignment_verified=false. Đây là cấu hình nháp cố ý chưa chạy huấn luyện; không thay REPLACE bằng mã ngẫu nhiên hoặc cấp mã người mới cho mỗi file. Không sửa file MAT gốc.

## Dùng trong ứng dụng

1. Mở `D:\dh\da1\Mo_UngDung_ML.cmd`, vào tab 4 SEED-VIG.
2. Tab tự hiển thị hồ sơ dữ liệu cục bộ nếu có; nút “Dữ liệu đã tải” mở lại thông tin. Đây là snapshot kiểm tra, không phải xác nhận checksum được tính lại mỗi lần mở app.
3. Có thể bấm “Xem biến trong file MAT” để kiểm tra một file đã giải nén bằng chức năng sẵn có.
4. Khi đã xác minh ánh xạ người và căn hàng nhãn, tạo bản làm việc từ một manifest nháp, sửa đúng các trường còn thiếu. Không chọn cả ba manifest rồi xem chúng là ba nhóm người khác nhau.
5. Chốt quy tắc phân loại PERCLOS có căn cứ trước khi xem kết quả test, rồi nhập bằng tab 4 và đánh giá riêng với NeuroSky. Không có ngưỡng mặc định nào được tự áp dụng trong lần cập nhật này.

Việc tiếp theo thuộc phần xử lý dữ liệu và thiết kế thí nghiệm: hoàn thiện ánh xạ ID, quy tắc nhãn và cấu hình đánh giá. Người dùng không cần tải thêm raw hoặc tự tính đặc trưng từ đầu cho ba nhánh này.

## Chạy lại kiểm tra

```powershell
.\.venv\Scripts\python.exe scripts/prepare_seed_vig_local.py 'D:\dh\SEED-VIG (2).zip'
```

Script giữ ZIP gốc; từ chối ghi đè file MAT nếu nội dung khác. Chỉ giải nén các nhánh dùng ở trên, kiểm tra CRC khi đọc từng thành viên và giữ cấu trúc thư mục. Dữ liệu được đặt trong data/external, vốn đã bị .gitignore bỏ qua. Không đưa MAT hoặc dữ liệu dẫn xuất hạn chế vào Git công khai.

Nguồn: README đi cùng bản phát hành; [trang SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html); [bài gốc của Zheng và Lu](https://weilongzheng.github.io/publication/zheng2017multimodal/zheng2017multimodal.pdf).
