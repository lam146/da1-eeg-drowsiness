# Khung bảng khảo sát M1

M1 cần ít nhất 8 nguồn tin cậy. Không chỉ liệt kê tài liệu; mỗi nguồn phải được so sánh theo cùng một bộ tiêu chí.

| Nguồn | Thiết bị/số kênh | Bối cảnh và nhãn | Đặc trưng | Mô hình | Kiểu chia test | Chỉ số | Điểm mạnh | Hạn chế/liên hệ N16 |
|---|---|---|---|---|---|---|---|---|
| NeuroSky ThinkGear Guide | MindWave, 1 kênh trán, raw 512 Hz | Không có nhãn | Raw/eSense | Không có | Không có | Không có | Giao thức chính thức | Chỉ là tài liệu thiết bị |
| Åkerstedt & Gillberg (1990) | EEG/EOG | KSS, thức qua đêm | Alpha/theta, EOG | Phân tích thống kê | Theo người/điểm đo | Tương quan | Nền tảng KSS | Mẫu nhỏ, không phải NeuroSky |
| Kaida et al. (2006) | EEG + PVT | KSS | Alpha/theta | Phân tích thống kê | Lặp theo ngày | Tương quan | Xác thực KSS với EEG/hành vi | Không phải bài toán phân loại real-time |
| Lal & Craig (2001) | Tổng quan đa phương thức | Mệt mỏi lái xe | Delta/theta và chỉ báo sinh lý | Tổng quan | - | - | Khung lý thuyết | Cũ, không đánh giá NeuroSky |
| Kim et al. (2016) | EEG trán 1 kênh | Nhãn blink thủ công | Lọc + luật | Rule-based | Tập 24 người | Detection metrics | Sát bài toán blink N16 | Cần hiệu chỉnh vị trí/thiết bị |
| Cao et al. (2019) | 32 kênh, 500 Hz | Lái mô phỏng, phản ứng lệch làn | Raw/EEG | Dataset | 27 người | Phản ứng | Dataset công khai có hành vi | Khác thiết bị và vị trí kênh |
| Cui et al. (2022) | EEG 1 kênh | Buồn ngủ lái xe | Raw/alpha spindle/theta burst | Compact CNN | Cross-subject | Accuracy | Chứng minh khả năng 1 kênh | Deep model, cần dữ liệu lớn |
| Cui et al. (2021) | EEG 1 kênh | Buồn ngủ | Raw/time sequence | CNN-LSTM | LOSO | Accuracy | Subject-independent, giải thích được | Phức tạp hơn phạm vi ĐA1 |
| Zheng & Lu (2017) | EEG + forehead EOG | Lái mô phỏng, vigilance | Đa phương thức | Regression/domain model | Cross-subject | Correlation/RMSE | Hợp nhất EEG và EOG | N16 chỉ có một cảm biến trán |
| Welch (1967) | Tín hiệu tổng quát | Không có | PSD Welch | Ước lượng phổ | - | Bias/variance | Cơ sở phương pháp phổ | Không trực tiếp về buồn ngủ |

## Khoảng trống dự kiến để kiểm chứng ở M1

- Nhiều nghiên cứu dùng EEG nhiều kênh hoặc thiết bị phòng thí nghiệm; N16 cần kiểm chứng mức khả dụng của một kênh trán giá thấp.
- Cross-window ngẫu nhiên dễ cho kết quả cao giả tạo; N16 dùng LOSO theo người.
- Artefact chớp mắt thường bị loại, nhưng N16 khai thác nó như nguồn đặc trưng bổ sung và phải đánh giá riêng bằng nhãn tay.
- Mục tiêu chính của Đồ án 1 là lập luận lựa chọn và prototype đo được, không chạy theo mô hình deep learning khi dataset nhỏ.
