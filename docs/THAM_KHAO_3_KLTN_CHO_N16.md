# Tham khảo ba khóa luận EEG cho đồ án N16

Ngày đối chiếu: 08/10/2026. Phạm vi: báo cáo đầy đủ, bản tóm tắt và một số đoạn mã nguồn trong ba ZIP người dùng cung cấp. Chưa chạy lại huấn luyện, chưa tái lập các chỉ số của tác giả. Các số liệu dưới đây là kết quả được báo cáo, không phải kết quả N16.

## 1. Kết luận cho nhóm N16

Giữ bài toán **phát hiện buồn ngủ từ EEG NeuroSky và đặc trưng chớp mắt**, không chuyển sang trầm cảm, động kinh hoặc tưởng tượng vận động. Ba khóa luận hữu ích ở mức phương pháp và kiến trúc triển khai, nhưng không thay thế nghiên cứu trực tiếp về buồn ngủ.

Ưu tiên đọc: **động kinh → SoC → trầm cảm**. Giữ hướng hiện tại: Python huấn luyện/đánh giá/xuất mô hình; C++17 thu COM, giải mã ThinkGear, DSP, suy luận và cảnh báo. Chưa cần thêm FPGA, Raspberry Pi, ESP32 hay IMU vào phạm vi Đồ án 1 chỉ vì tài liệu tham khảo có sử dụng chúng.

## 2. Thông tin tài liệu

Tên tác giả lấy từ trang bìa, không lấy theo tên ZIP; hai tài liệu có đồng tác giả.

1. Nguyễn Minh Nhật (2025, năm trên bìa). *Đánh giá mức độ trầm cảm từ tín hiệu EEG dựa trên phương pháp kết hợp hệ suy luận mờ loại 2 và Fuzzy C-Means*. Khóa luận tốt nghiệp, Khoa Kỹ thuật máy tính, Trường Đại học Công nghệ Thông tin, ĐHQG TP.HCM.
2. Lê Quang Phúc và Cao Văn Quý (2026). *Hiện thực hệ thống SoC cho bài toán phân loại tín hiệu EEG sử dụng mạng nơ-ron tích chập*. Khóa luận tốt nghiệp, cùng khoa và trường trên.
3. Trương Công Sơn và Bùi Nguyễn Đại Thành (2026). *Nghiên cứu và triển khai mô hình phát hiện cơn động kinh từ EEG đơn kênh trong môi trường thời gian thực*. Khóa luận tốt nghiệp, cùng khoa và trường trên.

## 3. Bảng đối chiếu dùng cho khảo sát M1

| Tài liệu | Bài toán và cách làm | Kết quả tác giả báo cáo | Phần N16 có thể tham khảo | Khác biệt cần nêu |
|---|---|---|---|---|
| Trầm cảm | FAA, theta/alpha, entropy phổ; FCM và suy luận mờ loại 2 | Accuracy 88,99%; ma trận TP=59, TN=38, FP=7, FN=5 | Công thức đặc trưng phổ; cách trình bày mô hình và ma trận nhầm lẫn | FAA cần hai kênh Fp1/Fp2; không tính được đúng từ một kênh NeuroSky. Nhãn trầm cảm không phải nhãn buồn ngủ |
| SoC | CNN-1D phân loại tưởng tượng vận động trái/phải; PhysioNet; triển khai fixed-point 16-bit và RTL Verilog | Accuracy 84,64% với 64 kênh, 78,45% với 8 kênh | Tách mô hình huấn luyện khỏi runtime; mô hình C/C++ tham chiếu; đánh đổi độ chính xác/tài nguyên/độ trễ | Dữ liệu đa kênh và nhãn vận động, không phải buồn ngủ. Báo cáo còn đặt đánh giá end-to-end trên board với EEG stream trong hướng phát triển |
| Động kinh | CHB-MIT đơn kênh FP1-F7; đặc trưng phổ, MLP; pipeline EMA và ngưỡng ON/OFF; prototype NeuroSky, Raspberry Pi, ESP32, IMU | Test độc lập: AUC 0,9165, sensitivity 78,57%, specificity 91,42%; prototype: 3 phiên, 14,94 phút, P95 thời gian xử lý 29,953 ms | Luồng thu → lọc → cửa sổ → đặc trưng → scaler → suy luận → cảnh báo → log; tách đánh giá mô hình/replay/thiết bị thật | Kênh lâm sàng FP1-F7 khác điện cực NeuroSky; mẫu prototype không có nhãn động kinh lâm sàng. Không dùng các số này để dự đoán hiệu năng N16 |

Không xếp hạng ba mô hình bằng accuracy/AUC: bài toán, nhãn, dữ liệu và cách chia tập khác nhau.

## 4. Những điểm đã phát hiện khi đối chiếu code

### 4.1. Khóa luận trầm cảm

- Báo cáo định nghĩa TAR bằng công suất theta chia công suất alpha. Nhưng `MA NGUON/depression_fcm_it2fis.py`, dòng 68, dùng `bp_theta * (bp_alpha + EPS)`. Đây là điểm không khớp giữa code và công thức; không sao chép biểu thức này.
- Dòng 93 tính membership giữa của FAA bằng tích hai hàm clip có miền dương không giao nhau. Với FAA hữu hạn, kết quả thành 0; cần kiểm tra lại định nghĩa tập mờ nếu muốn nghiên cứu thuật toán này.
- Phần thực nghiệm nói 109 mẫu và 20 mẫu kiểm tra, nhưng ma trận nhầm lẫn cộng lại bằng 109. Cần làm rõ tập thực sự dùng tính chỉ số và vai trò của 20 mẫu trước khi coi 88,99% là kết quả test độc lập.

Những nhận xét này là kiểm tra tĩnh, không phải kết luận rằng toàn bộ kết quả khóa luận sai.

### 4.2. Khóa luận động kinh

- `source/stage1/src/train.py`, dòng 240, fit scaler trên toàn bộ tập train; dòng 250 tăng mẫu; sau đó dòng 256–259 mới chia `StratifiedKFold` theo cửa sổ. Vì vậy 5-fold CV có nguy cơ rò rỉ từ thống kê chuẩn hóa và mẫu tăng cường gần mẫu gốc. Cách chia này không tương đương LOSO theo người.
- Tập test bệnh nhân độc lập phải được phân biệt với 5-fold CV trên train; nhận xét trên không tự động phủ định kết quả test độc lập.
- `source/stage2/eeg/sichiray_collector.py` có tần số thiết bị 512 Hz, đích 256 Hz và `resample_poly(..., up=1, down=2)` ở dòng 226. Đây là ví dụ cần đồng nhất tần số đầu vào và mô hình; N16 đang dùng raw 512 Hz nên không đổi thông số một cách riêng lẻ.
- Dữ liệu CHB-MIT có nhãn và dữ liệu prototype thật không có nhãn lâm sàng được tác giả phân biệt rõ. N16 cũng phải phân biệt dữ liệu giả lập, replay có nhãn và dữ liệu thu thật có nhãn KSS/blink.

### 4.3. Khóa luận SoC

Không viết rằng hệ thống đã được kiểm chứng hoàn chỉnh từ EEG thật đến kết quả trên board. Phần hướng phát triển còn yêu cầu đo end-to-end, công suất thực và độ ổn định liên tục; cần đọc cùng các phần mô phỏng, timing và tích hợp để xác định đúng mức kiểm chứng.

## 5. Áp dụng vào M1 mà không mở rộng phạm vi

1. Đưa ba khóa luận vào nhóm **nghiên cứu trong nước liên quan phương pháp xử lý/triển khai EEG**; không gọi chúng là ba nghiên cứu nhận diện buồn ngủ.
2. Vẫn giữ các nguồn trực tiếp về buồn ngủ, KSS, EEG một kênh, blink và giao thức NeuroSky. Hỏi GVHD nếu yêu cầu nguồn tin cậy không chấp nhận khóa luận chưa có bản công khai hoặc chưa phản biện.
3. Mỗi dòng khảo sát phải ghi bài toán, thiết bị/kênh, đặc trưng, mô hình, cách chia test, chỉ số và khác biệt với N16.
4. Lập luận chọn baseline gọn, dễ xuất sang C++ và kiểm chứng; không dùng độ chính xác của bài toán khác để hứa hẹn kết quả N16.
5. Chuẩn bị kiểm thử Python–C++ trên cùng dữ liệu: thứ tự đặc trưng, scaler, xác suất và quyết định cảnh báo; đo thời gian xử lý riêng với thời gian tích lũy cửa sổ.
6. Đánh giá theo người; fit scaler và mọi bước tăng mẫu chỉ trong phần train của từng fold. Đánh giá blink riêng bằng nhãn tay, mô hình buồn ngủ bằng nhãn thật phù hợp.
7. Với blink, phân biệt sự kiện chớp mắt được khai thác làm đặc trưng với đoạn EEG bị nhiễu chuyển động/tiếp xúc kém; không bê nguyên luật loại artifact của bài toán động kinh.
8. Chưa có thiết bị: chỉ báo cáo trạng thái phần mềm/mô phỏng đã kiểm thử; chưa công bố accuracy, F1 hoặc cảnh báo sai/giờ như kết quả thực nghiệm trên người.

### Đoạn tổng hợp có thể biên tập đưa vào M1

Các khóa luận trong nước được khảo sát cho thấy những hướng tiếp cận khác nhau đối với xử lý EEG, gồm đặc trưng phổ kết hợp suy luận mờ, CNN-1D triển khai trên SoC và mô hình đơn kênh tích hợp vào pipeline cảnh báo thời gian thực. Tuy nhiên, các công trình này giải quyết trầm cảm, tưởng tượng vận động hoặc động kinh, nên kết quả không thể chuyển trực tiếp sang bài toán buồn ngủ. Nhóm N16 tham khảo cách trích xuất đặc trưng, phân tách huấn luyện và triển khai, kiểm chứng mô hình trên runtime và tổ chức tầng cảnh báo; đồng thời lựa chọn thiết kế phù hợp với EEG NeuroSky một kênh, đặc trưng chớp mắt và đánh giá theo người. Hiệu năng nhận diện buồn ngủ cần được kiểm chứng bằng dữ liệu và nhãn của chính bài toán N16.

Đoạn trên là tổng hợp của nhóm, không phải trích dẫn nguyên văn. Khi sử dụng, thêm trích dẫn đến ba tài liệu theo mẫu GVHD yêu cầu. Không sao chép code khi chưa rõ giấy phép/quyền sử dụng.

## 6. Vị trí đọc lại

Số trang dưới đây là thứ tự trang trong PDF, không phải số in ở chân trang.

- Trầm cảm: PDF trang 27–29 (đặc trưng), 51–54 (mẫu, ma trận và chỉ số).
- SoC: PDF trang 76 (bảng khảo sát số kênh), 95 (giới hạn/hướng kiểm chứng end-to-end).
- Động kinh: PDF trang 38–39 (cửa sổ/đặc trưng/scaler), 46 (thu và lọc), 61–62 (CV/test), 67–68 (prototype và giới hạn kết luận).

Các báo cáo đã trích ra để đọc tại:

- [Báo cáo trầm cảm](../tmp/tham_khao_kltn/tram_cam/BAN%20BAO%20CAO%20KLTN.pdf)
- [Báo cáo SoC](../tmp/tham_khao_kltn/soc/KLTN-22521119-22521218.pdf)
- [Báo cáo động kinh](../tmp/tham_khao_kltn/dong_kinh/Bản%20báo%20cáo%20KLTN.pdf)

Thư mục `tmp` chỉ là bản trích để kiểm tra, không nên nộp cả thư mục này lên GitHub. Giữ ZIP gốc làm nguồn lưu trữ; không đưa toàn bộ khóa luận/mã nguồn của người khác vào repo công khai nếu chưa được phép.
