# Kế hoạch lấy mẫu hằng ngày cho nhóm N16

Ngày lập 23/10/2026. Đây là lịch đề xuất, chưa phải lịch hẹn với người tham gia hoặc kết quả đã thu. Mục tiêu là chuẩn bị dữ liệu có thể kiểm tra và tái hiện cho M2, không tạo đủ nhãn bằng cách yêu cầu người tham gia buồn ngủ.

## 1. Ba loại dữ liệu phải để riêng

| Loại | Mục đích | Có dùng làm nhãn buồn ngủ không |
|---|---|---|
| TECH hoặc PILOT | Kiểm tra kết nối, CSV, đồng bộ và chất lượng | Không mặc định; không gộp vào đánh giá chính thức sau khi đã dùng chỉnh pipeline |
| CAL hoặc BLINK | Hiệu chuẩn, kiểm tra chớp mắt chủ động hoặc tự nhiên có video | Không dùng chớp/nhắm mắt theo lệnh làm nhãn buồn ngủ |
| MAIN | Mô phỏng lái xe có KSS, nhật ký sự cố và raw | Chỉ cửa sổ đủ chất lượng, thuộc vùng nhãn đã khóa |

File test1.csv hiện có giữ nguyên và ghi loại TECH_BLINK_TEST. Người dùng cho biết đã thử chớp mắt liên tục; file chưa có nhật ký sự kiện đủ để chấm precision/recall. Không tạo ngược nhãn chớp mắt từ đỉnh của chính thuật toán.

Tất cả bảng minh họa trong tài liệu là cách ghi, không phải số đo mới. Mã S01-S08 và lịch bên dưới là kế hoạch. Chỉ điền dữ liệu thực sau từng phiên. Không tự tạo KSS, video, thời điểm blink hoặc chất lượng để lấp ô trống.

## 2. An toàn và đồng ý tham gia

Chỉ thu trong môi trường mô phỏng có người giám sát. Không đo khi đang lái xe thật, không yêu cầu thiếu ngủ, uống thuốc hoặc thay đổi caffeine để ép xuất hiện lớp buồn ngủ. Không yêu cầu nhịn chớp mắt. Nếu khó chịu, chóng mặt, mệt quá mức hoặc muốn dừng, kết thúc ngay và lưu lý do; không thu kéo dài để đủ nhãn.

Giải thích mục đích, dữ liệu lưu và quyền rút lui trước khi thu. Video khuôn mặt cần đồng ý riêng. Nếu không đồng ý video, vẫn có thể thu MAIN nếu protocol cho phép, nhưng không coi phiên đó có ground truth blink chính xác theo video. Chốt người được truy cập, nơi lưu và thời hạn xóa dữ liệu với GVHD/nhóm trước thu. Không công khai video hoặc bảng nối mã với tên thật.

## 3. Lịch theo ngày

Đề xuất 8 người, mỗi người 2 phiên MAIN ở hai ngày khác nhau, mỗi phiên 30 phút nhiệm vụ. Đây là quy mô kế hoạch, không bảo đảm đủ hai lớp hoặc kết luận tổng quát. Có thể đổi lịch theo người tham gia; giữ đúng mã người và ghi giờ thực. Không đổi giờ chỉ để ép KSS tăng.

| Ngày | Người / loại phiên | Công việc | Sản phẩm cần có |
|---|---|---|---|
| 24/10 | Người vận hành / TECH | Kiểm tra tai nghe, cột CSV, fs, đơn vị, đường chất lượng; thu 2-3 đoạn ngắn | Raw kiểm tra, cấu hình xuất, log lỗi, quyết định đường thu |
| 25/10 | S01 và S02 / PILOT, BLINK_CAL | Mỗi người thử ngắn 5-10 phút; kiểm tra quy trình KSS và mốc video | Nhật ký pilot, đoạn chớp có nhãn, lỗi cần sửa; chưa tính MAIN |
| 26/10 | S01 P01, S02 P01 / MAIN | Hai phiên chính thức theo cùng protocol | 2 bộ hồ sơ phiên |
| 27/10 | S03 P01, S04 P01 / MAIN | Như trên | 2 bộ hồ sơ phiên |
| 28/10 | S05 P01, S06 P01 / MAIN | Như trên | 2 bộ hồ sơ phiên |
| 29/10 | S07 P01, S08 P01 / MAIN | Như trên | 2 bộ hồ sơ phiên |
| 30/10 | S01 P02, S02 P02 / MAIN | Phiên thứ hai; giữ ID cũ, tên phiên mới | 2 bộ hồ sơ phiên |
| 31/10 | S03 P02, S04 P02 / MAIN | Như trên | 2 bộ hồ sơ phiên |
| 01/11 | S05 P02, S06 P02 / MAIN | Như trên | 2 bộ hồ sơ phiên |
| 02/11 | S07 P02, S08 P02 / MAIN | Như trên | 2 bộ hồ sơ phiên |
| 03/11 | Không bắt buộc thu mới | Hoàn tất gán nhãn, kiểm tra chất lượng và manifest | Dataset và bảng số liệu theo người/lớp; giữ cả phiên không dùng |
| 04/11 | Phân tích | Tạo feature theo protocol đã khóa; chạy baseline LOSO nếu đủ điều kiện | Metrics từng người, OOF, danh sách fold thiếu lớp |
| 05/11 | Tổng hợp | So sánh phổ/blink/kết hợp, kiểm tra lỗi và chuẩn bị M2 | Kết quả thật, demo và giới hạn; không đổi nhãn sau xem test |

Nếu đủ 16 phiên MAIN ×30 phút nhiệm vụ thì có 480 phút =8 giờ nhiệm vụ danh định. Không phải toàn bộ 8 giờ đều có nhãn hoặc đủ chất lượng. Chưa tính kiểm tra thiết bị, hiệu chuẩn, trả lời kéo dài, nghỉ và xử lý sau thu. Dự trù khoảng 50-60 phút/người cho vận hành, thêm thời gian gán nhãn video riêng; đây là dự toán công việc, không phải giới hạn buộc người tham gia phải hoàn tất.

Ngày kỹ thuật/pilot là điểm kiểm soát. Nếu chưa xác nhận đơn vị, chất lượng hoặc đồng bộ, sửa trước khi chạy hàng loạt. Có thể lùi lịch, giảm phạm vi và báo GVHD; không ghi mọi phiên là đạt khi thiếu điều kiện.

## 4. Vai trò của hai thành viên

Người vận hành A thiết lập kết nối, cấu hình và ghi raw, theo dõi trạng thái, đặt tên file. Người vận hành B phụ trách đồng ý tham gia, hướng dẫn KSS, đồng hồ/sự kiện, video nếu được phép và kiểm tra hồ sơ. Khi chính một thành viên là người được đo, người còn lại điều khiển; không để người đang thực hiện nhiệm vụ tự ghi toàn bộ nhật ký.

Không coi hai thành viên đo lặp nhiều lần là đủ 8 người. Mỗi người thật có một subject_id ổn định trong tất cả các ngày.

## 5. Mỗi lần thu cần làm gì

### Bước 1 Chuẩn bị trước khi đeo

1. Xác nhận đồng ý tham gia và đồng ý video riêng.
2. Ghi mã người, phiên, ngày giờ, số giờ ngủ tự báo, thông tin caffeine tự báo, nhiệm vụ, người vận hành và phiên bản protocol.
3. Tạo tên file mới. Ví dụ S03_20261027_P01_MAIN.csv; các file cùng phiên dùng cùng tiền tố.
4. Kiểm tra pin, điện cực, adapter, COM và thư mục có quyền ghi. Không dùng tên test1.csv lặp qua nhiều người.
5. Ghi software/driver, fs cấu hình 512 Hz, tên cột, đơn vị xuất và đường chất lượng. Không dùng COM4 trong ảnh hướng dẫn nếu máy thực là cổng khác.

### Bước 2 Kiểm tra kết nối khoảng 2-3 phút

Đeo đúng hướng dẫn thiết bị, kiểm tra tiếp xúc trán và kẹp tai, tránh tóc nằm dưới cảm biến. Mở tuyến OpenViBE theo hướng dẫn GVHD hoặc tuyến COM trực tiếp đã kiểm tra; không cho hai ứng dụng tranh một COM.

Kiểm tra tín hiệu biến thiên, ghi file được và trạng thái tiếp xúc. Nếu chỉnh tai nghe hoặc có xung lớn thì ghi thời điểm. Không tuyên bố sạch chỉ vì đồ thị đẹp hoặc timestamp đều. Nếu thiết bị báo không tiếp xúc, sửa và ghi lại đoạn lỗi thay vì gán tỉnh.

### Bước 3 Ghi CAL riêng khoảng 3 phút

| Mốc trong file CAL | Thực hiện | Lưu ý |
|---|---|---|
| 0-30 s | Ngồi yên, mở mắt, chớp tự nhiên | Không cố nhịn chớp |
| 30-60 s | Nhắm mắt thoải mái nếu đồng ý | Chỉ là hiệu chuẩn; không gán buồn ngủ |
| 60-90 s | Mở mắt, chớp tự nhiên | Ghi chuyển trạng thái |
| 90-150 s | Chớp nhẹ một lần khoảng mỗi 5 s | Ghi thời điểm thực từ video, không lấy lịch lệnh làm nhãn |
| 150-180 s | Chớp tự nhiên trở lại | Đánh dấu kết thúc CAL |

Chớp mạnh/liên tục có thể làm các xung chồng nhau hoặc chạm giới hạn, khó đánh giá số sự kiện. Ưu tiên chớp nhẹ, có khoảng cách ở CAL. Không đặt trước “12 lệnh =12 chớp thật”; phải quan sát thực tế. Không đưa CAL vào phân loại tỉnh/buồn ngủ.

### Bước 4 Đồng bộ EEG và video

Quy ước t_EEG=0 là đầu file EEG; t_video=0 là đầu video. Đánh dấu mốc chung đầu và cuối bằng phương pháp nhìn thấy/ghi nhận ở cả hai nguồn và lưu cách thực hiện. Nếu dùng chuỗi chớp mắt đặc trưng chỉ để ước lượng căn chỉnh, phải ghi đó là sync, loại khỏi đánh giá và không dùng detector cần chấm để tự tạo ground truth.

Ghi ánh xạ thời gian, ví dụ t_EEG=a*t_video+b, độ lệch đầu/cuối và độ bất định. Nếu chỉ có stopwatch/bấm tay chưa kiểm tra độ chính xác, ghi sync chưa xác nhận. Không chấm sự kiện với dung sai +/-200 ms như bằng chứng chính xác khi độ bất định đồng bộ đã lớn tương đương. Khi không đủ đồng bộ, video vẫn hỗ trợ xem lại/count nhưng phải giới hạn kết luận về thời điểm.

### Bước 5 Bắt đầu MAIN riêng

Ghi KSS ban đầu theo cùng một bảng thang đo được GVHD thống nhất. Mở file MAIN mới, đánh dấu MAIN_START và mốc đồng bộ. Thực hiện nhiệm vụ mô phỏng đơn điệu 30 phút có giám sát; không thử nghiệm trên đường thật.

Phân biệt thời gian trong file EEG và thời gian nhiệm vụ. Nếu MAIN_START nằm ở t_EEG=12,4 s thì t_task=0 tại đó, không tự sửa raw cho bắt đầu 0. Lưu main_start_eeg_s=12,4 trong metadata (đây chỉ là ví dụ). Mọi nhật ký nên có t_EEG; có thể thêm t_task=t_EEG-main_start_eeg_s. Không ghép nhật ký của file CAL sang MAIN khi không đổi mốc.

### Bước 6 KSS mỗi 5 phút

| Mốc nhiệm vụ dự kiến | Việc làm |
|---|---|
| Đầu phiên | Ghi KSS ban đầu và thời gian thực |
| 5 phút | Hỏi KSS, ghi giờ bắt đầu hỏi và kết thúc trả lời |
| 10 phút | Như trên |
| 15 phút | Như trên |
| 20 phút | Như trên |
| 25 phút | Như trên |
| 30 phút | Ghi KSS cuối rồi kết thúc và kiểm tra file |

Đây là lịch nhắc, không phải thời gian nhãn được đo chính xác. Nếu hỏi muộn 15 s, ghi 315 s chứ không sửa thành 300 s. Đồng hồ nhiệm vụ và file vẫn chạy liên tục qua lúc trả lời; đánh dấu đoạn hỏi/trả lời để loại. Nếu tạm dừng nhiệm vụ, ghi PAUSE/RESUME nhưng không tự xóa thời gian khỏi EEG. Giữ ghi EEG đến khi kết thúc trả lời cuối, nên raw có thể dài hơn 30 phút.

Người tham gia tự đánh giá, không gợi ý số điểm phải tăng dần. Có thể cả phiên đều tỉnh. Caffeine/giấc ngủ/giờ đo là metadata, không thay KSS. Nhãn dự kiến 1-5=tỉnh, 7-9=buồn ngủ, 6=chuyển tiếp, cần GVHD duyệt. Không dùng Attention/Meditation làm nhãn thật.

### Bước 7 Trong lúc đo

Ghi bắt đầu/kết thúc khi chỉnh tai nghe, mất tiếp xúc/kết nối, nói chuyện, nghỉ hoặc thay nhiệm vụ. Không nhắc chớp mắt theo nhịp trong MAIN; để blink tự nhiên. Nếu đang thu chỉ để quan sát mà phát cảnh báo liên tục, cảnh báo có thể làm thay đổi trạng thái; tắt can thiệp ở thu baseline hoặc ghi rõ điều kiện cảnh báo nếu protocol yêu cầu bật. Không trộn hai điều kiện mà không ghi.

### Bước 8 Kết thúc và sao lưu

Ghi KSS cuối, MAIN_END, lý do kết thúc và mốc đồng bộ cuối. Dừng ghi đúng phần mềm, chờ file đóng, kiểm tra có thể mở. Giữ bản raw gốc bất biến, không mở rồi save đè bằng Excel. Kiểm tra số dòng, thời gian, tín hiệu, sự kiện/nhãn và đường dẫn video. Sao lưu vào nơi được phép trước khi người tham gia rời đi.

## 6. Bộ dữ liệu phải ghi chi tiết

Tên một bộ phiên minh họa: S03_20261027_P01. Dưới đây là schema, không phải các file đã thu. Có thể ghi bằng bảng tính hoặc nhật ký rồi chuyển CSV; các header cần giữ rõ. Không để tên thật trong file dùng phân tích.

### A Metadata một dòng mỗi phiên

| Trường | Nội dung cần ghi |
|---|---|
| subject_id, session_id, session_type | Mã người, mã phiên và TECH/CAL/MAIN |
| date_local, start_local, timezone | Ngày giờ thực; Asia/Bangkok hoặc UTC+07:00 |
| operator_id, protocol_version | Người vận hành và protocol đã khóa |
| device_model, device_id, software_version | Thiết bị/adapter/phần mềm; không biết thì ghi unknown |
| acquisition_route, com_port, fs_hz | OpenViBE hoặc COM; cổng thực; cấu hình 512 |
| signal_column, signal_unit, unit_verified | Electrode; uV/count/unknown; bằng chứng đơn vị |
| quality_source | Raw poorSignal hay trạng thái/log khác; không có thì ghi missing |
| eeg_file, video_file, event_file, kss_file | Đường dẫn tương đối đến file thật |
| consent_record_id, video_consent | Tham chiếu hồ sơ đồng ý; yes/no |
| sleep_hours_previous_night, caffeine_note | Người tham gia tự báo; thiếu thì để missing |
| task_description, main_start_eeg_s, main_end_eeg_s | Mô tả nhiệm vụ, mốc thực trong EEG |
| sync_method, sync_uncertainty_ms | Phương pháp và độ bất định đã đo; chưa đo ghi unknown |
| end_reason, notes | Hoàn tất, rút lui, sự cố; ghi trung tính |

### B Raw một hàng mỗi mẫu

Giữ CSV OpenViBE gốc với Time:512Hz, Epoch, Electrode và các cột thiết bị xuất. N16 lấy Electrode làm tín hiệu để xử lý; Delta/Theta/Alpha của hãng không thay cho raw. Chỉ ghi đơn vị uV khi cấu hình/đường xuất đã được xác nhận, không suy từ tên cột hoặc biên độ.

512 Hz danh định tương ứng 30 s=15.360 mẫu; 60 s=30.720; 180 s=92.160; 1.800 s=921.600. Đây là phép tính lý thuyết, không phải số mẫu mọi file bắt buộc giống hệt vì raw có thể gồm thời gian thêm hoặc gián đoạn. Timestamp đều có thể do phần mềm tạo; cần kiểm tra thêm trạng thái truyền và chất lượng.

### C KSS một hàng mỗi lần tự đánh giá

```text
subject_id,session_id,target_task_s,request_eeg_s,response_end_eeg_s,kss,annotator_id,notes
```

target_task_s là lịch nhắc; request_eeg_s/response_end_eeg_s là thời điểm thật trong file MAIN. kss là số người tham gia trả lời, không tự suy từ raw. Ghi cả khoảng trả lời để loại cửa sổ giao với nó. KSS ban đầu chưa có vùng 60 s trước thuộc MAIN thì chỉ giữ như metadata, không tự kéo nhãn về CAL.

Ví dụ giả định chỉ minh họa: target_task_s=300, request_eeg_s=314, response_end_eeg_s=323, kss=4. Ý nghĩa: thời điểm ghi không phải đúng giây 300. Không chép các số này vào dữ liệu thật.

### D Sự cố một hàng mỗi khoảng

```text
subject_id,session_id,start_eeg_s,end_eeg_s,event_type,source,notes
```

event_type có thể dùng HEADSET_ADJUST, CONTACT_LOSS, CONNECTION_GAP, KSS_RESPONSE, TALKING, TASK_PAUSE, SYNC. source ghi người vận hành/video/device_log. Ghi end thực; chưa biết end thì để missing và xử lý, không mặc định bằng start.

### E Blink một hàng mỗi sự kiện nhãn tay

```text
subject_id,session_id,blink_id,video_onset_s,video_offset_s,eeg_onset_s,eeg_offset_s,reference_time_ms,reference_kind,annotator_id,confidence,notes
```

Chốt reference_kind, ví dụ onset hoặc một mốc trung tâm được định nghĩa rõ. Detector hiện trả đỉnh xung EEG; đỉnh điện và onset video không nhất thiết đồng thời. Phải thống nhất phép đối chiếu và dung sai phù hợp, không gọi thời gian đỉnh điện là onset video thật. Có thể giữ onset/offset để kiểm tra đóng mắt riêng. Không tự tạo onset/offset bằng nhịp ra lệnh.

confidence có thể là clear/uncertain với định nghĩa trước gán nhãn; không dùng để tự loại các dự đoán sai sau khi biết kết quả. Nếu có hai người gán nhãn, lưu bản gán độc lập và cách giải quyết bất đồng. Tổng số blink đếm tay chỉ hỗ trợ count, chưa đủ TP/FP/FN theo sự kiện.

### F Chất lượng một hàng mỗi khoảng đã xem

```text
subject_id,session_id,start_eeg_s,end_eeg_s,quality_source,raw_poor_signal,quality_status,reason,reviewer_id
```

raw_poor_signal chỉ điền khi là giá trị thật từ ThinkGear. quality_status có thể usable/exclude/unknown theo protocol; trạng thái thủ công không được giả thành poorSignal=0. Ví dụ ghi khoảng nghi clipping, mất tiếp xúc hoặc nhiễu vận động bằng lý do cụ thể. Không xóa cửa sổ chỉ vì nó làm accuracy giảm.

### G Kiểm kê hằng ngày

```text
date_local,subject_id,session_id,session_type,duration_s,samples,kss_records,video_available,sync_verified,quality_available,review_status,exclusion_reason
```

Mỗi phiên có một dòng kể cả bị lỗi/rút lui. Không đánh dấu usable nếu mới chỉ mở CSV được. Giữ missing/unknown khác 0 và no. Không lập bảng tất cả check=true sẵn trước khi đo.

## 7. Điều kiện được chuyển sang tạo feature

1. Xác nhận người, phiên, fs, đơn vị và raw gốc; MAIN tách CAL.
2. Đồng bộ nhật ký KSS/sự cố vào đúng thời gian file; video cần xác minh riêng nếu chấm blink.
3. Có quy tắc chất lượng và ghi nhận thực. CSV test1 không có poorSignal; không tự thêm cột toàn 0.
4. Khóa quy tắc gán nhãn trước chạy test. Đề xuất M1 là tối đa 60 s sạch trước request KSS, chỉ lấy cửa sổ nằm hoàn toàn trong vùng đó, không giao lỗi hoặc khoảng trả lời. Đây là đề xuất cần duyệt, không phải kết luận mọi phút trước KSS đều đồng nhất.
5. KSS=6 hoặc thiếu nhãn giữ ngoài baseline; không gán cả phiên bằng KSS cuối. Mọi mẫu ngoài vùng nhãn vẫn giữ raw để có thể xem lại.

Ví dụ giả định: request ở 314 s, dùng vùng trước 60 s thì vùng ứng viên [254,314) s; nếu chất lượng loại [280,290), không lấy cửa sổ giao đoạn này. Cửa sổ 4 s phải nằm trọn vùng sạch. Không dùng khoảng trả lời [314,323) s.

**Giới hạn code hiện tại:** featurize.py còn forward-fill KSS đến 5 phút và lấy nhãn cuối cửa sổ; nó chưa thực thi đầy đủ vùng 60 s trước request, loại sự kiện và kiểm tra nhãn toàn cửa sổ ở trên. Không đưa nhật ký vào rồi coi lệnh hiện tại tự áp đúng protocol mới. Cần hoàn thiện bước ánh xạ nhãn/chất lượng sau khi GVHD chốt trước tạo dataset thực nghiệm. Thu raw và nhật ký đầy đủ từ bây giờ giúp không phải đo lại vì thiếu mốc.

CSV OpenViBE nhập bằng openvibe_import để kiểm tra cấu trúc, không tự sinh chất lượng hoặc KSS. Trường hợp chất lượng thủ công cần adapter/logic riêng, chưa có sẵn chỉ bằng đặt tên cột. Người dùng được hướng dẫn bước thu và lưu; tài liệu này không tự thay code/protocol cũ.

## 8. Áp dụng vào báo cáo và mô hình

| Bằng chứng | Đưa vào phần nào | Không được suy ra |
|---|---|---|
| CSV pilot, cấu hình, số mẫu và lỗi | M2 module thu dữ liệu trên thiết bị | Model đạt accuracy hoặc tai nghe không mất mẫu chỉ vì dt đều |
| Video đồng bộ và nhãn blink độc lập | Đánh giá precision/recall/F1, lỗi đếm | Có thể phát hiện mọi nhắm mắt lâu từ độ rộng xung |
| MAIN + KSS + cờ chất lượng | Dataset và phân loại theo người | Chớp chủ động/nhắm mắt theo lệnh là buồn ngủ |
| Feature phổ/blink/cả hai cùng cửa sổ | Ablation, lập luận fusion | Fusion chắc chắn tốt hơn trước thí nghiệm |
| Toàn bộ phiên theo subject_id | LOSO và bảng chỉ số từng người | Nhiều cửa sổ thay cho nhiều người độc lập |
| Mốc sự kiện, quyết định và âm thanh | Thực nghiệm độ trễ phù hợp | Timing tính toán là toàn bộ trễ <3 s |

Một phiên MAIN 30 phút có thể có 6 lần KSS sau bắt đầu tại các mốc dự kiến 5..30 phút, ngoài KSS ban đầu. Số vùng có nhãn còn phụ thuộc mức KSS, chất lượng và protocol. Không tính 30 phút raw thành 30 phút ground truth mặc định.

Tạo feature từng phiên riêng sau khóa gán nhãn; ghép bảng có cùng cấu hình/đơn vị/nguồn. LOSO giữ toàn bộ mọi phiên của người test ra khỏi train. Người test một lớp vẫn được ghi lại; AUC thiếu lớp để N/A. Không ép đủ hai lớp bằng thay nhãn; nếu thiếu lớp báo giới hạn và trao đổi phạm vi với GVHD. Dataset SEED-VIG dùng PERCLOS, đánh giá riêng; không trộn nhãn KSS thành cùng kết quả.

Giữ CAL/PILOT dùng chỉnh detector riêng với đoạn đánh giá chính thức. Nếu chọn siêu tham số theo người, đặt bước tuning bên trong train của LOSO hoặc ghi rõ đánh giá chỉ mang tính phát triển. Không nhìn toàn bộ test rồi điều chỉnh và báo lại như test độc lập.

## 9. Công việc cuối mỗi ngày

- Kiểm kê mọi phiên, không chỉ phiên đẹp; ghi thiếu file/nhãn/sync/chất lượng.
- Xem một số đoạn đầu, giữa, cuối và các đoạn bị log lỗi; phân tích định lượng không thay việc xem nguồn/video.
- Sao lưu raw/video theo quyền đã thống nhất; không đưa lên GitHub công khai.
- Điền metadata và KSS khi còn nhớ; nếu quên thì ghi missing, không hồi tưởng thành số chính xác.
- Gán nhãn một phần video theo định nghĩa đã khóa, lưu người gán và phiên bản.
- Cập nhật số người/phiên thực thu, phút raw, phút có nhãn, phút bị loại và phân bố KSS. Chưa tính accuracy chỉ để quyết định hôm sau sửa nhãn.

## 10. Tài liệu dự án liên quan

- HDSD_NeuroSky_MindWave.docx của GVHD: cách kết nối/xuất OpenViBE theo bộ thiết bị được cung cấp.
- docs/AP_DUNG_TAI_LIEU_GVHD.md: lệnh nhập CSV/MAT và scikit-learn; giới hạn hiện thực.
- src/n16_eeg/featurize.py: hành vi gán nhãn hiện tại cần đối chiếu trước áp protocol mới.
- docs/PROTOCOL_THU_DU_LIEU.md: protocol trước đó; tài liệu kế hoạch này là đề xuất vận hành chi tiết, chưa tự thay thế bản đã duyệt.
- https://developer.neurosky.com/docs/doku.php?id=thinkgear_communications_protocol : định nghĩa raw/gói/chất lượng.
- https://pubmed.ncbi.nlm.nih.gov/16679057/ : nghiên cứu KSS, không xác thực riêng cutoff hoặc vùng 60 s của N16.

Trước ngày MAIN đầu tiên, cần nhóm/GVHD xác nhận: protocol version; nhãn KSS và vùng thời gian; nguồn chất lượng; đồng bộ và mốc blink; phân biệt hiệu chuẩn/test; kế hoạch bảo mật. Ngày trong lịch là đề xuất quản lý công việc, không phải yêu cầu sinh lý hoặc quy định của nhà sản xuất.
