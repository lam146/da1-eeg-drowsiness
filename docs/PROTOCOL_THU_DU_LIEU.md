# Quy trinh thu du lieu N16

## 1. Nguyen tac an toan

- Chi thu trong mo phong lai xe, tuyet doi khong thu khi nguoi tham gia dang dieu khien xe that.
- Khong yeu cau thieu ngu chu dong. Uu tien dau gio chieu hoac mot phien mo phong don dieu tu 30-45 phut.
- Co nguoi giam sat; dung ngay neu nguoi tham gia met qua muc, dau dau, chong mat, kho chiu hoac muon rut lui.
- Giai thich muc dich, du lieu thu va quyen rut lui; lay dong y tu nguyen truoc khi thu.
- Ma hoa nguoi tham gia bang S01, S02...; khong ghi ten that trong file EEG.

## 2. Quy mo de xuat

- Toi thieu 8 nguoi, moi nguoi 2 phien o hai thoi diem khac nhau.
- Moi phien 30-45 phut, KSS moi 5 phut.
- Moi nguoi co du ca hai lop: tinh va buon ngu. Neu mot nguoi chi co mot lop, fold LOSO cua nguoi do khong du dieu kien tinh ROC-AUC.

## 3. Truoc moi phien

1. Ghi ma nguoi, ma phien, thoi gian, so gio ngu dem truoc, caffeine 6 gio gan nhat.
2. Lap tai nghe dung vi tri Fp1, dam bao tiep xuc va `poor_signal = 0` hoac on dinh o muc thap.
3. Thu 60 giay baseline: 30 giay mo mat, 30 giay nham mat.
4. Danh dau cac lan chop mat chu dong de hieu chinh nguong EOG.

## 4. Trong phien

- Ghi raw EEG 512 Hz, `poor_signal`, attention/meditation (chi de tham khao), timestamp don dieu va event.
- Moi 5 phut, tam dung ngan de nguoi tham gia tu danh gia KSS 1-9.
- Quay video mat chi khi co dong y rieng; video chi dung de gan nhan chop mat va duoc xoa theo ke hoach bao mat.
- Nguoi gan nhan danh dau thoi diem chop mat tren mot phan du lieu de tao ground truth.

## 5. Quy tac nhan nhi phan de xuat

- KSS 1-5: `alert = 0`.
- KSS 7-9: `drowsy = 1`.
- KSS 6: vung chuyen tiep, loai khoi baseline nhi phan nhung giu de phan tich sau.
- Gan nhan cua mot muc KSS chi ap dung cho khoang gan no; khong noi suy qua su kien nghi, thao tai nghe hoac mat tin hieu.

## 6. Kiem soat chat luong

- Loai cac cua so co `poor_signal > 50` trong qua 20% so mau.
- Luu nguyen ban raw bat bien; tien xu ly sang thu muc `processed` kem phien ban ma nguon.
- Khong tron cac cua so cua cung mot nguoi vao ca train va test.
- Scale/standardize chi fit tren train trong tung fold LOSO.

## 7. Danh gia

- Chop mat: ghep su kien du doan voi nhan tay trong sai so +/-200 ms; bao cao precision, recall, F1 va sai so dem.
- Phan loai: LOSO theo subject; bao cao macro-F1, balanced accuracy, ROC-AUC tung nguoi va trung binh.
- Do tre: tu mau cuoi cua cua so den luc bat dau am thanh; bao cao median va p95 tren it nhat 100 lan suy luan.
