"""Batch offline screening; feature candidates stay unlabeled until KSS alignment is verified."""
from contextlib import redirect_stdout
from pathlib import Path
import hashlib
import io
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from filter_pilot_offline import run
from n16_eeg.signal_features import _periodogram, BANDS
from n16_eeg.blink import blink_window_features

ROOT=Path(__file__).resolve().parents[1]
FILES=[
    ('01','S01_25102026_PILOT_01.csv','moc1,2.docx'),
    ('02','S01_25102026_PILOT_02.csv','02.docx'),
    ('03','S01_26102026_PILOT_03.csv','03.docx'),
    ('04','S01_27102026_PILOT_04.csv','04.docx'),
    ('05','S01_27102026_PILOT_05.csv','05.docx'),
    ('06','S01_28102026_PILOT_06.csv','06.docx'),
]


def spectral_features(z):
    f,p=_periodogram(z,512)
    total_mask=(f>=.5)&(f<=45)
    total=max(float(np.trapezoid(p[total_mask],f[total_mask])),np.finfo(float).eps)
    out={};powers={}
    for name,(lo,hi) in BANDS.items():
        m=(f>=lo)&(f<hi)
        v=float(np.trapezoid(p[m],f[m]));powers[name]=v
        out[f'offline_abs_{name}']=v;out[f'offline_rel_{name}']=v/total
    for a,b in [('theta','alpha'),('theta','beta'),('alpha','beta')]:
        out[f'offline_{a}_{b}']=powers[a]/max(powers[b],np.finfo(float).eps)
    q=p[total_mask]/max(float(p[total_mask].sum()),np.finfo(float).eps)
    out['offline_spectral_entropy']=float(-np.sum(q*np.log2(q+np.finfo(float).eps))/np.log2(len(q)))
    out['offline_rms']=float(np.sqrt(np.mean(z*z)))
    return out


def read_note(path):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(path) as z:
        root=ET.fromstring(z.read('word/document.xml'))
    paragraphs=[''.join(t.text or '' for t in p.findall('.//w:t',ns)).strip() for p in root.findall('.//w:p',ns)]
    return [p for p in paragraphs if p]


def process(output):
    output.mkdir(parents=True,exist_ok=False)
    summaries=[];features=[];window_rows=[];kss_rows=[];note_inventory=[]
    for sid,filename,note in FILES:
        source=ROOT/'raw'/filename
        target=output/f'S01_PILOT_{sid}'
        with redirect_stdout(io.StringIO()):
            r=run(source,target)
        d=pd.read_csv(target/'eeg_filtered_offline.csv')
        raw=pd.read_csv(source,usecols=['Electrode']).Electrode.to_numpy(float)
        filtered=d.eeg_filtered_count.to_numpy(float)
        flags=d.quality_screen.to_numpy()
        counts=dict(total=0,excluded=0,filter_edge_review=0,candidate=0)
        for a in range(0,len(d)-2048+1,2048):
            b=a+2048;counts['total']+=1
            st='candidate'
            if np.any(flags[a:b]=='exclude_screening'):st='excluded'
            elif np.any(flags[a:b]=='review_filter_edge'):st='filter_edge_review'
            counts[st]+=1
            window_rows.append(dict(subject_id='S01',session_id=f'PILOT_{sid}',start_s=a/512,end_s=b/512,
                                    quality_screen=st,label_status='unassigned_alignment_pending'))
            if st!='candidate':continue
            rec=dict(subject_id='S01',session_id=f'PILOT_{sid}',window_start_ms=a/512*1000,
                     window_end_ms=b/512*1000,signal_unit='count',fs_hz=512,
                     dataset_id='openvibe_offline_diagnostic',quality_screen='not_flagged_by_screening',
                     label_status='unassigned_alignment_pending')
            rec.update(spectral_features(filtered[a:b]))
            rec.update({f'estimated_{k}':v for k,v in blink_window_features(raw[a:b],512).items()})
            features.append(rec)
        paragraphs=read_note(ROOT/'raw/note'/note)
        note_inventory.append(dict(session_id=f'PILOT_{sid}',source=str(ROOT/'raw/note'/note),
                                   sha256=hashlib.sha256((ROOT/'raw/note'/note).read_bytes()).hexdigest(),paragraphs=paragraphs))
        for marker,minute in [('Đầu phiên',0),('Phút 5',5),('Phút 10',10),('Phút 15',15)]:
            # Record only actually filled entries, never planned defaults or missing points.
            for i,p in enumerate(paragraphs):
                if p==marker and i+3<len(paragraphs):
                    wall=paragraphs[i+1].strip('[]');score=paragraphs[i+2].strip('[]')
                    if re.fullmatch(r'[1-9]',score) and re.fullmatch(r'\d{1,2}[:.]\d{2}',wall):
                        kss_rows.append(dict(subject_id='S01',session_id=f'PILOT_{sid}',planned_minute=minute,
                                             wall_clock_as_recorded=wall,kss=int(score),raw_time_s='',
                                             alignment_status='not_verified',source_note=note))
                        break
        candidates=counts['candidate']*4
        r.update(session_id=f'PILOT_{sid}',window_counts=counts,candidate_nonoverlap_s=candidates)
        summaries.append(r)
        print(sid,r['duration_s'],counts,flush=True)
    pd.DataFrame(features).to_csv(output/'feature_candidates_UNLABELED.csv',index=False)
    pd.DataFrame(window_rows).to_csv(output/'window_screening.csv',index=False)
    pd.DataFrame(kss_rows).to_csv(output/'kss_observations_UNALIGNED.csv',index=False)
    (output/'notes_snapshot.json').write_text(json.dumps(note_inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'batch_summary.json').write_text(json.dumps(dict(sessions=summaries,labels_assigned=False,
        model_trained=False,omitted_files=['S01_26102026_PILOT_04.csv','test1.csv'],
        feature_method='Hann periodogram on offline SOS-filtered contiguous segments; 4s nonoverlapping windows',
        blink_method='Existing baseline on original raw within candidate windows, estimates only; no manual reference',
        warning='Candidate time is not confirmed clean or labeled training time. No KSS interpolation or imputation.'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Kết quả xử lý sáu phiên S01', '',
        'Đã xử lý riêng cả sáu phiên, giữ nguyên CSV và Word gốc. File 04 ngày 26/10 và test1 không nằm trong bộ này.', '',
        'Lọc ngoại tuyến: notch 50 Hz Q=30, Butterworth bậc thiết kế 4 thông dải 0,5–45 Hz, lọc hai chiều zero-phase trên từng đoạn liên tục. Đây không phải bộ lọc thời gian thực trong ứng dụng.', '',
        'Đánh dấu cửa sổ 4 giây chứa raw chạm/vượt khoảng ASIC dự kiến −2048..2047 hoặc chuỗi giữ nguyên ít nhất 0,25 giây. Không nội suy lấp vùng lỗi; 4 giây ở mỗi đầu đoạn lọc được giữ riêng để xem ảnh hưởng biên. Đây là quy tắc sàng lọc kỹ thuật, không phải ngưỡng đạt của GVHD.', '',
        '| Phiên | Raw (giây) | Loại theo sàng lọc (giây) | Cửa sổ ứng viên 4 giây | Thời lượng ứng viên (giây) |',
        '|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['session_id']} | {r['duration_s']:.4f} | {r['screening_excluded_s']:.4f} | {r['window_counts']['candidate']} | {r['candidate_nonoverlap_s']:.0f} |")
    total=sum(r['candidate_nonoverlap_s'] for r in summaries)
    lines+=['',f'Tổng phần ứng viên: {total:.0f} giây, khoảng {total/60:.2f} phút. Đây là phần không bị quy tắc sàng lọc gắn cờ và tránh biên lọc, chưa phải thời lượng sạch đã có nhãn ML.', '',
        '| Phiên | Tỷ lệ năng lượng 49–51 / 0,5–100 Hz trước lọc | Sau lọc |',
        '|---|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['session_id']} | {r['median_line_fraction_before']*100:.3f}% | {r['median_line_fraction_after']*100:.3f}% |")
    lines+=['', 'Chỉ số phổ là trung vị trên cửa sổ 4 giây, bỏ vùng loại; không phải % dữ liệu hỏng. Giảm thành phần 50 Hz không chứng minh mọi nhiễu đã hết.', '',
        '## Các tệp bạn cần biết','',
        '- Mỗi thư mục phiên có eeg_filtered_offline.csv, screening_exclusions.csv và processing.json. Tín hiệu sau lọc chỉ phục vụ nhánh phổ; giữ raw gốc cho nhánh blink.',
        '- feature_candidates_UNLABELED.csv: đặc trưng phổ và blink ước lượng của phần ứng viên. Cố ý không có cột label; chưa đưa thẳng vào Huấn luyện + LOSO.',
        '- window_screening.csv: cửa sổ bị loại, cần xem biên lọc hoặc là ứng viên.',
        '- kss_observations_UNALIGNED.csv: điểm KSS thật lấy từ phiếu; raw_time_s để trống vì chưa xác minh căn thời gian.', '',
        '## Nhãn và giới hạn còn lại','',
        'Các phiên 01–03 có KSS phía tỉnh táo; 04–06 có KSS phía buồn ngủ. Đây là mô tả các điểm tự đánh giá, không dùng nó để gán toàn bộ cửa sổ của phiên. Hoạt động ở một số phiên còn trống; phần sự kiện trống hiểu là không ghi nhận bất thường theo xác nhận người dùng.', '',
        'Phiên 01 thiếu KSS cuối. Tên CSV phiên 02 chứa ngày cũ; ngày đo lấy theo phiếu. Phiên 06 lưu 780,0625 giây trong khi người đo xác nhận 15 phút: không tự bù 2 phút hoặc ép mốc KSS phút 15 vào file. Không kết luận do nhiễu hay mất Bluetooth chỉ từ CSV.', '',
        'Bộ lọc này khác nhánh FFT trong app hiện tại; không trộn các đặc trưng hai cách lọc trong một thí nghiệm. Các ước lượng blink chưa được đối chiếu nguồn độc lập, không dùng báo cáo đạt 90%. Chưa huấn luyện, chưa có accuracy/F1 và chưa kết luận đủ dữ liệu nghiệm thu.', '',
        'Bước tiếp theo: đối chiếu căn KSS với raw và khóa quy tắc nhãn, kiểm tra thêm các đoạn nghi nhiễu còn lại rồi mới tạo dữ liệu có nhãn. Đủ nhãn S01 vẫn chưa đủ kiểm định ra người mới; dữ liệu S02/LOSO là bước riêng. Bạn chưa cần đo lại toàn bộ sáu phiên chỉ vì các vùng lỗi đã phát hiện.']
    (output/'KET_QUA_XU_LY_S01.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summaries


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=ROOT/'output/processed_pilot/S01_batch_20261028_v1'
    process(p)
