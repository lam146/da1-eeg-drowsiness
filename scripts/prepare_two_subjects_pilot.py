"""Exploratory two-person baseline with explicitly provisional nominal-time labels.

Raw-folder identity is user-confirmed. No names or class labels inferred from Awake/Tired.
No original file is edited. This is not verified KSS-to-raw alignment.
"""
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd

from filter_pilot_offline import run as filter_raw
from process_s01_pilots import spectral_features
from n16_eeg.blink import blink_window_features
from n16_eeg.sklearn_train import evaluate_frame

ROOT=Path(__file__).resolve().parents[1]
POLICY='PROVISIONAL_NOMINAL_TIME_SAME_CLASS_ENDPOINTS_v1'
LIMIT='Exploratory only. KSS times are rounded and raw alignment is unverified; not a final N16 accuracy claim.'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def note_data(path):
    with zipfile.ZipFile(path) as z:root=ET.fromstring(z.read('word/document.xml'))
    paragraphs=[''.join(t.text or '' for t in p.findall('.//w:t',NS)).strip() for p in root.findall('.//w:p',NS)]
    paragraphs=[p for p in paragraphs if p]
    points={}
    for marker,minute in [('Đầu phiên',0),('Phút 5',5),('Phút 10',10),('Phút 15',15)]:
        for i,p in enumerate(paragraphs):
            if (p==marker or re.fullmatch(re.escape(marker)+r' \(\d{1,2}:\d{2}\)',p)) and i+3<len(paragraphs):
                wall=paragraphs[i+1].strip('[]');score=paragraphs[i+2].strip('[]')
                if re.fullmatch(r'[1-9]',score) and re.fullmatch(r'\d{1,2}[:.]\d{2}',wall):
                    points[minute]=dict(kss=int(score),wall_clock_as_recorded=wall,reply_end=paragraphs[i+3])
                    break
    return paragraphs,points


def binary(kss):
    if 1<=kss<=5:return 0
    if 7<=kss<=9:return 1
    return None


def intervals(points,duration):
    """Nominal time is an explicit assumption; never stretch intervals to fit a file."""
    result=[]
    times=sorted(points)
    for left,right in zip(times,times[1:]):
        a,b=binary(points[left]['kss']),binary(points[right]['kss'])
        if right-left==5 and right*60<=duration+1e-6 and a is not None and a==b:
            result.append(dict(start_s=left*60,end_s=right*60,label=a,
                               left_kss=points[left]['kss'],right_kss=points[right]['kss'],
                               time_alignment_verified=False,label_policy=POLICY))
    return result


def run(target):
    target.mkdir(parents=True,exist_ok=False)
    note_paths=[p for p in (ROOT/'raw/note').rglob('*.docx') if not p.name.startswith('~$')]
    inputs=[];blink_files=[]
    for subject in ['S01','S02']:
        for path in sorted((ROOT/'raw/Raw data'/subject).glob('*.csv')):
            if path.name=='test1.csv':continue
            if re.search(r'_blink(?:_?\d+)?(?:_|$)',path.stem,re.IGNORECASE):
                blink_files.append(str(path))
                continue
            match=re.search(r'_(PILOT|MAIN)_(\d+)',path.name)
            if not match:raise ValueError(f'Unknown session {path}')
            kind,num=match.groups()
            if subject=='S01':note=ROOT/'raw/note/S01'/f'{num}.docx'
            else:
                choices=[p for p in note_paths if f'_MAIN_{num}_' in p.name]
                local_choices=[p for p in choices if p.parent.name==subject]
                choices=local_choices or choices
                if len(choices)!=1:raise ValueError(f'Need one note for {path.name}: {choices}')
                note=choices[0]
            inputs.append((subject,f'{kind}_{num}',path,note))
    summary=[];records=[];window_audit=[];observation_rows=[];note_snapshots=[]
    # Flag exact repeated native 1-second fragments across sessions, sampled every 32 raw points.
    seen_fragments={};overlap_counts={};overlap_examples=[]
    for subject,session,source,note in inputs:
        original=source.read_bytes();raw_hash=hashlib.sha256(original).hexdigest()
        paragraphs,points=note_data(note)
        folder=target/subject/session
        with redirect_stdout(io.StringIO()):report=filter_raw(source,folder)
        processed=pd.read_csv(folder/'eeg_filtered_offline.csv')
        raw=pd.read_csv(source,usecols=['Time:512Hz','Electrode'])
        x=raw.Electrode.to_numpy(float);signal=processed.eeg_filtered_count.to_numpy(float)
        quality=processed.quality_screen.to_numpy();duration=len(x)/512
        spans=intervals(points,duration)
        pd.DataFrame(spans,columns=['start_s','end_s','label','left_kss','right_kss','time_alignment_verified','label_policy']).to_csv(folder/'provisional_label_intervals.csv',index=False)
        note_snapshots.append(dict(subject_id=subject,session_id=session,path=str(note),
                                   sha256=hashlib.sha256(note.read_bytes()).hexdigest(),paragraphs=paragraphs))
        for minute,obs in points.items():
            observation_rows.append(dict(subject_id=subject,session_id=session,planned_minute=minute,**obs,
                                         raw_time_verified=False,raw_time_assumed_s=minute*60,
                                         within_nominal_raw=minute*60<=duration))
        counts=dict(total=0,excluded_signal=0,review_filter_edge=0,unlabeled=0,kept=0)
        source_features=[]
        for a in range(0,len(x)-2048+1,2048):
            b=a+2048;left,right=a/512,b/512;counts['total']+=1
            span=next((v for v in spans if v['start_s']<=left and right<=v['end_s']),None)
            if np.any(quality[a:b]=='exclude_screening'):status='excluded_signal'
            elif np.any(quality[a:b]=='review_filter_edge'):status='review_filter_edge'
            elif span is None:status='unlabeled'
            else:status='kept'
            counts[status]+=1
            window_audit.append(dict(subject_id=subject,session_id=session,start_s=left,end_s=right,status=status))
            if status!='kept':continue
            rec=dict(subject_id=subject,session_id=session,window_start_ms=left*1000,window_end_ms=right*1000,
                     label=span['label'],dataset_id='openvibe',signal_unit='count',fs_hz=512,
                     source_is_synthetic=False,time_alignment_verified=False,label_policy=POLICY,
                     processing_branch='offline_sos_v1',source_file=source.name)
            rec.update({k.removeprefix('offline_'):v for k,v in spectral_features(signal[a:b]).items()})
            rec.update(blink_window_features(x[a:b],512))
            source_features.append(rec);records.append(rec)
        pd.DataFrame(source_features).to_csv(folder/'features_PROVISIONAL.csv',index=False)
        # Fingerprints are diagnostics only; absence of a match is not proof of independence.
        for a in range(0,len(x)-512+1,32):
            sig=hashlib.blake2b(np.asarray(x[a:a+512],dtype='<f8').tobytes(),digest_size=16).digest()
            prior=seen_fragments.get(sig)
            here=(subject,session,a)
            if prior and prior[:2]!=here[:2]:
                pair='|'.join(sorted([prior[0]+'/'+prior[1],subject+'/'+session]))
                overlap_counts[pair]=overlap_counts.get(pair,0)+1
                if len(overlap_examples)<12:overlap_examples.append(dict(pair=pair,first_s=prior[2]/512,second_s=a/512))
            else:seen_fragments[sig]=here
        assert hashlib.sha256(source.read_bytes()).hexdigest()==raw_hash
        one=dict(subject_id=subject,session_id=session,source=str(source),source_sha256=raw_hash,note=str(note),
                 note_subject_folder=note.parent.name,identity_basis='Raw-folder grouping confirmed by user on 2026-11-04',
                 raw_duration_s=duration,rows=len(x),raw_min=float(x.min()),raw_max=float(x.max()),
                 kss=[points[t]['kss'] for t in sorted(points)],counts=counts,
                 provisional_class_counts=dict(pd.Series([r['label'] for r in source_features]).value_counts().items()),
                 processing_report=report)
        summary.append(one)
        print(subject,session,'KSS',one['kss'],'windows',counts,flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(target/'all_subjects_features_PROVISIONAL.csv',index=False)
    for subject,group in frame.groupby('subject_id'):group.to_csv(target/f'{subject}_features_PROVISIONAL.csv',index=False)
    pd.DataFrame(window_audit).to_csv(target/'window_audit.csv',index=False)
    pd.DataFrame(observation_rows).to_csv(target/'kss_observations.csv',index=False)
    (target/'notes_snapshot.json').write_text(json.dumps(note_snapshots,ensure_ascii=False,indent=2),encoding='utf-8')
    spectral=['rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma','theta_alpha','theta_beta','alpha_beta','spectral_entropy']
    blink=['blink_rate_per_min','blink_width_mean','long_blink_count']
    configs=[('logistic_combined','logistic',spectral+blink),('logistic_spectral','logistic',spectral),
             ('logistic_blink','logistic',blink),('svm_combined','svm',spectral+blink),
             ('random_forest_combined','random-forest',spectral+blink)]
    run_reports=[]
    for name,model,features in configs:
        trained,report,predictions=evaluate_frame(frame,features,'openvibe',model)
        report.update(label_policy=POLICY,primary_model_preselected=name=='logistic_combined',
                      time_alignment_verified=False,experimental_status='EXPLORATORY_PROVISIONAL_LABELS',
                      label_warning=LIMIT,model_selection='No model selected by test scores; logistic_combined predeclared primary.')
        for fold in report['folds']:
            test_person=fold['subject'];train_people=set(fold['train_subjects'])
            assert test_person not in train_people
        out=target/name;out.mkdir()
        joblib.dump(trained,out/'pipeline.joblib')
        predictions.to_csv(out/'oof_predictions_PROVISIONAL.csv',index=False)
        (out/'metrics_PROVISIONAL.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        run_reports.append(dict(name=name,summary=report['summary_by_person'],folds=report['folds']))
        print('EVALUATED',name,report['summary_by_person'],flush=True)
    # Balanced accuracy is also saved separately from simple overall accuracy because classes are unequal.
    overall=dict(created_utc=datetime.now(timezone.utc).isoformat(),subjects=['S01','S02'],session_count=len(summary),
                 included_subject_policy='User confirmed folder identity; S03 excluded. Note header conflicts retained in audit.',
                 confirmed_user_context='KSS numbers in tables are actual; S02 uses the same NeuroSky/OpenViBE driver as S01.',
                 label_policy=POLICY,label_warning=LIMIT,source_files_unchanged=True,
                 intervals='Consecutive nominal 5-min KSS points, same binary class; no KSS6; no interval ending beyond raw duration; no final point-only extension.',
                 counts_by_person_label=frame.groupby(['subject_id','label']).size().rename('windows').reset_index().to_dict('records'),
                 sessions=summary,runs=run_reports,shared_fragment_counts=overlap_counts,shared_fragment_examples=overlap_examples,
                 excluded_files=[str(p) for p in (ROOT/'raw/Raw data/S03').glob('*.csv')]+['test1.csv']+blink_files)
    # JSON integer conversion for pandas/numpy count values.
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(str(type(v)))
    (target/'experiment_PROVISIONAL.json').write_text(json.dumps(overall,ensure_ascii=False,indent=2,default=scalar,allow_nan=False),encoding='utf-8')
    lines=['# Kết quả ML thăm dò trên S01 và S02','',
           '**Đây là kết quả thăm dò với nhãn thời gian danh định, chưa phải độ chính xác cuối cùng của đồ án.**',
           'Mã người lấy theo thư mục raw như người dùng xác nhận; S03 và test1 được loại. KSS lấy từ số trong bảng, không lấy Awake/Tired từ tên file. Thiết bị/driver của S02 đã được người dùng xác nhận cùng loại với S01.', '',
           '## Quy tắc đã cố định trước khi xem kết quả','',
           'Chỉ dùng khoảng giữa hai mốc KSS cách nhau 5 phút, cả hai cùng phía 1–5 hoặc 7–9. Mốc có KSS 6 hoặc khoảng có đầu/cuối khác lớp bị bỏ. Chỉ giữ khoảng mà cả hai mốc danh định nằm trong thời lượng raw. Điểm cuối rời rạc không được kéo dài thành nhãn cho đuôi file.',
           'Giả định tạm: mốc đầu KSS tương ứng raw 0 và các mốc phút 5/10/15 tương ứng 300/600/900 giây. **Giả định này chưa được xác minh** vì nhật ký chỉ ghi đến phút và một số file lệch thời lượng. Không sửa giờ gốc hay khẳng định đã đồng bộ tới giây.',
           'Lọc ngoại tuyến và sàng lọc theo cùng bộ quy tắc đã dùng cho S01; cửa sổ 4 giây không chồng lấn, bỏ vùng lỗi và biên lọc. Đặc trưng blink là ước lượng chưa đối chiếu thủ công. Chuẩn hóa trong train từng lượt LOSO; không trộn người vào train/test. Logistic kết hợp được chọn trước làm mô hình chính; các mô hình khác là đối chiếu, không chọn ngưỡng theo test.', '',
           '## Dữ liệu sau chuẩn bị','', '| Người | Nhãn 0 theo quy tắc tạm | Nhãn 1 theo quy tắc tạm |','|---|---:|---:|']
    for person in ['S01','S02']:
        c=frame.loc[frame.subject_id==person,'label'].value_counts()
        lines.append(f'| {person} | {c.get(0,0)} | {c.get(1,0)} |')
    lines+=['','## Từng phiên','', '| Người / phiên | KSS ghi trong bảng | Raw giây | Cửa sổ dùng trong vòng thăm dò |','|---|---|---:|---:|']
    for r in summary:lines.append(f"| {r['subject_id']} / {r['session_id']} | {', '.join(map(str,r['kss']))} | {r['raw_duration_s']:.2f} | {r['counts']['kept']} |")
    lines+=['','## Kết quả ngoài tập học','', '| Cấu hình | Người test | Accuracy | Macro F1 | Balanced accuracy |','|---|---|---:|---:|---:|']
    for report in run_reports:
        for fold in report['folds']:
            lines.append(f"| {report['name']} | {fold['subject']} | {fold.get('accuracy',0):.3f} | {fold.get('macro_f1',0):.3f} | {fold.get('balanced_accuracy',0):.3f} |")
    lines+=['','## Giới hạn và bước tiếp theo','',
        'Có đúng hai người, nên chỉ có hai lượt LOSO và chưa thể khẳng định tổng quát hóa tốt cho nhiều người. S02 có rất ít phiên tỉnh so với buồn ngủ; cần xem recall từng lớp và nguồn tạo khác biệt, không chỉ accuracy. Nhiệm vụ S01/S02 khác nhau hoặc chưa được ghi đầy đủ có thể gây nhiễu kết luận.',
        'MAIN_01 của S02 có tên Awake nhưng KSS 7,8,8 nên dùng nhãn buồn ngủ. Khi có nhiều phiếu trùng mã phiên, ưu tiên phiếu trong thư mục cùng người với CSV. Mã trong nội dung phiếu được giữ nguyên trong notes_snapshot.json. MAIN_02 thuộc raw S03 nên nằm ngoài lần chạy.',
        'Một số phiếu còn văn bản cũ báo chưa có KSS hoặc số mẫu của phiên khác. Đợt này dùng bảng KSS đã được xác nhận và tính lại số liệu raw, không sửa file Word gốc.',
        'Ưu tiên xác minh căn thời gian và chốt nhãn trước khi báo kết quả cuối. Kết quả không chứng minh blink ≥90%, độ trễ <3 giây hoặc mô hình C++ tương đương. Model joblib chỉ dùng ngoại tuyến; không phải JSON runtime C++.',
        f'Các fingerprint giống nhau giữa phiên: {overlap_counts or "chưa phát hiện ở lưới kiểm tra 1 giây/bước 32 mẫu"}. Đây không phải bằng chứng chắc chắn các phiên không chồng lấn theo thời gian.',
        'Dữ liệu, dự đoán, mô hình và báo cáo được lưu cục bộ dưới output; không đăng lên GitHub. CSV có chữ PROVISIONAL phải được giữ kèm quy tắc và giới hạn của lần chạy.']
    (target/'KET_QUA_ML_THAM_DO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return overall


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(ROOT/'output/two_subjects_20261104_v1')
