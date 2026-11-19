"""Two fixed development hypotheses; existing held-out people are now development data.

No labels, rows, test thresholds or source recordings are changed to improve scores.
Requires independent future data for confirmation.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/two_subjects_20261104_v1'
FEATURES=['rel_delta','rel_theta','rel_alpha','rel_beta','rel_gamma','theta_alpha','theta_beta','alpha_beta',
          'spectral_entropy','blink_rate_per_min','blink_width_mean','long_blink_count']


def session_weights(frame):
    """Equal people, equal available classes per person, equal sessions within each class."""
    counts=frame.groupby(['subject_id','label','session_id']).size()
    classes=frame.groupby('subject_id').label.nunique()
    sessions=frame.groupby(['subject_id','label']).session_id.nunique()
    weights=np.array([1/(frame.subject_id.nunique()*classes.loc[s]*sessions.loc[(s,y)]*counts.loc[(s,y,t)])
                      for s,y,t in frame[['subject_id','label','session_id']].itertuples(index=False,name=None)])
    return weights*(len(frame)/weights.sum())


def estimator(weighted):
    return Pipeline([
        ('log1p',FunctionTransformer(np.log1p,feature_names_out='one-to-one')),
        ('scale',RobustScaler(quantile_range=(25,75))),
        ('classifier',LogisticRegression(C=.1,max_iter=3000,solver='lbfgs',
             class_weight=None if weighted else 'balanced',random_state=16))])


def run(target, source_folder=SOURCE):
    source=source_folder/'all_subjects_features_PROVISIONAL.csv'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    frame=pd.read_csv(source);X=frame[FEATURES].to_numpy(float);y=frame.label.to_numpy(int)
    assert np.isfinite(X).all() and np.all(X>=0)
    assert set(frame.subject_id)=={'S01','S02'}
    target.mkdir(parents=True,exist_ok=False)
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=str(source),source_sha256=source_hash,
              status='DEVELOPMENT_ONLY_NOT_INDEPENDENT_CONFIRMATION',source_labels='PROVISIONAL',
              prior_results_already_inspected=True,external_data_used=False,
              features=FEATURES,threshold=.5,regularization_C=.1,hyperparameter_search=False,
              variants=['log1p_robust','log1p_robust_sessionbalanced'],
              future_confirmation_required='New sessions with verified timing and both classes; new person for cross-person confirmation.')
    (target/'development_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    table=[];reports=[]
    for name,weighted in [('log1p_robust',False),('log1p_robust_sessionbalanced',True)]:
        out=target/name;out.mkdir();folds=[];oof=[]
        for person in sorted(frame.subject_id.unique()):
            tr=frame.subject_id.ne(person).to_numpy();te=~tr
            model=estimator(weighted)
            if weighted:
                weights=session_weights(frame.loc[tr])
                model.fit(X[tr],y[tr],classifier__sample_weight=weights)
            else:
                weights=None;model.fit(X[tr],y[tr])
            probability=model.predict_proba(X[te])[:,1];pred=(probability>=.5).astype(int)
            cm=confusion_matrix(y[te],pred,labels=[0,1])
            scaler=model.named_steps['scale']
            expected=np.median(np.log1p(X[tr]),axis=0)
            assert np.allclose(scaler.center_,expected)
            fold=dict(subject=person,train_subjects=sorted(frame.loc[tr].subject_id.unique()),
                      accuracy=float(accuracy_score(y[te],pred)),macro_f1=float(f1_score(y[te],pred,average='macro')),
                      balanced_accuracy=float(balanced_accuracy_score(y[te],pred)),roc_auc=float(roc_auc_score(y[te],probability)),
                      confusion_matrix=cm.tolist(),class_recalls=(np.diag(cm)/cm.sum(axis=1)).tolist(),
                      scaler_center_train_only=scaler.center_.tolist(),samples=int(te.sum()))
            folds.append(fold)
            table.append(dict(model=name,**{k:fold[k] for k in ['subject','accuracy','macro_f1','balanced_accuracy','roc_auc']}))
            for idx,prediction,p in zip(np.flatnonzero(te),pred,probability):
                oof.append(dict(row_index=int(idx),subject_id=person,true_label=int(y[idx]),prediction=int(prediction),probability=float(p)))
            joblib.dump(model,out/f'train_on_{fold["train_subjects"][0]}.joblib')
        report=dict(**plan,variant=name,weighted=weighted,folds=folds,
                    mean={key:float(np.mean([f[key] for f in folds])) for key in ['accuracy','macro_f1','balanced_accuracy','roc_auc']})
        (out/'metrics_DEVELOPMENT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        pd.DataFrame(oof).to_csv(out/'predictions_DEVELOPMENT.csv',index=False)
        reports.append(report);print(name,report['mean'])
    pd.DataFrame(table).to_csv(target/'comparison_DEVELOPMENT.csv',index=False)
    baseline=json.loads((source_folder/'logistic_combined/metrics_PROVISIONAL.json').read_text(encoding='utf-8'))
    session_rows=[]
    predictions=pd.read_csv(source_folder/'logistic_combined/oof_predictions_PROVISIONAL.csv').set_index('row_index')
    for (person,session),g in frame.groupby(['subject_id','session_id']):
        yp=predictions.loc[g.index,'prediction'].to_numpy()
        session_rows.append(dict(subject_id=person,session_id=session,rows=len(g),label=int(g.label.iloc[0]),
                                 baseline_accuracy=float(np.mean(yp==g.label.to_numpy())),
                                 zero_estimated_blink_fraction=float(g.blink_count.eq(0).mean())))
    pd.DataFrame(session_rows).to_csv(target/'session_diagnostics.csv',index=False)
    lines=['# Phát triển mô hình trên dữ liệu pilot','',
           '**Các con số dưới đây là kết quả phát triển trên dữ liệu đã xem trước, không phải kiểm chứng độc lập về mức tăng độ chính xác. Nhãn vẫn là nhãn thời gian tạm.**','',
           f'Giữ nguyên {len(frame)} cửa sổ của bộ dữ liệu đầu vào, nhãn, phân chia theo người và ngưỡng 0,5. Hai cấu hình được ghi trong development_plan.json.', '',
           '## Chẩn đoán dữ liệu','',
           'Xem session_diagnostics.csv để đối chiếu kết quả từng phiên và tỷ lệ cửa sổ có ước lượng blink bằng 0. Tỷ lệ này cần nguồn đối chiếu thủ công để diễn giải.', '',
           '## Hai phương án phần mềm','',
           'A: biến đổi log1p các đặc trưng không âm, RobustScaler theo median/IQR của train và logistic với C=0,1 cố định. Mục đích là giảm ảnh hưởng các giá trị lớn, không bảo đảm điểm sẽ tăng.',
           'B: cùng cách xử lý A, bổ sung trọng số chỉ tính trên train để mỗi người, mỗi lớp của người đó và mỗi phiên trong lớp có đóng góp cân bằng. Không nhân bản dữ liệu, không thay phân phối test.', '',
           '| Cấu hình | Accuracy trung bình hai người | Balanced accuracy | Macro F1 |','|---|---:|---:|---:|',
           f"| Logistic ban đầu | {baseline['summary_by_person']['accuracy']['mean']*100:.2f}% | {baseline['summary_by_person']['balanced_accuracy']['mean']*100:.2f}% | {baseline['summary_by_person']['macro_f1']['mean']*100:.2f}% |"]
    for r in reports:lines.append(f"| {r['variant']} | {r['mean']['accuracy']*100:.2f}% | {r['mean']['balanced_accuracy']*100:.2f}% | {r['mean']['macro_f1']*100:.2f}% |")
    lines+=['','Không lấy cấu hình có điểm cao nhất trong bảng này rồi gọi đó là kết quả test cuối. Đã dùng S01/S02 để phân tích và phát triển, nên cần dữ liệu mới để xác nhận.', '',
           '## Việc có tác động trực tiếp tiếp theo','',
           '1. Đối chiếu mã người và mã phiên trong phiếu với thư mục raw đã xác nhận.',
           '2. Chốt và cải thiện căn KSS–raw. Dữ liệu mới cần ghi mốc bắt đầu đến giây hoặc có sự kiện đồng bộ; đo thời gian thực và log mất kết nối nếu có. Không thể khôi phục chắc phần thời gian cũ đã quên.',
           '3. Khi thu bổ sung, ưu tiên S02 tỉnh táo ở các phiên/ngày khác nhau với cùng nhiệm vụ, thiết bị và tiếp xúc như phía buồn ngủ. MAIN_01 dù tên Awake vẫn có KSS 7,8,8 nên không đổi thành tỉnh để cân lớp.',
           '4. Thu một đoạn blink có nguồn đối chiếu độc lập và mốc đồng bộ. Có thể kiểm tra bỏ sót/báo nhầm, thay vì chọn detector theo accuracy phân loại KSS.',
           '5. Khóa quy trình và mô hình trước khi đánh giá dữ liệu mới. Dữ liệu S03 chưa dùng có thể phục vụ kiểm tra theo người sau khi đủ hồ sơ và đã khóa lựa chọn; chưa sử dụng S03 trong lần này.', '',
           'Tất cả đầu ra giữ cục bộ. Mô hình ngoại tuyến không tự đáp ứng cảnh báo <3 giây hoặc xuất sang runtime C++.', '',
           'Nguồn kỹ thuật: [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html), [tránh rò rỉ dữ liệu](https://scikit-learn.org/stable/common_pitfalls.html), [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).']
    (target/'PHAN_TICH_VA_CAI_THIEN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    return reports


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(ROOT/'output/model_development_20261104_v1')
