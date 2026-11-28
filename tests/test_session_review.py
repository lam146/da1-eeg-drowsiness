import csv
import json

from n16_eeg.session_review import read_review, write_review


def test_wearer_feedback_requires_matching_fresh_decision_and_attached_sample(tmp_path):
    active = dict(timestamp_ms=1000, sample_index=1535, raw_segment=0, state="CANH_BAO",
                  alarm=1, risk=.78, episode_id=1, audio_requested_ms=1010)
    quiet = dict(timestamp_ms=1500, sample_index=1791, raw_segment=0, state="TINH",
                 alarm=0, risk=.35, episode_id=None, audio_requested_ms=None)
    poor = dict(quiet, state="TIN_HIEU_KEM", timestamp_ms=1800, sample_index=1919, risk="")
    with (tmp_path/"predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=active)
        writer.writeheader()
        writer.writerows([active,quiet,poor])
    def feedback(event, shown, when, sample, segment=0):
        return dict(event=event,displayed_decision=shown,requested_timestamp_ms=when,next_sample_index=sample,
                    raw_segment=segment,last_displayed_audio_request=active)

    markers = [dict(event="KSS",value=3,next_sample_index=0,raw_segment=0),
               feedback("SELF_REPORT_AWAKE",active,1100,1600),
               feedback("SELF_REPORT_DROWSY",quiet,1600,1820),
               feedback("SELF_REPORT_DROWSY",poor,1850,1950),
               feedback("SELF_REPORT_AWAKE",active,8000,5000),
               feedback("SELF_REPORT_AWAKE",active,1250,1800,segment=1),
               feedback("SELF_REPORT_AWAKE",active,1300,None)]
    (tmp_path/"raw.session.json").write_text(json.dumps(dict(subject_id="<S>",session_id="TEST",
        samples=5200,configuration=dict(fs_hz=512),observations=markers)))
    review=read_review(tmp_path)
    assert review["episodes"] == review["tones"] == 1
    assert review["counts"]["Tự báo tỉnh khi đang cảnh báo"] == 1
    assert review["counts"]["Tự báo buồn ngủ khi chưa cảnh báo"] == 1
    assert review["counts"]["Tín hiệu chưa đủ chất lượng"] == 1
    assert review["counts"]["Chưa đủ dữ liệu để đối chiếu"] == 3
    assert review["observations"][-1][0] == "Chưa gắn mẫu"
    assert review["observations"][1][-2].startswith("#1")
    assert review["observations"][-2][-2] == "—"
    with (tmp_path/"raw.csv").open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["raw_segment","sample_index","raw_eeg","poor_signal"])
        writer.writerows((0,i,i%17,0) for i in range(1800))
    report=write_review(tmp_path).read_text(encoding="utf-8")
    assert "&lt;S&gt;" in report and report.count("<polyline") == 2
    assert "mẫu 1535" in report and "mẫu 1791" in report
