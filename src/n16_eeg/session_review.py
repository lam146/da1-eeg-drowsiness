"""Summarize a recorded HUD session using the decision the wearer actually saw."""

import csv
import json
import os
from collections import Counter, deque
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from math import isfinite


REPORT = "Doi_chieu_phien.html"
SELF_REPORTS = {"SELF_REPORT_AWAKE": "Đang tỉnh", "SELF_REPORT_DROWSY": "Buồn ngủ"}
MAX_DISPLAY_AGE_MS = 2000


def clock(seconds):
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes:02d}:{rest:02d}"


def read_review(folder):
    folder = Path(folder)
    metadata = json.loads((folder/"raw.session.json").read_text(encoding="utf-8-sig"))
    fs = float(metadata["configuration"]["fs_hz"])
    if fs <= 0:
        raise ValueError("Tần số lấy mẫu không hợp lệ.")
    markers = metadata["observations"]
    decision_rows = {}
    tones = set()
    episodes = set()
    scored = rejected = lost = 0
    with (folder/"predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"state","alarm","risk","timestamp_ms"} <= set(reader.fieldnames or ()):
            raise ValueError("Thiếu các cột dự đoán của phiên đo trực tiếp.")
        for row in reader:
            if row["state"] == "MAT_TIN_HIEU":
                lost += 1
                continue
            if row["risk"] == "":
                rejected += 1
            else:
                scored += 1
            if not row.get("sample_index") or row.get("raw_segment") in ("", None):
                continue
            key = int(row["raw_segment"]), int(row["sample_index"])
            decision_rows[key] = row
            if row["alarm"] == "1" and row.get("episode_id"):
                episodes.add((key[0],int(row["episode_id"])))
            if row.get("audio_requested_ms"):
                tones.add((*key,int(row["audio_requested_ms"]),int(row["episode_id"])))

    counts = Counter()
    observations = []
    snippets = set()
    for marker in markers:
        event = marker.get("event")
        if event != "KSS" and event not in SELF_REPORTS:
            continue
        sample = marker.get("next_sample_index")
        segment = marker.get("raw_segment")
        observed = SELF_REPORTS.get(event, f"KSS {marker.get('value')}")
        shown = marker.get("displayed_decision") or {}
        row = decision_rows.get((segment, shown.get("sample_index")))
        pressed = marker.get("requested_timestamp_ms")
        linked = (sample is not None and row is not None and shown.get("raw_segment")==segment
                  and shown.get("timestamp_ms")==int(row["timestamp_ms"])
                  and shown.get("state")==row["state"] and shown.get("sample_index")<=sample
                  and isinstance(pressed,int) and 0<=pressed-int(row["timestamp_ms"])<=MAX_DISPLAY_AGE_MS)
        finding = "Mốc KSS" if event=="KSS" else "Chưa đủ dữ liệu để đối chiếu"
        state, score, episode, audio = "—","—","—","—"
        if event in SELF_REPORTS and linked:
            state = row["state"]
            score = row["risk"] or "—"
            episode = row.get("episode_id") or "—"
            if not row["risk"]:
                finding = "Tín hiệu chưa đủ chất lượng"
            elif event=="SELF_REPORT_AWAKE" and row["alarm"]=="1":
                finding = "Tự báo tỉnh khi đang cảnh báo"
            elif event=="SELF_REPORT_DROWSY" and row["alarm"]=="0":
                finding = "Tự báo buồn ngủ khi chưa cảnh báo"
            else:
                finding = "Phản hồi phù hợp với trạng thái cảnh báo"
        if event in SELF_REPORTS:
            counts[finding] += 1
            if linked and finding in ("Tự báo tỉnh khi đang cảnh báo","Tự báo buồn ngủ khi chưa cảnh báo"):
                snippets.add((segment,shown["sample_index"]))
            tone = marker.get("last_displayed_audio_request") or {}
            stamp = tone.get("audio_requested_ms")
            if (sample is not None and segment==tone.get("raw_segment")
                    and (segment,tone.get("sample_index"),stamp,tone.get("episode_id")) in tones
                    and isinstance(pressed,int) and pressed>=stamp):
                audio = f"#{tone['episode_id']} · {(pressed-stamp)/1000:.1f} s trước"
        observations.append([clock(sample/fs) if sample is not None else "Chưa gắn mẫu",
                             sample if sample is not None else "—",shown["sample_index"] if linked else "—",
                             observed,state,score,episode,audio,finding])
    return dict(subject=metadata["subject_id"],session=metadata["session_id"],duration=metadata["samples"]/fs,
                scored=scored,rejected=rejected,lost=lost,episodes=len(episodes),tones=len(tones),
                observations=observations, counts=counts,snippets=snippets,fs=fs,
                window=int(metadata["configuration"].get("window_samples",round(3*fs))))


def review_snippets(folder, report):
    """Read only far enough to recover the exact windows behind the disputed decisions."""
    wanted = set(report["snippets"])
    raw = folder/"raw.csv"
    if not wanted:
        return ""
    cards = []
    if raw.is_file():
        recent = deque(maxlen=report["window"])
        previous = None
        with raw.open(encoding="utf-8-sig",newline="") as handle:
            reader = csv.DictReader(handle)
            fields = {"raw_segment","sample_index","raw_eeg","poor_signal"}
            if fields <= set(reader.fieldnames or ()):
                for row in reader:
                    segment,index = int(row["raw_segment"]),int(row["sample_index"])
                    if previous!=(segment,index-1):
                        recent.clear()
                    previous = segment,index
                    recent.append((float(row["raw_eeg"]),int(row["poor_signal"])))
                    if previous not in wanted:
                        continue
                    if len(recent)==recent.maxlen and all(isfinite(v) for v,_ in recent):
                        values = [v for v,_ in recent]
                        low,high = min(values),max(values)
                        points = []
                        bucket = max(1,len(values)//140)
                        for first in range(0,len(values),bucket):
                            chunk = values[first:first+bucket]
                            for pos in sorted({chunk.index(min(chunk)),chunk.index(max(chunk))}):
                                x = (pos+first)*460/max(1,len(values)-1)
                                y = 25 if high==low else 47-(chunk[pos]-low)*44/(high-low)
                                points.append(f"{x:.1f},{y:.1f}")
                        secs=index/report["fs"]
                        unknown=sum(q==255 for _,q in recent)
                        poor=sum(50<q<255 for _,q in recent)
                        cards.append(f'<div><b>Quyết định tại {clock(secs)} · mẫu {index}</b> · '
                            f'EEG {low:g} đến {high:g} count · {poor}/{len(recent)} mẫu tiếp xúc kém, '
                            f'{unknown} mẫu chưa xác minh.<br><svg viewBox="0 0 460 50" width="460" '
                            f'role="img" aria-label="EEG trước quyết định" style="max-width:100%;background:#eef6f6">'
                            f'<polyline fill="none" stroke="#126f68" points="{" ".join(points)}"/></svg></div>')
                        wanted.remove(previous)
                    if not wanted:
                        break
    if wanted:
        cards.append(f"<p>Có {len(wanted)} cửa sổ chưa truy xuất được từ raw.csv.</p>")
    return "<h2>EEG tại các quyết định cần xem lại</h2>"+"".join(cards)


def write_review(folder):
    folder = Path(folder)
    report = read_review(folder)
    rows = "".join("<tr>"+"".join(f"<td>{escape(str(cell))}</td>" for cell in row)+"</tr>"
                   for row in report["observations"])
    counts = report["counts"]
    html = f"""<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đối chiếu phiên EEG</title><style>body{{font:16px/1.5 'Segoe UI',sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#193442}}h1,h2{{line-height:1.2}}header{{border-bottom:3px solid #167269}}section{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid #d3dfe2;padding:7px}}th{{background:#edf4f3}}</style>
<header><h1>Đối chiếu phiên EEG</h1><p>{escape(str(report['subject']))} · {escape(str(report['session']))} · {clock(report['duration'])} dữ liệu EEG</p></header>
<p><b>{report['episodes']} đợt cảnh báo · {report['tones']} lần yêu cầu phát âm</b></p>
<p>{report['scored']} cửa sổ có điểm, {report['rejected']} cửa sổ bị loại và {report['lost']} lần mất luồng EEG.</p>
<h2>Phản hồi cần xem lại</h2>
<p>{counts['Tự báo tỉnh khi đang cảnh báo']} lượt tự báo tỉnh khi app đang cảnh báo.
{counts['Tự báo buồn ngủ khi chưa cảnh báo']} lượt tự báo buồn ngủ khi app chưa cảnh báo.</p>
<p>Mỗi lượt là một lần bấm. Cột đợt giúp nhận ra nhiều lần bấm trong cùng một cảnh báo. Mốc mẫu cho phép tìm lại đoạn tương ứng trong raw.csv.
Bảng chỉ ghép trạng thái HUD khi quyết định khớp nhật ký cùng đoạn EEG và còn trong vòng 2 giây khi bấm. Cột âm thanh ghi thời gian từ yêu cầu phát gần nhất mà HUD đã nhận.</p>
<section><table><thead><tr><th>Phút:giây EEG</th><th>Mẫu khi bấm</th><th>Mẫu quyết định</th><th>Người đo</th><th>Trạng thái HUD</th><th>Điểm</th><th>Đợt</th><th>Yêu cầu âm gần nhất</th><th>Đối chiếu</th></tr></thead><tbody>
{rows or '<tr><td colspan="9">Phiên này chưa có KSS hoặc phản hồi đã ghi.</td></tr>'}</tbody></table></section>
{review_snippets(folder, report)}
<p>Phản hồi tự đánh giá dùng để rà soát từng thời điểm. Tỷ lệ nhận đúng của cả phiên cần nhãn độc lập cho các khoảng được chấm.</p></html>"""
    target = folder/REPORT
    with NamedTemporaryFile("w", encoding="utf-8", dir=folder, delete=False) as handle:
        temp = Path(handle.name)
        handle.write(html)
    try:
        os.replace(temp,target)
    finally:
        temp.unlink(missing_ok=True)
    return target
