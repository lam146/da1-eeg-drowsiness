"""Replay two locked HUD profiles on one recording and compare their alerts."""
import csv
import json
from bisect import bisect_right
from collections import Counter
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from .csv_test import test_recording


REPORT = "So_sanh_hai_ban.html"
DETAILS = "quyet_dinh_doi_chieu.csv"


def read_marks(raw, metadata, stop):
    marks = []
    with raw.open(encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        try:
            dialect = csv.Sniffer().sniff(first, delimiters=",;\t")
        except csv.Error as exc:
            raise ValueError("Không xác định được định dạng CSV.") from exc
        handle.seek(0)
        reader = csv.DictReader(handle, dialect=dialect)
        fields = {"sample_index","raw_segment","event","kss","subject_id","session_id"}
        if not fields <= set(reader.fieldnames or ()):
            if metadata is not None:
                raise ValueError("raw.csv trong ZIP thiếu các cột nhật ký N16.")
            return marks
        count = 0
        for index,row in enumerate(reader):
            if stop.is_set():
                return []
            if int(row["sample_index"]) != index:
                raise ValueError("Chỉ số mẫu trong nhật ký bị gián đoạn.")
            if metadata is not None and (row["subject_id"],row["session_id"]) != (
                    metadata.get("subject_id"),metadata.get("session_id")):
                raise ValueError("Mã người hoặc mã phiên trong ZIP không khớp raw.csv.")
            events = row["event"].split("|")
            if "KSS" in events:
                value = row["kss"]
                label = None
                if events.count("KSS")==1 and value in {"1","2","3","4","5","7","8","9"}:
                    label = int(int(value)>=7)
                marks.append((index,int(row["raw_segment"]),label))
            count += 1
        if metadata is not None and count!=metadata.get("samples"):
            raise ValueError("Số mẫu của CSV khác với nhật ký trong ZIP.")
    return marks


def kss_intervals(marks, fs):
    # A minute of tolerance around the project's five-minute KSS cadence.
    return [(left,right,seg,y) for (left,seg,y),(right,seg2,y2) in zip(marks,marks[1:])
            if seg==seg2 and y is not None and y==y2 and 0<right-left<=360*fs]


def summarize(files, detail, intervals, detectors, stop):
    starts = [s[0] for s in intervals]
    totals = Counter()
    last_segment = None
    previous = [False,False]
    window = detectors[0].window_samples
    with files[0].open(encoding="utf-8-sig") as a, files[1].open(encoding="utf-8-sig") as b, detail.open(
            "x",encoding="utf-8-sig",newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["window_end_s","sample_index","raw_segment","kss_class","valid_both",
                         "reference_risk","candidate_risk","reference_alarm","candidate_alarm"])
        for old,new in zip(csv.DictReader(a),csv.DictReader(b),strict=True):
            if stop.is_set():
                return None
            if any(old[k]!=new[k] for k in ("sample_index","raw_segment","window_end_s")):
                raise ValueError("Hai cấu hình có lưới quyết định khác nhau.")
            end,segment = int(old["sample_index"]),int(old["raw_segment"])
            if segment!=last_segment:
                previous=[False,False]
            last_segment=segment
            label = None
            i=bisect_right(starts,end-window+1)-1
            if i>=0:
                _,right,seg,y=intervals[i]
                if seg==segment and end<right:
                    label=y
            valid = old["risk"]!="" and new["risk"]!=""
            active=[old["alarm"]=="1",new["alarm"]=="1"]
            writer.writerow([old["window_end_s"],end,segment,label,int(valid),old["risk"],new["risk"],
                             int(active[0]),int(active[1])])
            totals["decisions"]+=1
            if valid:
                totals["valid"]+=1
                totals["different"]+=active[0]!=active[1]
                totals[label]+=1
                for j,alarm in enumerate(active):
                    onset=alarm and not previous[j]
                    totals["episodes",j]+=onset
                    totals[label,"onsets",j]+=onset
                    totals[label,"active",j]+=alarm
            previous=active
    return totals


def make_report(path, name, counts, detectors, marks, intervals):
    def fraction(n,d):
        return f"{100*n/d:.1f}% ({n}/{d})" if d else "Chưa có khoảng đủ nhãn"
    def row(title, values):
        return "<tr><td>"+title+"</td>"+"".join(f"<td>{v}</td>" for v in values)+"</tr>"
    tables = [
        row("Số đợt cảnh báo bắt đầu trong cả bản ghi",[counts["episodes",j] for j in range(2)]),
        row("Số đợt bắt đầu trong khoảng KSS tỉnh",[counts[0,"onsets",j] if counts[0] else "—" for j in range(2)]),
        row("Cửa sổ đang cảnh báo khi KSS tỉnh",[fraction(counts[0,"active",j],counts[0]) for j in range(2)]),
        row("Cửa sổ chưa cảnh báo khi KSS buồn ngủ",[fraction(counts[1]-counts[1,"active",j],counts[1]) for j in range(2)])]
    page=f"""<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>So sánh hai bản cảnh báo N16</title><style>body{{font:16px/1.55 'Segoe UI',sans-serif;max-width:850px;margin:32px auto;padding:0 16px;color:#173647}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #cddcdd}}th{{background:#e9f3f2}}a{{color:#075f59}}</style>
<h1>Hai bản cảnh báo trên cùng một phiên EEG</h1><p>{escape(name)}</p>
<p>{counts['valid']} cửa sổ có điểm ở cả hai bản · {counts['decisions']-counts['valid']} cửa sổ bị loại khỏi so sánh · {counts['different']} cửa sổ cho trạng thái cảnh báo khác nhau.</p>
<table><tr><th>Chỉ số</th><th>Đối chứng · {detectors[0].threshold:g}</th><th>Bản đang thử · {detectors[1].threshold:g}</th></tr>{''.join(tables)}</table>
<p>Đọc được {len(marks)} mốc bấm KSS, trong đó có {len(intervals)} khoảng giữa hai mốc đủ điều kiện. App chấm cửa sổ nằm trọn giữa hai mốc liên tiếp cùng lớp (1–5 tỉnh, 7–9 buồn ngủ), cùng đoạn EEG và cách nhau tối đa 6 phút. Mức 6 và phản hồi bấm tay về cảnh báo được giữ ngoài phép chấm này.</p>
<p>Các tỷ lệ dùng trạng thái cảnh báo sau bước xác nhận trên những cửa sổ có điểm ở cả hai bản. Nhiều cửa sổ có thể thuộc cùng một đợt. Một phiên một lớp chỉ cho biết kết quả ở lớp đó. Việc phát lại CSV không đo thời điểm loa phát âm.</p>
<p><a href="{DETAILS}">Xem quyết định của hai mô hình tại từng mẫu</a></p></html>"""
    path.write_text(page,encoding="utf-8")


def compare_recording(source, destination, detectors, stop):
    """Publish a new result directory only when both replays and the report complete."""
    destination = Path(destination)
    source = Path(source)
    if destination.exists():
        raise FileExistsError(f"Thư mục kết quả đã tồn tại: {destination}")
    if (detectors[0].fs,detectors[0].window_samples,detectors[0].step_samples)!=(
            detectors[1].fs,detectors[1].window_samples,detectors[1].step_samples):
        raise ValueError("Hai mô hình cần cùng độ dài cửa sổ và bước cập nhật.")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        tmp=Path(tmp)
        raw=source
        meta=None
        if source.suffix.lower()==".zip":
            raw=tmp/"raw.csv"
            try:
                with ZipFile(source) as z:
                    if any(z.namelist().count(member)!=1 for member in ("raw.csv","raw.session.json")):
                        raise ValueError("Chọn ZIP của một phiên đo được xuất từ N16 EEG Monitor.")
                    meta=json.loads(z.read("raw.session.json").decode("utf-8-sig"))
                    if not isinstance(meta,dict):
                        raise ValueError("Nhật ký phiên trong ZIP không hợp lệ.")
                    with z.open("raw.csv") as src,raw.open("xb") as dst:
                        while chunk:=src.read(1024*1024):
                            if stop.is_set():
                                return None
                            dst.write(chunk)
            except BadZipFile as exc:
                raise ValueError("Không đọc được tệp ZIP của phiên đo.") from exc
        marks=read_marks(raw,meta,stop)
        intervals=kss_intervals(marks,detectors[0].fs)
        logs=[tmp/"reference.csv",tmp/"candidate.csv"]
        for log,detector in zip(logs,detectors):
            if stop.is_set() or test_recording(raw,log,detector,stop,screen_raw=True) is None:
                return None
        stage=tmp/"results"
        stage.mkdir()
        counts=summarize(logs,stage/DETAILS,intervals,detectors,stop)
        if counts is None or stop.is_set():
            return None
        make_report(stage/REPORT,source.name,counts,detectors,marks,intervals)
        stage.rename(destination)
    return counts
