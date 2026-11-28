"""Windows interface for the N16 live detector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from tkinter import filedialog, messagebox
from zipfile import ZipFile, ZIP_DEFLATED

from serial.tools import list_ports
from n16_eeg import __version__

from n16_eeg.csv_test import test_recording
from n16_eeg.live import LiveDetector, _beep, run_live
from n16_eeg.model import load_model
from n16_eeg.monitor_hud import MonitorHUD
from n16_eeg.session_review import REPORT, write_review
from n16_eeg.paired_replay import DETAILS, REPORT as COMPARE_REPORT, compare_recording


STATES = {
    "TINH": ("Chưa có cảnh báo", "Đang theo dõi tín hiệu EEG.", "#146747"),
    "NGUY_CO": ("Đang xác nhận", "Đang đối chiếu các cửa sổ EEG liên tiếp.", "#85520a"),
    "CANH_BAO": ("Cảnh báo buồn ngủ", "Hệ thống đã kích hoạt cảnh báo trong phiên đo.", "#B01828"),
    "TIN_HIEU_KEM": ("Tín hiệu yếu", "Chỉnh lại điện cực trán và kẹp tai.", "#8C5000"),
    "MAT_TIN_HIEU": ("Mất tín hiệu EEG", "Kiểm tra cổng kết nối và tai nghe.", "#B01828"),
}


def load_detector(directory: Path, variant: str = "LowRepeat") -> LiveDetector:
    manifest = json.loads((directory / "release.json").read_text(encoding="utf-8-sig"))
    profile = manifest["variants"][variant]
    model_path = directory / profile["file"]
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != profile["sha256"]:
        raise ValueError("Tệp mô hình không khớp bản N16 đã chốt.")
    model = load_model(model_path)
    if model.threshold != profile["threshold"]:
        raise ValueError("Ngưỡng mô hình không khớp cấu hình demo.")
    detector = LiveDetector(model, window_s=manifest["window_s"], step_s=manifest["step_s"],
                        consecutive=manifest["consecutive"],
                        release_consecutive=profile["release_consecutive"],
                        support_threshold=profile.get("support_threshold"),
                        support_consecutive=profile.get("support_consecutive", 1),
                        screen_raw=profile.get("screen_raw", False))
    detector.model_sha256 = profile['sha256']
    return detector


class LiveWindow:
    def __init__(self, root: tk.Tk, detector: LiveDetector, log_dir: Path, reference_directory=None) -> None:
        self.root = root
        self.detector = detector
        self.log_dir = log_dir
        self.reference_directory = reference_directory
        self.messages = Queue()
        self.stop_event = Event()
        self.running = False
        self.closing = False
        self.port = tk.StringVar()
        self.sound = tk.BooleanVar(value=True)
        self.sound_event = Event()
        self.risk = "—"
        self.subject = tk.StringVar()
        self.kss = tk.StringVar()
        self.kss_hint = tk.StringVar(value="KSS từ 1–9 theo mức buồn ngủ bạn tự chấm. Nhắc lại mỗi 5 phút.")
        self.observations = Queue()
        self.live_mode = False
        self.elapsed_s = self.last_kss_elapsed = 0.
        self.current_folder = log_dir
        self.last_error = None
        self.displayed_decision = None
        self.last_audio_request = None
        self.session_files = []

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.hud = MonitorHUD(self)
        self.refresh_ports()
        self.root.after(75, self.poll)

    def refresh_ports(self) -> None:
        try:
            ports = [f"{entry.device} — {getattr(entry, 'description', '')}" for entry in sorted(list_ports.comports())]
        except OSError:
            ports = []
        self.ports.configure(values=ports)
        if len(ports) == 1 and not self.port.get():
            self.port.set(ports[0])
        self.hud.port_hint.configure(text=f"Tìm thấy {len(ports)} cổng COM" if ports else "Chưa tìm thấy cổng COM")
        if self.session_files or hasattr(self, "csv_name"):
            return
        if not ports:
            self.hud.status("Chờ thiết bị", "Ghép đôi NeuroSky bằng Bluetooth rồi bấm Quét, hoặc mở tệp CSV đã đo.")
        else:
            self.hud.status("Sẵn sàng đo", "Chọn cổng NeuroSky rồi bấm Bắt đầu đo.")

    def start(self) -> None:
        port = self.port.get().strip().split(' ')[0].upper()
        if not re.fullmatch(r"COM[1-9][0-9]*", port):
            messagebox.showinfo("Chọn cổng", "Chọn hoặc nhập cổng NeuroSky, ví dụ COM4.", parent=self.root)
            return
        subject = self.subject.get().strip().upper() or 'UNASSIGNED'
        if not re.fullmatch(r'[A-Z0-9_-]+', subject):
            messagebox.showinfo('Mã người', 'Dùng chữ, số và dấu gạch dưới, ví dụ S03.', parent=self.root)
            return
        self.live_mode = True
        self.last_error = None
        self.displayed_decision = None
        self.last_audio_request = None
        self.observations = Queue()
        self.elapsed_s = self.last_kss_elapsed = 0.
        self.hud.reset_plot()
        self.risk = "—"
        self.hud.live_metrics()
        self.kss_hint.set("KSS từ 1–9 theo mức buồn ngủ bạn tự chấm.")
        session = f"{subject}_LIVE_{datetime.now():%Y%m%d_%H%M%S_%f}"
        self.current_folder = self.log_dir / session
        if self.kss.get():
            self.kss_hint.set(f"KSS {self.kss.get()} đang chờ mẫu EEG đầu tiên.")
            self.observations.put(dict(event='KSS', value=int(self.kss.get()),
                                       requested_timestamp_ms=time.time_ns() // 1_000_000))
            self.kss.set('')
        self.stop_event.clear()
        self.detector.reset()
        self.set_running(True)
        self.hud.status("Đang chờ EEG", f"{port}  ·  Đang chờ mẫu đầu tiên từ NeuroSky.")
        log = self.current_folder / 'predictions.csv'
        self.session_files = [log, log.parent / 'raw.csv', log.parent / 'raw.session.json']
        self.update_sound()
        Thread(target=self.run, args=(port, log, self.sound_event, subject, session), daemon=True).start()

    def run(self, port: str, log: Path, sound: Event, subject: str, session: str) -> None:
        review_error = None
        try:
            run_live(port, self.detector, log, sound=sound, stop=self.stop_event,
                     on_decision=lambda row: self.messages.put(("decision", row)),
                     raw_output=log.parent / 'raw.csv', subject_id=subject, session_id=session,
                     observations=self.observations,
                     on_observation=lambda row: self.messages.put(("observation",row)),
                     on_status=lambda row: self.messages.put(('telemetry', row)))
        except (OSError, RuntimeError, ValueError) as exc:
            self.messages.put(("error", str(exc)))
        finally:
            if log.with_name("raw.session.json").is_file():
                try:
                    write_review(log.parent)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    review_error = str(exc)
            self.messages.put(("done", review_error))

    def choose_csv(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("CSV EEG", "*.csv")])
        if not path:
            return
        source = Path(path)
        self.live_mode = False
        self.current_folder = self.log_dir
        log = self.log_dir / f"CSV_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        self.session_files = [log, source]
        self.stop_event.clear()
        self.set_running(True)
        self.csv_name = source.name
        self.hud.reset_plot(csv=True)
        self.hud.metrics(("Cửa sổ có điểm", "Đợt cảnh báo", "Cửa sổ bị loại"), ("—", "—", "—"))
        self.hud.status("Đang kiểm tra CSV", self.csv_name)
        Thread(target=self.run_csv, args=(source, log), daemon=True).start()

    def run_csv(self, source: Path, log: Path) -> None:
        try:
            result = test_recording(source, log, self.detector, self.stop_event, screen_raw=True)
            self.messages.put(("csv_done", (result, log)))
        except (OSError, ValueError) as exc:
            self.messages.put(("csv_error", str(exc)))

    def choose_comparison(self):
        path = filedialog.askopenfilename(parent=self.root,
            filetypes=[("ZIP phiên N16 hoặc CSV EEG", "*.zip *.csv")])
        if not path:
            return
        source = Path(path)
        target = self.log_dir / f"SoSanh_{datetime.now():%Y%m%d_%H%M%S_%f}"
        self.csv_name = source.name
        self.current_folder = self.log_dir
        self.live_mode = False
        self.session_files = []
        self.stop_event.clear()
        self.set_running(True)
        self.hud.reset_plot(csv=True)
        self.hud.metrics(("Cửa sổ hợp lệ", "Khác cảnh báo", "Cửa sổ bị loại"), ("—","—","—"))
        self.hud.status("Đang so sánh hai bản", source.name)
        Thread(target=self.run_comparison, args=(source,target), daemon=True).start()

    def run_comparison(self, source, target):
        try:
            models = (load_detector(self.reference_directory), self.detector)
            result = compare_recording(source,target,models,self.stop_event)
            self.messages.put(("compare_done",(result,target)))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, csv.Error) as exc:
            self.messages.put(("compare_error",str(exc)))

    def set_running(self, running: bool) -> None:
        self.running = running
        idle_state = "disabled" if running else "normal"
        for widget in (self.ports, self.start_button, self.refresh_button, self.csv_button, self.subject_entry):
            widget.configure(state=idle_state)
        if self.reference_directory is not None:
            self.compare_button.configure(state=idle_state)
        self.stop_button.configure(state="normal" if running else "disabled")
        self.export_button.configure(state='normal' if not running and any(path.exists() for path in self.session_files) else 'disabled')
        has_report = any(path.name in (REPORT,COMPARE_REPORT) for path in self.session_files)
        self.review_button.configure(state="normal" if not running and has_report else "disabled")
        for widget in (self.kss_button, self.feedback_button, self.drowsy_button):
            widget.configure(state='normal' if running and self.live_mode else 'disabled')

    def stop(self) -> None:
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.hud.status("Đang dừng…", "Đang kết thúc xử lý dữ liệu phiên.")

    def close(self) -> None:
        self.closing = True
        if self.running:
            self.stop()
        else:
            self.root.destroy()

    def poll(self) -> None:
        try:
            while True:
                kind, data = self.messages.get_nowait()
                if kind == "decision" and not self.stop_event.is_set():
                    self.displayed_decision = data
                    if data.get("audio_requested_ms") is not None:
                        self.last_audio_request = {key: data[key] for key in
                            ("episode_id", "audio_requested_ms", "raw_segment", "sample_index")}
                    self.hud.status(*STATES[data["state"]])
                    score = data.get("risk", "")
                    self.risk = f"{score:.2f}" if score not in ("", None) else "—"
                    self.hud.metric_values[0].configure(text=self.risk)
                elif kind == "error":
                    self.last_error = data
                elif kind == "observation":
                    if data["event"] == "KSS":
                        self.last_kss_elapsed = self.elapsed_s
                        self.kss_hint.set(f"Đã ghi KSS {data['value']} tại mẫu {data['next_sample_index']:,}.")
                    else:
                        self.kss_hint.set(f"Đã lưu phản hồi tại mẫu {data['next_sample_index']:,}.")
                elif kind == 'telemetry':
                    self.show_telemetry(data)
                elif kind in {"csv_done", "csv_error"}:
                    if self.closing:
                        self.root.destroy()
                        return
                    if kind == "csv_error" or data[0] is None:
                        self.session_files = []
                    self.set_running(False)
                    self.show_csv_result(kind, data)
                elif kind in ("compare_done","compare_error"):
                    if self.closing:
                        self.root.destroy()
                        return
                    if kind=="compare_error":
                        self.hud.status("Lỗi so sánh",self.csv_name,"#a92c39")
                        self.hud.plot_hint=str(data)
                    elif data[0] is None:
                        self.hud.status("Đã dừng so sánh",self.csv_name)
                        self.hud.plot_hint="Phiên phân tích đã dừng."
                    else:
                        counts,self.current_folder=data
                        self.session_files=[self.current_folder/COMPARE_REPORT,self.current_folder/DETAILS]
                        self.hud.metrics(("Cửa sổ hợp lệ","Khác cảnh báo","Cửa sổ bị loại"),
                                         (counts["valid"],counts["different"],counts["decisions"]-counts["valid"]))
                        self.hud.status("Đã so sánh hai bản",self.csv_name,"#146747")
                        self.hud.plot_hint="Bấm Xem lại phiên để xem số đợt cảnh báo và kết quả theo KSS."
                    self.hud.draw_trace()
                    self.set_running(False)
                elif kind == "done":
                    if self.closing:
                        self.root.destroy()
                        return
                    report = self.current_folder/REPORT
                    if report.is_file():
                        self.session_files.append(report)
                    self.set_running(False)
                    self.hud.status("Đã dừng đo", "Bấm Xem lại phiên để đối chiếu các phản hồi với EEG."
                                    if report.is_file() else "Các chỉ số hiển thị thuộc phiên vừa kết thúc.")
                    if data:
                        self.kss_hint.set(f"Chưa tạo được bản đối chiếu. {data}")
                    if self.last_error:
                        self.hud.status("Lỗi kết nối", "Kiểm tra cổng COM và đóng ứng dụng khác đang sử dụng cổng.", "#a92c39")
                        messagebox.showerror("Lỗi kết nối", f"Kiểm tra cổng COM và đóng OpenViBE nếu đang dùng cổng.\n\n{self.last_error}", parent=self.root)
        except Empty:
            self.root.after(75, self.poll)

    def show_csv_result(self, kind, data) -> None:
        if kind == "csv_error":
            self.hud.status("Lỗi đọc CSV", self.csv_name, "#a92c39")
            self.hud.plot_hint = str(data)
            self.hud.draw_trace()
            return
        result, log = data
        if result is None:
            self.hud.status("Đã dừng kiểm tra", self.csv_name)
            self.hud.plot_hint = "Phiên phân tích đã dừng."
        else:
            self.hud.status("Đã phân tích CSV", self.csv_name, "#146747")
            self.hud.metrics(("Cửa sổ có điểm", "Đợt cảnh báo", "Cửa sổ bị loại"),
                             [f"{result[key]:,}" for key in ("windows", "episodes", "poor_windows")])
            quality = ("Có dữ liệu chất lượng điện cực." if result["device_quality"]
                       else "CSV thiếu poor_signal. Chất lượng tiếp xúc chưa được xác minh.")
            self.hud.plot_hint = (
                f"{result['samples']:,} mẫu EEG  ·  {result['artifact_windows']:,} cửa sổ có mẫu bão hòa hoặc đứng giá trị."
                f"\n{quality}\n\nTệp dự đoán từng cửa sổ\n{log.name}")
        self.hud.draw_trace()

    def open_logs(self) -> None:
        try:
            self.current_folder.mkdir(parents=True, exist_ok=True)
            os.startfile(self.current_folder)
        except OSError as exc:
            messagebox.showerror("Không mở được log", str(exc), parent=self.root)

    def test_sound(self) -> None:
        Thread(target=_beep, daemon=True).start()

    def open_review(self):
        try:
            name = COMPARE_REPORT if self.current_folder/COMPARE_REPORT in self.session_files else REPORT
            os.startfile(self.current_folder/name)
        except OSError as exc:
            messagebox.showerror("Không mở được bản đối chiếu", str(exc), parent=self.root)

    def update_sound(self):
        if self.sound.get():
            self.sound_event.set()
        else:
            self.sound_event.clear()

    def export_session(self):
        files = [path for path in self.session_files if path.is_file()]
        if self.running or not files:
            return
        name = filedialog.asksaveasfilename(parent=self.root, defaultextension='.zip',
            initialfile=self.current_folder.name + '.zip', filetypes=[('ZIP phiên đo', '*.zip')])
        if not name:
            return
        destination = Path(name)
        if destination.suffix.lower() != '.zip' or any(destination.resolve() == path.resolve() for path in files):
            messagebox.showerror('Chọn tệp ZIP', 'Chọn một tên .zip riêng cho phiên đo.', parent=self.root)
            return
        try:
            with ZipFile(name, 'w', compression=ZIP_DEFLATED) as archive:
                for path in files:
                    archive.write(path, arcname=path.name)
            messagebox.showinfo('Đã xuất phiên', f'Tệp ZIP đã lưu tại\n{name}', parent=self.root)
        except OSError as exc:
            messagebox.showerror('Không xuất được phiên', str(exc), parent=self.root)

    def mark(self, event):
        if not self.running or not self.live_mode or self.stop_event.is_set():
            return
        value = None
        if event == 'KSS':
            if self.kss.get() not in [str(score) for score in range(1, 10)]:
                messagebox.showinfo('Ghi KSS', 'Chọn số KSS bạn vừa tự chấm.', parent=self.root)
                return
            value = int(self.kss.get())
            self.last_kss_elapsed = self.elapsed_s
            self.kss.set('')
            self.kss_hint.set(f"KSS {value} đang chờ mẫu EEG kế tiếp.")
        else:
            state = "đang tỉnh" if event == "SELF_REPORT_AWAKE" else "buồn ngủ"
            self.kss_hint.set(f"Phản hồi {state} đang chờ mẫu EEG kế tiếp.")
        observation = dict(event=event, value=value, requested_timestamp_ms=time.time_ns() // 1_000_000)
        if event in ('SELF_REPORT_AWAKE', 'SELF_REPORT_DROWSY'):
            observation.update(displayed_decision=self.displayed_decision,
                               last_displayed_audio_request=self.last_audio_request)
        self.observations.put(observation)

    def show_telemetry(self, data):
        self.elapsed_s = data['elapsed_s']
        seconds = int(self.elapsed_s)
        quality = data['poor_signal']
        contact = ('chưa xác minh' if quality == 255 else 'chưa tiếp xúc' if quality == 200
                   else 'kém' if quality > 50 else 'có nhiễu' if quality else 'tốt')
        self.hud.live_metrics(self.risk, f"{seconds // 60:02d}:{seconds % 60:02d}", contact.capitalize())
        self.hud.scope_unit.configure(text=f"EEG gốc  ·  {self.detector.fs} Hz  ·  count  ·  {data['samples']:,} mẫu")
        if self.elapsed_s - self.last_kss_elapsed >= 300:
            self.kss_hint.set('Đến mốc 5 phút. Tự chấm KSS, chọn số rồi bấm Ghi KSS.')
        self.hud.trace(data['trace'])


def main() -> None:
    here = Path(__file__).resolve().parent
    release = here / "n16_release"
    reference = here / "n16_reference"
    if not getattr(sys, "frozen", False):
        reference = here.parent / "output" / "N16_Research_20261128" / "welch_blend"
        release = here.parent / "output" / "N16_Reviewed_20261127" / "release"
        if not (release / "release.json").exists():
            release = here.parent / "output" / "N16_Chot_20261110"
        candidate = here.parent / "output" / "N16_Optimize4_20261128" / "profile"
        if (candidate/"release.json").is_file():
            release = candidate
        published = here.parent / "releases" / f"v{__version__}"
        if (published/"candidate"/"release.json").is_file():
            release = published/"candidate"
            reference = published/"reference"
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=f"N16 EEG Monitor {__version__}")
    parser.add_argument("--check", action="store_true", help="Test startup without EEG hardware")
    parser.add_argument("--csv", type=Path, help="Run the CSV test from the command line")
    parser.add_argument("--output", type=Path, help="Output CSV for --csv")
    parser.add_argument("--release-dir", type=Path, help="Directory containing release.json and the model")
    parser.add_argument("--review-session", type=Path, help="Create a feedback report for a saved live session folder")
    parser.add_argument("--compare", type=Path, help="Replay one session ZIP or CSV on both models; --output specifies a new folder")
    args = parser.parse_args()
    if args.review_session:
        write_review(args.review_session)
        return
    if args.release_dir is not None:
        release = args.release_dir
    if args.compare:
        if args.csv or not args.output:
            parser.error("--compare requires --output and cannot be combined with --csv")
        compare_recording(args.compare,args.output,(load_detector(reference),load_detector(release)),Event())
        return
    if bool(args.csv) != bool(args.output):
        parser.error("--csv and --output must be used together")
    if args.csv:
        test_recording(args.csv, args.output, load_detector(release), Event(), screen_raw=True)
        return
    root = tk.Tk()
    root.withdraw()
    if sys.platform == "win32":
        root.iconbitmap(str(here / "n16_monitor.ico"))
    try:
        detector = load_detector(release)
        log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "N16" / "logs"
        LiveWindow(root, detector, log_dir, reference if (reference/"release.json").is_file() else None)
        if args.check:
            for _ in range(detector.window_samples):
                result = detector.add_sample(0., 0.)
            if result is None or result["state"] not in STATES:
                raise RuntimeError("Kiểm tra suy luận thất bại.")
            root.update_idletasks()
            root.destroy()
            return
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        if not args.check:
            messagebox.showerror("Không mở được N16", str(exc), parent=root)
        root.destroy()
        raise SystemExit(str(exc)) from exc
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
