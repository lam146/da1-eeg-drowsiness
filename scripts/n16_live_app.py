"""Windows interface for the N16 live detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from tkinter import filedialog, messagebox, ttk

from serial.tools import list_ports

from n16_eeg.csv_test import test_recording
from n16_eeg.live import LiveDetector, _beep, run_live
from n16_eeg.model import load_model


STATES = {
    "TINH": ("Tỉnh táo", "#146747"),
    "NGUY_CO": ("Có nguy cơ buồn ngủ", "#8C5000"),
    "CANH_BAO": ("Cảnh báo buồn ngủ", "#B01828"),
    "TIN_HIEU_KEM": ("Tín hiệu yếu. Chỉnh lại tai nghe.", "#8C5000"),
    "MAT_TIN_HIEU": ("Mất tín hiệu EEG. Kiểm tra cổng và tai nghe.", "#B01828"),
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
    return LiveDetector(model, window_s=manifest["window_s"], step_s=manifest["step_s"],
                        consecutive=manifest["consecutive"],
                        release_consecutive=profile["release_consecutive"],
                        support_threshold=profile.get("support_threshold"),
                        support_consecutive=profile.get("support_consecutive", 1))


class LiveWindow:
    def __init__(self, root: tk.Tk, detector: LiveDetector, log_dir: Path) -> None:
        self.root = root
        self.detector = detector
        self.log_dir = log_dir
        self.messages = Queue()
        self.stop_event = Event()
        self.running = False
        self.closing = False
        self.port = tk.StringVar()
        self.sound = tk.BooleanVar(value=True)
        self.risk = tk.StringVar(value="Điểm nguy cơ  —")

        root.title("N16 | Nhận biết buồn ngủ")
        root.minsize(600, 340)
        root.protocol("WM_DELETE_WINDOW", self.close)
        content = ttk.Frame(root, padding=20)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Nhận biết buồn ngủ từ NeuroSky",
                  font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(0, 16))
        ttk.Label(content, text=f"Ngưỡng cảnh báo {detector.threshold:.2f}".replace(".", ","))\
            .pack(anchor="w", pady=(0, 8))

        connection = ttk.Frame(content)
        connection.pack(fill="x")
        ttk.Label(connection, text="Cổng Bluetooth").pack(side="left", padx=(0, 10))
        self.ports = ttk.Combobox(connection, textvariable=self.port, width=12)
        self.ports.pack(side="left")
        self.refresh_button = ttk.Button(connection, text="Quét cổng", command=self.refresh_ports)
        self.refresh_button.pack(side="left", padx=8)
        self.sound_button = ttk.Checkbutton(connection, text="Bật cảnh báo âm thanh", variable=self.sound)
        self.sound_button.pack(side="right")

        self.status = ttk.Label(content, text="Chọn cổng COM và nhấn Bắt đầu.",
                                font=("Segoe UI", 12, "bold"), wraplength=500)
        self.status.pack(anchor="w", pady=(24, 4))
        ttk.Label(content, textvariable=self.risk).pack(anchor="w")

        actions = ttk.Frame(content)
        actions.pack(fill="x", pady=(20, 0))
        self.start_button = ttk.Button(actions, text="Bắt đầu", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Dừng", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Thử loa", command=self.test_sound).pack(side="left")
        ttk.Button(actions, text="Mở thư mục log", command=self.open_logs).pack(side="right")

        ttk.Separator(content).pack(fill="x", pady=14)
        self.csv_button = ttk.Button(content, text="Chọn CSV để kiểm tra", command=self.choose_csv)
        self.csv_button.pack(anchor="w")
        ttk.Label(content, text="Đọc CSV raw OpenViBE hoặc N16 và lưu dự đoán vào thư mục log.",
                  wraplength=520).pack(anchor="w", pady=(4, 0))

        self.refresh_ports()
        self.root.after(75, self.poll)

    def refresh_ports(self) -> None:
        try:
            ports = [entry.device for entry in sorted(list_ports.comports())]
        except OSError:
            ports = []
        self.ports.configure(values=ports)
        if len(ports) == 1 and not self.port.get():
            self.port.set(ports[0])
        if not ports:
            self.status.configure(text="Chưa thấy cổng COM. Ghép Bluetooth rồi nhấn Quét cổng.")
        else:
            self.status.configure(text="Chọn cổng COM của NeuroSky rồi nhấn Bắt đầu.")

    def start(self) -> None:
        port = self.port.get().strip().upper()
        if not re.fullmatch(r"COM[1-9][0-9]*", port):
            messagebox.showinfo("Chọn cổng", "Chọn hoặc nhập cổng NeuroSky, ví dụ COM4.", parent=self.root)
            return
        self.stop_event.clear()
        self.detector.reset()
        self.set_running(True)
        self.status.configure(text=f"Đang chờ EEG từ {port}…", foreground="#263449")
        self.risk.set("Điểm nguy cơ  —")
        log = self.log_dir / f"N16_{port}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        Thread(target=self.run, args=(port, log, self.sound.get()), daemon=True).start()

    def run(self, port: str, log: Path, sound: bool) -> None:
        try:
            run_live(port, self.detector, log, sound=sound, stop=self.stop_event,
                     on_decision=lambda row: self.messages.put(("decision", row)))
        except (OSError, RuntimeError, ValueError) as exc:
            self.messages.put(("error", str(exc)))
        finally:
            self.messages.put(("done", None))

    def choose_csv(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("CSV EEG", "*.csv")])
        if not path:
            return
        source = Path(path)
        log = self.log_dir / f"CSV_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        self.stop_event.clear()
        self.set_running(True)
        self.status.configure(text=f"Đang kiểm tra {source.name}…", foreground="#263449")
        self.risk.set("Điểm nguy cơ  —")
        Thread(target=self.run_csv, args=(source, log), daemon=True).start()

    def run_csv(self, source: Path, log: Path) -> None:
        try:
            result = test_recording(source, log, self.detector, self.stop_event, screen_raw=True)
            self.messages.put(("csv_done", (result, log)))
        except (OSError, ValueError) as exc:
            self.messages.put(("csv_error", str(exc)))

    def set_running(self, running: bool) -> None:
        self.running = running
        idle_state = "disabled" if running else "normal"
        for widget in (self.ports, self.start_button, self.refresh_button, self.sound_button, self.csv_button):
            widget.configure(state=idle_state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def stop(self) -> None:
        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status.configure(text="Đang dừng…", foreground="#263449")

    def close(self) -> None:
        self.closing = True
        if self.running:
            self.stop()
            self.root.after(1500, self.root.destroy)
        else:
            self.root.destroy()

    def poll(self) -> None:
        error = None
        try:
            while True:
                kind, data = self.messages.get_nowait()
                if kind == "decision" and not self.stop_event.is_set():
                    label, color = STATES[data["state"]]
                    self.status.configure(text=label, foreground=color)
                    score = data.get("risk", "")
                    self.risk.set(f"Điểm nguy cơ  {score:.2f}" if score != "" else "Điểm nguy cơ  —")
                elif kind == "error":
                    error = data
                elif kind in {"csv_done", "csv_error"}:
                    if self.closing:
                        self.root.destroy()
                        return
                    self.set_running(False)
                    self.show_csv_result(kind, data)
                elif kind == "done":
                    if self.closing:
                        self.root.destroy()
                        return
                    self.set_running(False)
                    self.risk.set("Điểm nguy cơ  —")
                    self.status.configure(text="Đã dừng.", foreground="#263449")
                    if error:
                        self.status.configure(text="Không thể đọc dữ liệu EEG.", foreground="#B01828")
                        messagebox.showerror("Lỗi kết nối", f"Kiểm tra cổng COM và đóng OpenViBE nếu đang dùng cổng.\n\n{error}", parent=self.root)
        except Empty:
            self.root.after(75, self.poll)

    def show_csv_result(self, kind, data) -> None:
        if kind == "csv_error":
            self.status.configure(text="Không đọc được CSV.", foreground="#B01828")
            messagebox.showerror("Lỗi CSV", data, parent=self.root)
            return
        result, log = data
        if result is None:
            self.status.configure(text="Đã dừng kiểm tra CSV.", foreground="#263449")
            return
        self.status.configure(text="Đã kiểm tra CSV và lưu kết quả.", foreground="#146747")
        self.risk.set(f"{result['windows']:,} cửa sổ có điểm  |  {result['episodes']:,} đợt cảnh báo")
        quality_note = ("" if result["device_quality"] else
                        "\nCSV thiếu poor_signal. Dự đoán chạy với chất lượng tín hiệu chưa xác minh.")
        messagebox.showinfo("Kết quả kiểm tra CSV", f"{self.risk.get()}\n"
                            f"{result['poor_windows']:,} cửa sổ bị loại, gồm "
                            f"{result['artifact_windows']:,} cửa sổ có mẫu bão hòa hoặc đứng giá trị.\n"
                            f"Kết quả dự đoán từng cửa sổ được lưu tại\n{log}{quality_note}", parent=self.root)

    def open_logs(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(self.log_dir)
        except OSError as exc:
            messagebox.showerror("Không mở được log", str(exc), parent=self.root)

    def test_sound(self) -> None:
        Thread(target=_beep, daemon=True).start()


def main() -> None:
    here = Path(__file__).resolve().parent
    release = (here / "n16_release" if getattr(sys, "frozen", False)
               else here.parent / "output" / "N16_Chot_20261110")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Test startup without EEG hardware")
    parser.add_argument("--csv", type=Path, help="Run the CSV test from the command line")
    parser.add_argument("--output", type=Path, help="Output CSV for --csv")
    parser.add_argument("--release-dir", type=Path, help="Directory containing release.json and the model")
    args = parser.parse_args()
    if args.release_dir is not None:
        release = args.release_dir
    if bool(args.csv) != bool(args.output):
        parser.error("--csv and --output must be used together")
    if args.csv:
        test_recording(args.csv, args.output, load_detector(release), Event(), screen_raw=True)
        return
    root = tk.Tk()
    root.withdraw()
    try:
        detector = load_detector(release)
        log_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "N16" / "logs"
        LiveWindow(root, detector, log_dir)
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
