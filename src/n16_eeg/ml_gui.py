"""Vietnamese desktop CSV-to-ML workbench. No COM access or automatic labels."""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
from pathlib import Path

from . import ml_workbench as work

ROOT = Path(__file__).resolve().parents[2]


class MLApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from tkinter.scrolledtext import ScrolledText
        self.tk, self.ttk = tk, ttk
        self.dialog, self.messages = filedialog, messagebox
        self.root = root
        self.frame = None
        self.raw_path = None
        self.feature_paths = []
        self.last_output = None
        self.busy = False
        self.jobs = queue.Queue()
        self.buttons = []
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.title("N16 | Nạp CSV và học máy")
        root.geometry("1160x850")
        root.minsize(1000, 740)
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("TLabel", font=("Segoe UI", 10))
        outer = ttk.Frame(root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="N16 • Từ file CSV đến mô hình học máy", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="1. Xem file  →  2. Tạo đặc trưng khi có nhãn thật  →  3. Huấn luyện theo người").pack(anchor="w", pady=(4, 10))
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        inspect_tab, prepare_tab, train_tab, seed_tab = [ttk.Frame(self.tabs, padding=12) for _ in range(4)]
        for pane, name in zip([inspect_tab, prepare_tab, train_tab],
                              ["1. Nạp và xem CSV", "2. Tín hiệu → đặc trưng", "3. Huấn luyện và kết quả"]):
            self.tabs.add(pane, text=name)
        self.tabs.add(seed_tab, text="4. SEED-VIG")

        toolbar = ttk.Frame(inspect_tab); toolbar.pack(fill="x")
        self.add_button(toolbar, "Chọn file CSV...", self.choose_raw).pack(side="left")
        self.path_var = tk.StringVar(value="Chưa chọn file")
        ttk.Label(toolbar, textvariable=self.path_var, wraplength=860).pack(side="left", padx=12)
        self.inspect_text = ScrolledText(inspect_tab, height=8, font=("Consolas", 10), wrap="word")
        self.inspect_text.pack(fill="x", pady=10)
        self.set_text(self.inspect_text, "Chọn CSV OpenViBE (Time:512Hz, Electrode), CSV raw N16 hoặc bảng đặc trưng.\n"
                      "CSV raw chưa có nhãn vẫn xem được, nhưng không được tự huấn luyện tỉnh/buồn ngủ.")
        ttk.Label(inspect_tab, text="Xem trước tối đa 40 dòng, 12 cột (không sửa file gốc)").pack(anchor="w")
        preview_frame = ttk.Frame(inspect_tab); preview_frame.pack(fill="both", expand=True, pady=5)
        self.preview = ttk.Treeview(preview_frame, show="headings", height=7)
        xscroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview.xview)
        yscroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview.yview)
        self.preview.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.preview.grid(row=0, column=0, sticky="nsew"); yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        preview_frame.rowconfigure(0, weight=1); preview_frame.columnconfigure(0, weight=1)
        ttk.Label(inspect_tab, text="Dạng tín hiệu raw — biểu đồ để xem, không chứng minh chất lượng hoặc nhãn").pack(anchor="w")
        self.plot = tk.Canvas(inspect_tab, height=155, background="white", highlightthickness=1)
        self.plot.pack(fill="x", pady=5)
        self.plot.bind("<Configure>", lambda event: self.draw_plot())

        ttk.Label(prepare_tab, text="Dùng file raw ở tab 1. Cửa sổ 4 giây, bước 1 giây, 512 Hz.").pack(anchor="w")
        ttk.Label(prepare_tab, text="Không lấy cột KSS rời rạc rồi tự kéo nhãn. Chỉ dùng các khoảng đã được bạn đối chiếu.", wraplength=1000).pack(anchor="w", pady=(3, 10))
        config = ttk.LabelFrame(prepare_tab, text="Thông tin phiên", padding=10); config.pack(fill="x")
        self.subject = tk.StringVar(); self.session = tk.StringVar(); self.unit = tk.StringVar()
        self.protocol = tk.StringVar()
        for row, (label, variable) in enumerate([("Mã người (ví dụ S01)", self.subject),
                  ("Mã phiên (không đổi người)", self.session), ("Tên quy tắc gán nhãn đã thống nhất", self.protocol)]):
            ttk.Label(config, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(config, textvariable=variable, width=48).grid(row=row, column=1, sticky="ew", padx=10)
        ttk.Label(config, text="Đơn vị raw đã xác nhận").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Combobox(config, textvariable=self.unit, values=["count", "uV"], state="readonly", width=14).grid(row=3, column=1, sticky="w", padx=10)
        config.columnconfigure(1, weight=1)
        self.annotations = tk.StringVar()
        ann = ttk.LabelFrame(prepare_tab, text="File khoảng nhãn và chất lượng", padding=10); ann.pack(fill="x", pady=10)
        row = ttk.Frame(ann); row.pack(fill="x")
        ttk.Entry(row, textvariable=self.annotations).pack(side="left", fill="x", expand=True)
        self.add_button(row, "Chọn file nhãn...", self.choose_annotations).pack(side="left", padx=6)
        self.add_button(ann, "Tạo mẫu nhãn trống...", self.write_annotation_template).pack(anchor="w", pady=7)
        ttk.Label(ann, text="Cột: start_s, end_s, kss, quality_status\n"
                  "Thời gian tính từ mẫu đầu CSV = 0 giây. usable = đã xem và dùng được; exclude = loại; unknown = chưa rõ.\n"
                  "Cắt vùng lỗi thành khoảng riêng. Không có nhãn/quality thì giữ trống, không tự tạo dữ liệu.",
                  wraplength=1000).pack(anchor="w")
        self.reviewed = tk.BooleanVar(value=False)
        ttk.Checkbutton(prepare_tab, text="Tôi đã xác nhận đơn vị, căn thời gian nhãn và xem chất lượng các khoảng usable.",
                        variable=self.reviewed).pack(anchor="w", pady=6)
        self.add_button(prepare_tab, "Tạo đặc trưng và lưu thư mục mới...", self.prepare).pack(anchor="w", pady=6)
        ttk.Label(prepare_tab, text="KSS 1–5 → tỉnh; 7–9 → buồn ngủ; 6 bỏ khỏi baseline. Đây là quy tắc N16 cần được thống nhất.\n"
                  "Không suy ra buồn ngủ từ chớp theo lệnh. Không tạo poorSignal=0 khi thiếu.\n"
                  "Các khoảng quality thủ công được ghi là thủ công, không giả thành chất lượng thiết bị đo.", wraplength=1000).pack(anchor="w", pady=10)
        self.preparation_status = tk.StringVar(value="Chưa tạo đặc trưng. File test chớp mắt không tự có nhãn buồn ngủ.")
        ttk.Label(prepare_tab, textvariable=self.preparation_status, wraplength=1000).pack(anchor="w")

        ttk.Label(train_tab, text="Nạp bảng đặc trưng có subject_id và label (0/1). Không nạp Electrode trực tiếp vào classifier.").pack(anchor="w")
        trainbar = ttk.Frame(train_tab); trainbar.pack(fill="x", pady=8)
        self.add_button(trainbar, "Chọn các CSV đặc trưng...", self.choose_features).pack(side="left")
        self.add_button(trainbar, "Bỏ danh sách", self.clear_features).pack(side="left", padx=6)
        self.file_list = tk.Listbox(train_tab, height=3, font=("Segoe UI", 10)); self.file_list.pack(fill="x")
        settings = ttk.Frame(train_tab); settings.pack(fill="x", pady=8)
        self.dataset = tk.StringVar(value="openvibe"); self.classifier = tk.StringVar(value="logistic")
        ttk.Label(settings, text="Nguồn:").pack(side="left")
        ttk.Combobox(settings, textvariable=self.dataset, values=["openvibe", "neurosky", "seed-vig"], state="readonly", width=15).pack(side="left", padx=6)
        ttk.Label(settings, text="Mô hình:").pack(side="left", padx=(14, 0))
        ttk.Combobox(settings, textvariable=self.classifier, values=["logistic", "svm", "random-forest"], state="readonly", width=20).pack(side="left", padx=6)
        middle = ttk.Frame(train_tab); middle.pack(fill="x")
        choice = ttk.LabelFrame(middle, text="Đặc trưng dùng để học (Ctrl/Shift để chọn)", padding=7)
        choice.pack(side="left", fill="both", expand=True)
        self.feature_list = tk.Listbox(choice, selectmode="extended", exportselection=False, height=7)
        scroll = ttk.Scrollbar(choice, command=self.feature_list.yview)
        self.feature_list.configure(yscrollcommand=scroll.set)
        self.feature_list.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        actions = ttk.Frame(middle, padding=(10, 0)); actions.pack(side="left", fill="y")
        for label, group in [("Chọn tất cả", "all"), ("Chỉ phổ", "spectral"), ("Chỉ chớp mắt", "blink"),
                             ("Đặc trưng SEED-VIG", "seed")]:
            self.add_button(actions, label, lambda group=group: self.select_feature_group(group)).pack(fill="x", pady=3)
        self.add_button(actions, "Huấn luyện + LOSO...", self.train).pack(fill="x", pady=10)
        self.add_button(actions, "Mở thư mục kết quả", self.open_output).pack(fill="x")
        ttk.Label(train_tab, text="LOSO: kiểm tra từng người bằng mô hình không học từ người đó. Cần ≥2 người và đủ lớp trong train.\n"
                  "Kết quả là đánh giá ngoại tuyến. Joblib không phải model JSON để chạy C++; chưa chứng minh cảnh báo live.",
                  wraplength=1020).pack(anchor="w", pady=8)
        self.results = ScrolledText(train_tab, height=10, font=("Consolas", 10), wrap="word")
        self.results.pack(fill="both", expand=True)
        self.set_text(self.results, "Chưa huấn luyện. Chọn file, nguồn dữ liệu và các đặc trưng cần dùng.")
        self.build_seed_tab(seed_tab, ScrolledText)
        self.status = tk.StringVar(value="Sẵn sàng. Không thay đổi file CSV gốc.")
        ttk.Label(outer, textvariable=self.status, wraplength=1080).pack(anchor="w", pady=(8, 0))
        root.after(100, self.poll)

    def close(self):
        if self.busy:
            self.messages.showinfo("Đang xử lý", "Chờ xử lý xong rồi đóng để lưu kết quả đầy đủ.")
            return
        self.root.destroy()

    def build_seed_tab(self, pane, text_class):
        ttk, tk = self.ttk, self.tk
        ttk.Label(pane, text="Nhập đặc trưng MAT SEED-VIG đã được cấp quyền, đánh giá riêng với dữ liệu NeuroSky.",
                  wraplength=1010).pack(anchor="w")
        ttk.Label(pane, text="Chưa có dữ liệu? Xin quyền qua trang chính thức trước. Ứng dụng không tự tải hoặc ký giấy phép.",
                  wraplength=1010).pack(anchor="w", pady=(4, 10))
        bar = ttk.Frame(pane); bar.pack(fill="x")
        self.add_button(bar, "Xem biến trong file MAT...", self.inspect_seed_mat).pack(side="left")
        self.add_button(bar, "Tạo manifest mẫu trống...", self.seed_template).pack(side="left", padx=8)
        self.add_button(bar, "Dữ liệu đã tải", self.show_local_seed).pack(side="left")
        self.seed_manifest = tk.StringVar()
        manifest = ttk.LabelFrame(pane, text="Manifest đã điền từ hồ sơ dữ liệu thật", padding=10)
        manifest.pack(fill="x", pady=10)
        ttk.Entry(manifest, textvariable=self.seed_manifest).pack(side="left", fill="x", expand=True)
        self.add_button(manifest, "Chọn JSON...", self.choose_seed_manifest).pack(side="left", padx=8)
        settings = ttk.LabelFrame(pane, text="Quy tắc nhãn PERCLOS đã chốt trước khi xem kết quả kiểm tra", padding=10)
        settings.pack(fill="x")
        self.seed_alert, self.seed_drowsy, self.seed_policy = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for row, (label, variable) in enumerate([
            ("Tỉnh khi PERCLOS ≤ (0 đến 1)", self.seed_alert),
            ("Buồn ngủ khi PERCLOS ≥ (lớn hơn ngưỡng tỉnh)", self.seed_drowsy),
            ("Tên quy tắc và căn cứ chọn ngưỡng", self.seed_policy)]):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(settings, textvariable=variable, width=44).grid(row=row, column=1, sticky="ew", padx=10)
        settings.columnconfigure(1, weight=1)
        ttk.Label(pane, text="Không có ngưỡng mặc định. Dùng dấu chấm thập phân; vùng giữa hai ngưỡng sẽ bị loại.\n"
                  "Manifest cần mã người/phiên thật, file và tên biến MAT, trục mẫu và xác nhận căn chỉnh.\n"
                  "Không đổi tensor MAT thành raw 512 Hz, không đổi PERCLOS thành điểm KSS.",
                  wraplength=1010).pack(anchor="w", pady=10)
        self.seed_reviewed = tk.BooleanVar(value=False)
        ttk.Checkbutton(pane, variable=self.seed_reviewed,
            text="Tôi có quyền sử dụng; đã kiểm tra mã người, đồng bộ nhãn và quy tắc ngưỡng trước đánh giá.").pack(anchor="w")
        self.add_button(pane, "Nhập MAT thành CSV đặc trưng...", self.import_seed).pack(anchor="w", pady=10)
        self.seed_info = text_class(pane, height=10, font=("Consolas", 10), wrap="word")
        self.seed_info.pack(fill="both", expand=True)
        self.set_text(self.seed_info, "Chưa nhập dữ liệu SEED-VIG.\n"
                      "Hướng dẫn: docs/SEED_VIG_VA_BA_NGUON_GVHD.md\n"
                      "Cấp quyền: https://bcmi.sjtu.edu.cn/home/seed/downloads.html\n"
                      "Sau nhập, tab 3 được điền bảng đặc trưng và chọn nguồn seed-vig; bạn tự bấm huấn luyện.")
        if (ROOT / "data/external/seed-vig/_n16_inventory.json").is_file():
            self.show_local_seed()

    def show_local_seed(self):
        path = ROOT / "data/external/seed-vig/_n16_inventory.json"
        if not path.is_file():
            self.set_text(self.seed_info, "Chưa có hồ sơ kiểm tra bộ dữ liệu cục bộ.\n"
                          "Xem docs/SEED_VIG_DU_LIEU_THAT_20261028.md.")
            return
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            lines = ["ĐÃ CÓ DỮ LIỆU THẬT TRÊN MÁY", str(path.parent),
                     f"Số phiên có nhãn: {info['label_sessions']} (không đồng nghĩa số người)."]
            for name, layout in info["layouts"].items():
                lines.append(f"{name}: {layout['sessions']} phiên, {layout['rows']} hàng, "
                             f"{layout['flattened_features']} đặc trưng/hàng.")
            lines.extend(["Đã kiểm tra kích thước, số hữu hạn và nhãn PERCLOS trong [0,1].",
                          "Chưa huấn luyện; cần chốt mã người, căn chỉnh và ngưỡng PERCLOS.",
                          "Cấu hình nháp: output/seed_vig_local_20261028/manifest_*.draft.json",
                          "Hướng dẫn: docs/SEED_VIG_DU_LIEU_THAT_20261028.md",
                          "Đây là hồ sơ lần kiểm tra, không tự kiểm tra lại file sau khi bị thay đổi."])
            self.set_text(self.seed_info, "\n".join(lines))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.set_text(self.seed_info, f"Không đọc được hồ sơ SEED-VIG: {exc}")

    def inspect_seed_mat(self):
        name = self.dialog.askopenfilename(title="File MAT đã được cấp quyền", filetypes=[("MAT", "*.mat")])
        if not name: return
        from .seed_vig import inspect_mat
        self.run_job("Đang xem cấu trúc MAT", lambda: inspect_mat(Path(name)),
                     lambda info: self.set_text(self.seed_info, name + "\n" + json.dumps(info, ensure_ascii=False, indent=2)
                         + "\nĐây là tên biến/kích thước, chưa xác nhận được mã người hoặc nhãn đồng bộ.\n"
                           "Điền manifest theo hồ sơ dataset, không đoán người từ tên file."))

    def seed_template(self):
        name = self.dialog.asksaveasfilename(title="Lưu manifest mẫu chưa dùng được ngay",
                    initialfile="seed_vig_manifest.json", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not name: return
        try:
            # Relative example paths are intentionally incomplete, never auto-confirmed.
            template = (ROOT / "configs/seed_vig_manifest.example.json").read_text(encoding="utf-8")
            with Path(name).open("x", encoding="utf-8") as handle: handle.write(template)
            self.seed_manifest.set(name); self.seed_reviewed.set(False)
            self.messages.showinfo("Đã tạo mẫu", "Thay tên file, biến, trục mẫu và mã người/phiên bằng thông tin thật.\n"
                "Đường dẫn tương đối tính từ thư mục JSON. Chỉ đặt alignment_verified=true sau kiểm tra.\n"
                "Mẫu có REPLACE và chưa căn nhãn nên chưa dùng để nhập được.")
        except Exception as exc: self.messages.showerror("Không tạo được mẫu", str(exc))

    def choose_seed_manifest(self):
        local = ROOT / "output/seed_vig_local_20261028"
        name = self.dialog.askopenfilename(title="Manifest SEED-VIG", initialdir=local if local.is_dir() else ROOT / "configs",
                                          filetypes=[("JSON", "*.json")])
        if name:
            self.seed_manifest.set(name); self.seed_reviewed.set(False)

    def import_seed(self):
        try:
            path = Path(self.seed_manifest.get())
            if not self.seed_manifest.get() or not path.is_file():
                raise ValueError("Chọn manifest JSON đã điền và kiểm tra trước.")
            alert, drowsy = float(self.seed_alert.get()), float(self.seed_drowsy.get())
            if not (0 <= alert < drowsy <= 1):
                raise ValueError("Cần 0 ≤ ngưỡng tỉnh < ngưỡng buồn ngủ ≤ 1.")
            policy, reviewed = self.seed_policy.get().strip(), self.seed_reviewed.get()
            if not policy or not reviewed:
                raise ValueError("Ghi quy tắc ngưỡng và xác nhận quyền sử dụng/căn chỉnh dữ liệu trước.")
        except ValueError as exc:
            self.messages.showwarning("Kiểm tra đầu vào SEED-VIG", str(exc)); return
        parent = self.dialog.askdirectory(title="Nơi lưu lần nhập SEED-VIG mới", initialdir=ROOT / "output")
        if not parent: return
        def done(value):
            directory, report = value
            self.last_output = directory
            self.feature_paths = [directory / "features.csv"]
            self.file_list.delete(0, "end"); self.file_list.insert("end", str(self.feature_paths[0]))
            self.feature_list.delete(0, "end")
            for name in report["feature_names"]: self.feature_list.insert("end", name)
            self.feature_list.selection_set(0, "end"); self.dataset.set("seed-vig")
            message = (f"ĐÃ LƯU: {directory}\nNgười: {report['subjects']} | Hàng đặc trưng: {report['rows']}\n"
                       "Đã lưu features.csv và seed_import.json (ngưỡng, nguồn và SHA-256).\n"
                       "Tab 3 đã chọn nguồn seed-vig và các đặc trưng nhập. Chưa tự huấn luyện.\n"
                       "Cần ≥2 người thật và đủ lớp trong tập học để chạy LOSO.\n" + report["warning"])
            self.set_text(self.seed_info, message); self.set_text(self.results, message)
        self.run_job("Đang nhập dữ liệu SEED-VIG", lambda: work.import_seed_manifest(
            path, alert, drowsy, policy, reviewed, Path(parent)), done)

    def add_button(self, parent, label, command):
        button = self.ttk.Button(parent, text=label, command=command)
        self.buttons.append(button)
        return button

    @staticmethod
    def set_text(widget, value):
        widget.configure(state="normal"); widget.delete("1.0", "end"); widget.insert("1.0", value)
        widget.configure(state="disabled")

    def run_job(self, label, function, success):
        if self.busy:
            return
        self.busy = True
        self.status.set(label + " — vui lòng chờ, chưa đóng ứng dụng.")
        for button in self.buttons:
            button.configure(state="disabled")
        def worker():
            try:
                self.jobs.put((True, function(), success))
            except Exception as exc:
                self.jobs.put((False, str(exc), success))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            ok, value, success = self.jobs.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            for button in self.buttons:
                button.configure(state="normal")
            if ok:
                self.status.set("Đã hoàn tất. File gốc được giữ nguyên.")
                try:
                    success(value)
                except Exception as exc:
                    self.messages.showerror("Lỗi hiển thị", str(exc))
            else:
                self.status.set("Chưa thực hiện được — xem thông báo, không tự điền dữ liệu thiếu.")
                self.messages.showerror("Cần kiểm tra dữ liệu", value)
        self.root.after(100, self.poll)

    def choose_raw(self):
        name = self.dialog.askopenfilename(initialdir=ROOT / "raw", filetypes=[("CSV", "*.csv")])
        if name:
            self.load_path(Path(name))

    def load_path(self, path):
        def load():
            frame = work.read_csv_file(path)
            return frame, work.inspect_frame(frame)
        def loaded(value):
            frame, details = value
            self.frame, self.raw_path = frame, path
            self.path_var.set(str(path))
            kinds = {"openvibe_raw": "Tín hiệu gốc OpenViBE", "canonical_raw": "Tín hiệu gốc N16",
                     "features": "Bảng đặc trưng", "unknown": "Chưa nhận dạng"}
            lines = [f"Loại file: {kinds[details['kind']]}",
                     f"Số dòng: {details['rows']:,} | Số cột: {details['columns']}",
                     f"Ô thiếu trên toàn bộ bảng: {details['missing_cells']:,}"]
            if "duration_s" in details:
                lines.extend([f"Thời lượng: {details['duration_s']:.3f} giây | Tần số: 512 Hz",
                              f"Biên độ raw: {details['minimum']:g} đến {details['maximum']:g} (chưa xác nhận đơn vị)"])
            if "subjects" in details:
                lines.append(f"Số người: {details['subjects']} | Số cửa sổ mỗi nhãn: {details['label_counts']}")
            lines.append(details["note"])
            self.set_text(self.inspect_text, "\n".join(lines))
            self.preview.delete(*self.preview.get_children())
            columns = list(frame.columns[:12])
            self.preview.configure(columns=columns)
            for column in columns:
                self.preview.heading(column, text=column)
                self.preview.column(column, width=140, stretch=False)
            for row in frame[columns].head(40).itertuples(index=False, name=None):
                self.preview.insert("", "end", values=["" if str(v) in {"nan", "<NA>"} else str(v) for v in row])
            for name, variable in [("subject_id", self.subject), ("session_id", self.session), ("signal_unit", self.unit)]:
                variable.set(str(frame[name].iloc[0]) if name in frame and frame[name].nunique() == 1 else "")
            self.reviewed.set(False)
            self.annotations.set("")
            self.preparation_status.set("CSV đã nạp. Kiểm tra thông tin và file khoảng nhãn ở tab 2.")
            self.draw_plot()
        self.run_job("Đang đọc và kiểm tra CSV", load, loaded)

    def draw_plot(self):
        self.plot.delete("all")
        if self.frame is None or work.file_kind(self.frame) not in {"openvibe_raw", "canonical_raw"}:
            self.plot.create_text(12, 20, anchor="nw", text="Chỉ hiển thị đường tín hiệu khi nạp CSV raw.")
            return
        import numpy as np
        _, values, _ = work.raw_arrays(self.frame)
        width, height = max(self.plot.winfo_width(), 300), 155
        low, high = float(values.min()), float(values.max())
        self.plot.create_text(8, 5, anchor="nw", text=f"Min {low:g} | Max {high:g} — đơn vị cần xác nhận")
        # Min/max envelope keeps large transients visible without plotting millions of points.
        buckets = np.array_split(values, min(len(values), max(1, width - 24)))
        for i, bucket in enumerate(buckets):
            x = 12 + i * (width - 24) / max(len(buckets) - 1, 1)
            y1 = 30 + (high - float(bucket.max())) / max(high - low, 1e-9) * (height - 42)
            y2 = 30 + (high - float(bucket.min())) / max(high - low, 1e-9) * (height - 42)
            self.plot.create_line(x, y1, x, max(y1 + 1, y2), fill="#16648b")

    def choose_annotations(self):
        name = self.dialog.askopenfilename(filetypes=[("CSV nhãn", "*.csv")])
        if name:
            self.annotations.set(name)

    def write_annotation_template(self):
        name = self.dialog.asksaveasfilename(defaultextension=".csv", initialfile="khoang_nhan.csv",
                                            filetypes=[("CSV", "*.csv")])
        if name:
            try:
                with Path(name).open("x", encoding="utf-8", newline="") as handle:
                    handle.write("start_s,end_s,kss,quality_status\n")
                self.annotations.set(name)
                self.messages.showinfo("Đã tạo mẫu trống", "Điền khoảng dữ liệu thật theo nhật ký đã căn thời gian.\n"
                                       "Mẫu chưa có dữ liệu nên chưa thể tạo đặc trưng. Không đổi dấu phẩy trong số thập phân.")
            except Exception as exc:
                self.messages.showerror("Không tạo mẫu", str(exc))

    def prepare(self):
        if self.frame is None or self.raw_path is None:
            self.messages.showwarning("Chưa có raw", "Chọn CSV tín hiệu ở tab 1 trước."); return
        if not self.annotations.get():
            self.messages.showwarning("Thiếu nhãn", "Cần file khoảng nhãn/chất lượng thật. CSV raw một mình chưa đủ."); return
        if not self.reviewed.get() or not self.unit.get() or not self.protocol.get().strip():
            self.messages.showwarning("Chưa xác nhận", "Ghi đơn vị, tên quy tắc và xác nhận đã kiểm tra nhãn/chất lượng."); return
        parent = self.dialog.askdirectory(title="Nơi lưu thư mục đặc trưng mới", initialdir=ROOT / "output")
        if not parent:
            return
        raw_path, ann_path = self.raw_path, Path(self.annotations.get())
        source = self.frame.copy()
        arguments = (self.subject.get(), self.session.get(), self.unit.get(), self.reviewed.get(), self.protocol.get())
        def job():
            intervals = work.read_csv_file(ann_path)
            features, report = work.prepare_features(source, intervals, *arguments)
            directory = work.save_prepared(features, report, Path(parent), raw_path, ann_path)
            return directory, report
        def done(value):
            self.last_output, report = value
            self.preparation_status.set(f"Đã tạo {report['counts']['kept']} cửa sổ. Lưu ở: {self.last_output}\n"
                                       "Sang tab 3 chọn features.csv của nhiều người; không chọn CAL/chớp theo lệnh.")
            self.messages.showinfo("Đã tạo đặc trưng", str(self.last_output))
        self.run_job("Đang tạo đặc trưng theo khoảng nhãn", job, done)

    def choose_features(self):
        names = self.dialog.askopenfilenames(title="Chọn các bảng đặc trưng cùng schema/nguồn", filetypes=[("CSV", "*.csv")])
        if not names:
            return
        paths = [Path(name) for name in names]
        def job():
            frame = work.merge_features(paths)
            return frame, work.feature_candidates(frame)
        def done(value):
            frame, candidates = value
            self.feature_paths = paths
            self.file_list.delete(0, "end")
            for path in paths: self.file_list.insert("end", str(path))
            self.feature_list.delete(0, "end")
            for name in candidates: self.feature_list.insert("end", name)
            self.feature_list.selection_set(0, "end")
            if "dataset_id" in frame and frame["dataset_id"].nunique() == 1:
                self.dataset.set(str(frame["dataset_id"].iloc[0]))
            self.set_text(self.results, f"Đã nạp {len(paths)} file, {len(frame)} cửa sổ, "
                          f"{frame['subject_id'].nunique()} người.\n"
                          f"Nhãn: {frame['label'].value_counts().to_dict()}\n"
                          f"Có {len(candidates)} đặc trưng được hỗ trợ. Chọn nguồn và mô hình rồi chạy LOSO.\n"
                          "Không dùng người mới khác mã nếu vẫn là cùng một người thật.")
        self.run_job("Đang kiểm tra các bảng đặc trưng", job, done)

    def clear_features(self):
        self.feature_paths = []
        self.file_list.delete(0, "end"); self.feature_list.delete(0, "end")

    def select_feature_group(self, group):
        self.feature_list.selection_clear(0, "end")
        for i, name in enumerate(self.feature_list.get(0, "end")):
            blink = name.startswith("blink_") or name == "long_blink_count"
            seed = name.startswith("seed_")
            if (group == "all" or (group == "blink" and blink) or (group == "seed" and seed)
                    or (group == "spectral" and not blink and not seed)):
                self.feature_list.selection_set(i)

    def train(self):
        names = [self.feature_list.get(i) for i in self.feature_list.curselection()]
        if not self.feature_paths or not names:
            self.messages.showwarning("Thiếu đầu vào", "Chọn CSV đặc trưng và ít nhất một đặc trưng."); return
        parent = self.dialog.askdirectory(title="Nơi lưu lần huấn luyện mới", initialdir=ROOT / "output")
        if not parent:
            return
        paths, dataset, classifier = list(self.feature_paths), self.dataset.get(), self.classifier.get()
        def done(value):
            self.last_output, report = value
            lines = [f"ĐÃ LƯU: {self.last_output}", f"Nguồn: {report['dataset']} | Model: {report['classifier']}",
                     f"Người: {report['subjects']} | Cửa sổ: {report['windows']}"]
            if report["warning"]: lines.append("CẢNH BÁO: " + report["warning"])
            for key, metric in report["summary_by_person"].items():
                value = "N/A" if metric["mean"] is None else f"{metric['mean']:.4f}"
                lines.append(f"{key}: {value} (tính được trên {metric['people']} người)")
            lines.append("\nTỪNG NGƯỜI:")
            for fold in report["folds"]:
                lines.append(f"{fold['subject']}: {fold['status']}; n={fold['samples']}; "
                             f"F1={fold.get('macro_f1', 'N/A')}; AUC={fold.get('roc_auc', 'N/A')}")
                if fold.get("reason"): lines.append("  " + fold["reason"])
                if fold.get("confusion_matrix"): lines.append("  Ma trận [thật 0,1] x [đoán 0,1]: " + str(fold["confusion_matrix"]))
            lines.extend(["\nĐã lưu pipeline.joblib, metrics.json, oof_predictions.csv, inputs.json.",
                          "Kết quả ngoại tuyến, không phải chứng nhận an toàn hoặc kiểm chứng C++ live."])
            self.set_text(self.results, "\n".join(lines))
        self.run_job("Đang huấn luyện và đánh giá LOSO", lambda: work.train_files(paths, names, dataset, classifier, Path(parent)), done)

    def open_output(self):
        if self.last_output is None:
            self.messages.showinfo("Chưa có kết quả", "Tạo đặc trưng hoặc huấn luyện trước.")
        elif os.name == "nt":
            os.startfile(str(self.last_output))
        else:
            self.messages.showinfo("Thư mục kết quả", str(self.last_output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Build UI and exit, without opening input files")
    args = parser.parse_args()
    import tkinter as tk
    root = tk.Tk()
    if args.check: root.withdraw()
    app = MLApp(root)
    if args.check:
        root.update_idletasks()
        assert len(app.tabs.tabs()) == 4
        root.destroy()
        print("N16 ML GUI: 4 tabs built successfully")
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
