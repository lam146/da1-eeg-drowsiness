"""Native Tk presentation for the live and recorded EEG monitor."""

import tkinter as tk
from . import __version__
from math import isfinite
from tkinter import ttk


PAGE = "#edf2f5"
WHITE = "#ffffff"
INK = "#142e40"
MUTED = "#485f70"
BORDER = "#cedae1"
NAVY = "#102b3b"
TEAL = "#086d65"
FONT = "Segoe UI"


def label(parent, text, size=10, color=INK, bold=False, **options):
    return tk.Label(parent, text=text, font=(FONT, size, "bold" if bold else "normal"),
                    bg=parent["bg"], fg=color, bd=0, anchor="w", **options)


def card(parent):
    return tk.Frame(parent, bg=WHITE, highlightthickness=1, highlightbackground=BORDER)


def button(parent, text, command, primary=False):
    return ttk.Button(parent, text=text, command=command,
                      style="Primary.TButton" if primary else "Monitor.TButton")


class MonitorHUD:
    def __init__(self, app):
        self.app = app
        self.root = root = app.root
        self.fs = app.detector.fs
        self.values = []
        self.plot_hint = "Chờ dữ liệu EEG"
        self.compact = None
        root.title(f"N16 EEG Monitor · v{__version__}")
        root.configure(bg=PAGE)
        root.geometry(f"{min(1100, root.winfo_screenwidth()-40)}x{min(738, root.winfo_screenheight()-95)}")
        root.minsize(480, 400)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=(FONT, 10))
        for name, bg, fg, hover in [
            ("Monitor.TButton", "#eef3f5", INK, "#dce9ec"),
            ("Primary.TButton", TEAL, WHITE, "#07584f"),
        ]:
            style.configure(name, background=bg, foreground=fg, padding=(9, 6),
                            borderwidth=0, focusthickness=1, focuscolor="#78b5b4")
            style.map(name, background=[("disabled", "#eff2f3"), ("pressed", hover), ("active", hover)],
                      foreground=[("disabled", "#788995")])
        style.configure("TEntry", fieldbackground=WHITE, padding=4)
        style.configure("TCombobox", padding=4)
        style.configure("TCheckbutton", background=WHITE)
        style.map("TCheckbutton", background=[("active", WHITE)])

        header = tk.Frame(root, bg=NAVY, padx=18, pady=10)
        header.pack(fill="x")
        brand = tk.Canvas(header, width=32, height=32, bg="#174c52", highlightthickness=0)
        brand.pack(side="left", padx=(0, 10))
        brand.create_line(3, 17, 9, 17, 12, 10, 16, 24, 20, 7, 23, 17, 29, 17,
                          fill="#84ebd4", width=2)
        label(header, "N16 EEG Monitor", 15, WHITE, True).pack(side="left")
        app.csv_button = button(header, "Mở tệp CSV…", app.choose_csv)
        app.csv_button.pack(side="right")
        if app.reference_directory is not None:
            app.compare_button = button(header, "So sánh 2 bản…", app.choose_comparison)
            app.compare_button.pack(side="right", padx=(0, 5))

        # Keep controls reachable on smaller displays and at larger Windows font scales.
        area = tk.Frame(root, bg=PAGE)
        area.pack(fill="both", expand=True)
        self.viewport = tk.Canvas(area, bg=PAGE, highlightthickness=0)
        self.scroll = ttk.Scrollbar(area, orient="vertical", command=self.viewport.yview)
        self.viewport.configure(yscrollcommand=self.scroll.set)
        self.viewport.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.viewport, bg=PAGE, padx=14, pady=14)
        self.item = self.viewport.create_window(0, 0, window=self.body, anchor="nw")
        self.main = tk.Frame(self.body, bg=PAGE)
        self.rail = card(self.body)
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)
        self.monitor_panel()
        self.control_panel()
        self.viewport.bind("<Configure>", self.layout)
        self.body.bind("<Configure>", self.scroll_region)
        root.bind("<MouseWheel>", self.wheel, add="+")

    def layout(self, event):
        compact = event.width < self.root.winfo_fpixels("1i") * 8.15
        if compact != self.compact:
            self.compact = compact
            self.main.grid(row=0, column=0, sticky="nsew", padx=(0, 0 if compact else 12))
            self.rail.grid(row=int(compact), column=int(not compact), sticky="new",
                           pady=(12 if compact else 0, 0))
        self.viewport.itemconfigure(self.item, width=event.width,
                                   height=max(event.height, self.body.winfo_reqheight()))

    def scroll_region(self, _event):
        height = max(self.viewport.winfo_height(), self.body.winfo_reqheight())
        self.viewport.itemconfigure(self.item, height=height)
        self.viewport.configure(scrollregion=(0, 0, self.viewport.winfo_width(), height))
        if height > self.viewport.winfo_height():
            self.scroll.pack(side="right", fill="y")
        else:
            self.scroll.pack_forget()

    def wheel(self, event):
        if self.scroll.winfo_ismapped() and event.widget.winfo_class() != "TCombobox":
            self.viewport.yview_scroll(-int(event.delta / 120), "units")

    def monitor_panel(self):
        app = self.app
        hero = card(self.main)
        hero.pack(fill="x")
        self.stripe = tk.Frame(hero, width=5, bg="#688493")
        self.stripe.pack(side="left", fill="y")
        inner = tk.Frame(hero, bg=WHITE, padx=14, pady=12)
        inner.pack(fill="x")
        self.mode = label(inner, "THEO DÕI TRỰC TIẾP", 9, TEAL, True)
        self.mode.pack(anchor="w")
        self.title = label(inner, "Chờ bắt đầu", 18, INK, True)
        self.title.pack(anchor="w")
        self.detail = label(inner, "", 10, MUTED, justify="left")
        self.detail.pack(fill="x")
        inner.bind("<Configure>", lambda e: self.detail.configure(wraplength=max(180, e.width-32)))

        metrics = tk.Frame(self.main, bg=PAGE)
        metrics.pack(fill="x", pady=9)
        self.metric_labels, self.metric_values = [], []
        for index in range(3):
            metrics.columnconfigure(index, weight=1, uniform="metric")
            tile = card(metrics)
            tile.grid(row=0, column=index, sticky="nsew", padx=(0, 7 if index < 2 else 0))
            caption = label(tile, "", 9, MUTED)
            caption.pack(anchor="w", padx=10, pady=(7, 0))
            value = label(tile, "", 14, INK, True)
            value.pack(anchor="w", padx=10, pady=(0, 7))
            self.metric_labels.append(caption)
            self.metric_values.append(value)
        self.live_metrics()

        scope = tk.Frame(self.main, bg=NAVY, padx=12, pady=10)
        scope.pack(fill="both", expand=True)
        self.scope_title = label(scope, "Dạng sóng EEG", 12, WHITE, True)
        self.scope_title.pack(anchor="w")
        self.scope_unit = label(scope, f"EEG gốc  ·  {self.fs} Hz  ·  count", 9, "#a9c5d0")
        self.scope_unit.pack(anchor="w")
        self.plot = tk.Canvas(scope, bg=NAVY, height=130, highlightthickness=0, width=150)
        self.plot.pack(fill="both", expand=True, pady=(6, 0))
        self.plot.bind("<Configure>", lambda _e: self.draw_trace())
        self.foot = label(self.main, "", 9, MUTED, justify="left")
        self.foot.pack(fill="x", pady=(6, 0))
        self.foot.configure(text=f"Ngưỡng cảnh báo  {app.detector.threshold:g}   ·   Phổ {app.detector.spectral_estimator.replace('_', ' ')}")
        actions = tk.Frame(self.main, bg=PAGE)
        actions.pack(fill="x", pady=(7, 0))
        button(actions, "Mở thư mục dữ liệu", app.open_logs).pack(side="left")
        app.review_button = button(actions, "Xem lại phiên", app.open_review)
        app.review_button.configure(state="disabled")
        app.review_button.pack(side="left", padx=(5, 0))
        app.export_button = button(actions, "Xuất phiên ZIP…", app.export_session)
        app.export_button.configure(state="disabled")
        app.export_button.pack(side="left", padx=6)

    def control_panel(self):
        app = self.app
        box = tk.Frame(self.rail, bg=WHITE, padx=12, pady=10)
        box.pack(fill="x")
        label(box, "Thiết lập phiên đo", 13, INK, True).pack(anchor="w")
        label(box, "Cổng NeuroSky", 9, MUTED).pack(anchor="w", pady=(6, 1))
        row = tk.Frame(box, bg=WHITE)
        row.pack(fill="x")
        app.ports = ttk.Combobox(row, textvariable=app.port, width=11)
        app.ports.pack(side="left", expand=True, fill="x")
        app.refresh_button = button(row, "Quét", app.refresh_ports)
        app.refresh_button.pack(side="right", padx=(4, 0))
        self.port_hint = label(box, "", 9, MUTED)
        self.port_hint.pack(anchor="w")
        label(box, "Mã người đo", 9, MUTED).pack(anchor="w", pady=(6, 1))
        app.subject_entry = ttk.Entry(box, textvariable=app.subject, width=15)
        app.subject_entry.pack(fill="x")
        row = tk.Frame(box, bg=WHITE)
        row.pack(fill="x", pady=(8, 0))
        app.sound_button = ttk.Checkbutton(row, text="Âm cảnh báo", variable=app.sound, command=app.update_sound)
        app.sound_button.pack(side="left")
        button(row, "Thử loa", app.test_sound).pack(side="right")
        row = tk.Frame(box, bg=WHITE)
        row.pack(fill="x", pady=(8, 0))
        app.start_button = button(row, "Bắt đầu đo", app.start, True)
        app.start_button.pack(side="left", fill="x", expand=True)
        app.stop_button = button(row, "Dừng", app.stop)
        app.stop_button.configure(state="disabled")
        app.stop_button.pack(side="right", padx=(5, 0))
        ttk.Separator(box).pack(fill="x", pady=12)
        label(box, "Nhật ký người đo", 12, INK, True).pack(anchor="w")
        label(box, "Điểm buồn ngủ KSS", 9, MUTED).pack(anchor="w", pady=(5, 1))
        row = tk.Frame(box, bg=WHITE)
        row.pack(fill="x")
        app.kss_entry = ttk.Combobox(row, textvariable=app.kss, width=4, state="readonly",
                                     values=list(range(1, 10)))
        app.kss_entry.pack(side="left", expand=True, fill="x")
        app.kss_button = button(row, "Ghi KSS", lambda: app.mark("KSS"))
        app.kss_button.configure(state="disabled")
        app.kss_button.pack(side="right", padx=(5, 0))
        app.feedback_button = button(box, "Đánh dấu đang tỉnh", lambda: app.mark("SELF_REPORT_AWAKE"))
        app.feedback_button.configure(state="disabled")
        app.feedback_button.pack(fill="x", pady=(6, 4))
        app.drowsy_button = button(box, "Đánh dấu buồn ngủ", lambda: app.mark("SELF_REPORT_DROWSY"))
        app.drowsy_button.configure(state="disabled")
        app.drowsy_button.pack(fill="x", pady=(0, 4))
        label(box, "", 9, MUTED, textvariable=app.kss_hint, wraplength=215, justify="left").pack(anchor="w")

    def status(self, title, detail="", color=INK):
        self.title.configure(text=title, fg=color)
        self.stripe.configure(bg=color)
        self.detail.configure(text=detail)

    def metrics(self, names, values):
        for caption, number, name, value in zip(self.metric_labels, self.metric_values, names, values):
            caption.configure(text=name)
            number.configure(text=value)

    def live_metrics(self, risk="—", elapsed="—", contact="—"):
        self.metrics(("Điểm nguy cơ", "Thời gian đo", "Tiếp xúc"), (risk, elapsed, contact))

    def reset_plot(self, csv=False):
        self.values = []
        self.mode.configure(text="KIỂM TRA CSV" if csv else "THEO DÕI TRỰC TIẾP")
        self.scope_title.configure(text="Kết quả phân tích" if csv else "Dạng sóng EEG")
        self.scope_unit.configure(text="" if csv else f"EEG gốc  ·  {self.fs} Hz  ·  count")
        self.plot_hint = "Đang phân tích…" if csv else "Chờ dữ liệu EEG"
        self.draw_trace()

    def draw_trace(self):
        canvas = self.plot
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if min(width, height) < 20:
            return
        if self.mode["text"] == "KIỂM TRA CSV":
            canvas.create_text(8, 12, anchor="nw", text=self.plot_hint, fill="#d7ecee",
                               width=max(100, width-20), font=(FONT, 11))
            return
        x0, x1, y0, y1 = 48, width-8, 10, height-24
        if x1 <= x0 or y1 <= y0:
            return
        for i in range(5):
            y = y0 + (y1-y0)*i/4
            canvas.create_line(x0, y, x1, y, fill="#284756")
        for i in range(9):
            x = x0 + (x1-x0)*i/8
            canvas.create_line(x, y0, x, y1, fill="#284756")
        if len(self.values) < 2:
            canvas.create_text((x0+x1)/2, (y0+y1)/2, text=self.plot_hint,
                               font=(FONT, 11), fill="#c2d6dd")
            return
        values = self.values
        low, high = min(values), max(values)
        if low == high:
            low, high = low-1, high+1
        for value, y in [(low, y1), (high, y0)]:
            canvas.create_text(x0-5, y, anchor="e", text=f"{value:.0f}", fill="#bed1d8")
        duration = (len(values)-1)/self.fs
        for x, anchor, tick in [(x0, "w", f"−{duration:.1f} s"), (x1, "e", "0 s")]:
            canvas.create_text(x, y1+11, anchor=anchor, text=tick, fill="#bed1d8")
        coords = []
        # Keep the envelope of short spikes when many samples map to a single pixel.
        group = max(1, (len(values)+x1-x0-1)//(x1-x0))
        for start in range(0, len(values), group):
            chunk = values[start:start+group]
            indices = [0] if len(chunk) == 1 else sorted({chunk.index(min(chunk)), chunk.index(max(chunk))})
            for local in indices:
                coords.extend((x0+(start+local)*(x1-x0)/(len(values)-1),
                               y1-(chunk[local]-low)*(y1-y0)/(high-low)))
        if len(coords) >= 4:
            canvas.create_line(*coords, fill="#6ce4c8", width=1.5)

    def trace(self, data):
        self.values = [value for value in data if isfinite(value)]
        self.draw_trace()
