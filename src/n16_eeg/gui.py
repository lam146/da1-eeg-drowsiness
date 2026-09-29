"""Tkinter dashboard for replaying the N16 software demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .demo import replay
from .model import LogisticModel


class Dashboard:
    def __init__(self, root, session_path: Path, model_path: Path | None) -> None:
        import tkinter as tk
        from tkinter import filedialog, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.root = root
        self.root.title("N16 EEG Drowsiness Demo")
        self.root.geometry("900x610")
        self.root.minsize(760, 520)
        self.session_path = session_path
        self.model_path = model_path
        self.rows: list[dict[str, object]] = []
        self.cursor = 0
        self.running = False
        self.history: list[float] = []
        self.threshold = 0.85

        style = ttk.Style()
        style.configure("Title.TLabel", font=("Arial", 18, "bold"))
        style.configure("Value.TLabel", font=("Arial", 14, "bold"))
        style.configure("Status.TLabel", font=("Arial", 24, "bold"))

        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="N16 EEG Drowsiness Detection", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Software demo - dữ liệu mô phỏng không phải kết quả thực nghiệm",
        ).pack(anchor="w", pady=(2, 14))

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x")
        self.path_var = tk.StringVar(value=str(session_path))
        ttk.Entry(toolbar, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(toolbar, text="Chọn CSV", command=self.choose_file).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Bắt đầu", command=self.start).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Dừng", command=self.stop).pack(side="left", padx=3)

        cards = ttk.Frame(outer)
        cards.pack(fill="x", pady=16)
        self.state_var = tk.StringVar(value="CHƯA CHẠY")
        self.risk_var = tk.StringVar(value="0.00")
        self.theta_var = tk.StringVar(value="0.00")
        self.blink_var = tk.StringVar(value="0")
        for column, (label, variable, value_style) in enumerate(
            [
                ("Trạng thái", self.state_var, "Status.TLabel"),
                ("Nguy cơ", self.risk_var, "Value.TLabel"),
                ("Theta Alpha", self.theta_var, "Value.TLabel"),
                ("Số chớp", self.blink_var, "Value.TLabel"),
            ]
        ):
            card = ttk.LabelFrame(cards, text=label, padding=12)
            card.grid(row=0, column=column, padx=4, sticky="nsew")
            cards.columnconfigure(column, weight=1)
            ttk.Label(card, textvariable=variable, style=value_style).pack()

        self.progress = ttk.Progressbar(outer, maximum=100)
        self.progress.pack(fill="x", pady=(0, 12))
        self.canvas = tk.Canvas(outer, height=250, bg="white", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)
        self.footer_var = tk.StringVar(value="Sẵn sàng")
        ttk.Label(outer, textvariable=self.footer_var).pack(anchor="w", pady=(8, 0))

    def choose_file(self) -> None:
        selected = self.filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if selected:
            self.path_var.set(selected)

    def start(self) -> None:
        try:
            frame = pd.read_csv(self.path_var.get())
            model = (
                LogisticModel.load(self.model_path)
                if self.model_path and self.model_path.exists()
                else None
            )
            self.threshold = model.threshold if model else 0.85
            result = replay(
                frame,
                output_log=Path("output/demo/gui_replay_log.csv"),
                speed=0,
                threshold=self.threshold,
                sound=False,
                model=model,
                verbose=False,
            )
            self.rows = result.to_dict("records")
        except Exception as exc:
            self.footer_var.set(f"Lỗi: {exc}")
            return
        self.cursor = 0
        self.history.clear()
        self.running = True
        self.footer_var.set(f"Đang phát {len(self.rows)} cửa sổ")
        self._tick()

    def stop(self) -> None:
        self.running = False
        self.footer_var.set("Đã dừng")

    def _tick(self) -> None:
        if not self.running or self.cursor >= len(self.rows):
            if self.cursor >= len(self.rows) and self.rows:
                self.footer_var.set("Đã phát xong")
            self.running = False
            return
        row = self.rows[self.cursor]
        risk = float(row["risk"])
        state = str(row["state"])
        self.state_var.set(state)
        self.risk_var.set(f"{risk:.2f}")
        self.theta_var.set(f"{float(row['theta_alpha']):.2f}")
        self.blink_var.set(str(row["blink_count"]))
        self.progress["value"] = risk * 100
        self.history.append(risk)
        self._draw_chart()
        if state == "CANH_BAO":
            try:
                import winsound

                winsound.Beep(1100, 120)
            except (ImportError, RuntimeError):
                pass
        self.cursor += 1
        self.root.after(250, self._tick)

    def _draw_chart(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 100)
        height = max(canvas.winfo_height(), 100)
        threshold_y = height - self.threshold * (height - 20) - 10
        canvas.create_line(0, threshold_y, width, threshold_y, fill="#cc4444", dash=(5, 3))
        canvas.create_text(8, threshold_y - 8, text="Ngưỡng", anchor="w", fill="#993333")
        values = self.history[-80:]
        if len(values) < 2:
            return
        points = []
        for index, value in enumerate(values):
            x = index * width / max(len(values) - 1, 1)
            y = height - value * (height - 20) - 10
            points.extend([x, y])
        canvas.create_line(*points, fill="#17365D", width=3, smooth=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the N16 EEG replay dashboard")
    parser.add_argument(
        "--session", type=Path, default=Path("data/synthetic/demo_session.csv")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("output/model/n16_logistic_model.json")
    )
    parser.add_argument("--check", action="store_true", help="Validate inputs without opening GUI")
    args = parser.parse_args()
    if args.check:
        if not args.session.exists():
            raise SystemExit(f"Missing session: {args.session}")
        pd.read_csv(args.session, nrows=5)
        if args.model.exists():
            LogisticModel.load(args.model)
        print("GUI input check passed")
        return
    import tkinter as tk

    root = tk.Tk()
    Dashboard(root, args.session, args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
