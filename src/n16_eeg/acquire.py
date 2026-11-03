"""Command-line acquisition of ThinkGear packets to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .thinkgear import ThinkGearParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record NeuroSky ThinkGear raw EEG")
    parser.add_argument("--port", help="Bluetooth COM port, e.g. COM5")
    parser.add_argument(
        "--list-ports", action="store_true", help="List serial ports and exit"
    )
    parser.add_argument("--baud", type=int, default=57600)
    parser.add_argument("--subject")
    parser.add_argument("--session")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration-min", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.duration_min <= 0:
        raise SystemExit("--duration-min must be positive")
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Install pyserial first: python -m pip install pyserial") from exc

    if args.list_ports:
        from serial.tools import list_ports

        ports = list(list_ports.comports())
        if not ports:
            print("No serial ports found")
        for port in ports:
            print(f"{port.device}\t{port.description}")
        return

    missing = [
        name
        for name, value in {
            "--port": args.port,
            "--subject": args.subject,
            "--session": args.session,
            "--output": args.output,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required recording arguments: {', '.join(missing)}")

    assert args.output is not None
    summary_path = args.output.with_suffix(".capture.json")
    if args.output.exists() or summary_path.exists():
        raise SystemExit("Recording exists; choose a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parser = ThinkGearParser()
    started = time.monotonic()
    poor_signal = attention = meditation = ""
    sample_index = 0
    first_raw_time = last_raw_time = None

    with serial.Serial(args.port, args.baud, timeout=0.25) as stream, args.output.open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp_ms",
                "subject_id",
                "session_id",
                "raw_eeg",
                "poor_signal",
                "attention",
                "meditation",
                "kss",
                "event",
                "sample_index", "signal_time_ms", "signal_unit", "dataset_id", "fs_hz",
            ]
        )
        while time.monotonic() - started < args.duration_min * 60:
            for packet in parser.feed(stream.read(4096)):
                poor_signal = packet.poor_signal if packet.poor_signal is not None else poor_signal
                attention = packet.attention if packet.attention is not None else attention
                meditation = packet.meditation if packet.meditation is not None else meditation
                for sample in packet.raw_eeg:
                    received = time.monotonic()
                    if first_raw_time is None:
                        first_raw_time = received
                    last_raw_time = received
                    writer.writerow(
                        [
                            int(time.time_ns() // 1_000_000),
                            args.subject,
                            args.session,
                            sample,
                            poor_signal,
                            attention,
                            meditation,
                            "",
                            "",
                            sample_index, sample_index * 1000.0 / 512.0,
                            "count", "neurosky", 512,
                        ]
                    )
                    sample_index += 1

    # Arrival-rate estimate is not proof of hardware sampling rate or no missing samples.
    span = last_raw_time - first_raw_time if first_raw_time is not None else 0.0
    summary = {
        "subject_id": args.subject, "session_id": args.session, "port": args.port,
        "baud": args.baud, "nominal_fs_hz": 512, "signal_unit": "count",
        "samples": sample_index, "elapsed_s": time.monotonic() - started,
        "arrival_rate_estimate_hz": (sample_index - 1) / span if span > 0 else None,
        "valid_packets": parser.valid_packets, "checksum_errors": parser.checksum_errors,
        "length_errors": parser.length_errors, "payload_errors": parser.payload_errors,
        "time_note": "timestamp_ms is PC receipt; signal_time_ms is nominal index/512, not gap-corrected",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Recorded {sample_index} raw samples; diagnostics: {summary_path}")


if __name__ == "__main__":
    main()
