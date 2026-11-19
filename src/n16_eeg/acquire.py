"""Command-line acquisition of ThinkGear packets to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from threading import Thread

from .continuity import RAW_GAP_S, SERIAL_POLL_S, RawStreamWatchdog
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
    parser.add_argument("--sync-markers", action="store_true",
                        help="Windows console: M logs a marker and requests a tone, Q ends capture")
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
    if args.sync_markers:
        if sys.platform != 'win32' or not sys.stdin.isatty():
            raise SystemExit('Run --sync-markers in an interactive Windows PowerShell window')
        import msvcrt
        import winsound
    summary_path = args.output.with_suffix(".capture.json")
    if args.output.exists() or summary_path.exists():
        raise SystemExit("Recording exists; choose a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parser = ThinkGearParser()
    poor_signal = attention = meditation = ""
    sample_index = 0
    first_raw_time = last_raw_time = None
    markers = []
    gaps = []
    pending_marker = None
    tone = None
    stop_reason = "duration_limit"
    serial_error = None

    with serial.Serial(args.port, args.baud, timeout=SERIAL_POLL_S) as stream, args.output.open(
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
                "sample_index", "signal_time_ms", "signal_unit", "dataset_id", "fs_hz", "raw_segment",
            ]
        )
        if args.sync_markers:
            print('Wait for EEG samples, then press M for each sync mark or Q to finish.', flush=True)
        started = time.monotonic()
        watchdog = RawStreamWatchdog(started)
        while time.monotonic() - started < args.duration_min * 60:
            try:
                buffered_bytes = stream.in_waiting
            except OSError as exc:
                serial_error = exc
                break
            if args.sync_markers and msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                if key == 'q':
                    stop_reason = "operator_stop"
                    break
                if key == 'm':
                    if (sample_index == 0 or watchdog.stalled or pending_marker is not None
                            or (tone is not None and tone.is_alive())):
                        print('No marker created. Wait for raw EEG and for the previous marker to finish.', flush=True)
                    else:
                        pending_marker = dict(event=f'SYNC_{len(markers)+1:02d}',
                                              requested_timestamp_ms=time.time_ns() // 1_000_000,
                                              requested_elapsed_ms=(time.monotonic() - started) * 1000,
                                              buffered_bytes=buffered_bytes)
                        markers.append(pending_marker)
                        tone = Thread(target=winsound.Beep, args=(1800, 120), daemon=True)
                        tone.start()
            try:
                chunk = stream.read(max(1, min(buffered_bytes, 4096)))
            except OSError as exc:
                serial_error = exc
                break
            received = time.monotonic()
            receipt_ms = time.time_ns() // 1_000_000
            packets = parser.feed(chunk)
            has_raw = any(packet.raw_eeg for packet in packets)
            last_raw = watchdog.last_raw
            was_stalled = watchdog.stalled
            new_gap = watchdog.check(received, has_raw)
            if new_gap:
                gaps.append({"sample_index_after_gap": sample_index,
                             "last_raw_elapsed_s": last_raw - started,
                             "recovered_elapsed_s": None})
                poor_signal = attention = meditation = ""
                print(f'Raw EEG interrupted at sample {sample_index}. Check the headset and connection.', flush=True)
            if has_raw and (was_stalled or new_gap):
                gaps[-1]['recovered_elapsed_s'] = received - started
                print(f'Raw EEG resumed at sample {sample_index}.', flush=True)
            if watchdog.stalled:
                continue
            for packet in packets:
                poor_signal = packet.poor_signal if packet.poor_signal is not None else poor_signal
                attention = packet.attention if packet.attention is not None else attention
                meditation = packet.meditation if packet.meditation is not None else meditation
                for sample in packet.raw_eeg:
                    event = ''
                    if pending_marker is not None:
                        event = pending_marker['event']
                        pending_marker.update(next_sample_index=sample_index,
                                              next_sample_receipt_ms=receipt_ms,
                                              next_sample_elapsed_ms=(received - started) * 1000,
                                              next_signal_time_ms=sample_index*1000/512)
                        print(f'{event} attached to received sample {sample_index}', flush=True)
                        pending_marker = None
                    if first_raw_time is None:
                        first_raw_time = received
                    last_raw_time = received
                    writer.writerow(
                        [
                            receipt_ms,
                            args.subject,
                            args.session,
                            sample,
                            poor_signal,
                            attention,
                            meditation,
                            "",
                            event,
                            sample_index, sample_index * 1000.0 / 512.0,
                            "count", "neurosky", 512, len(gaps),
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
        "raw_gap_threshold_s": RAW_GAP_S, "raw_receipt_gaps": gaps,
        "stop_reason": "serial_error" if serial_error is not None else stop_reason,
        "time_note": "timestamp_ms is PC receipt; signal_time_ms is nominal index/512, not gap-corrected",
    }
    if args.sync_markers:
        summary['sync_markers'] = markers
        summary['marker_note'] = 'Software marker at the first received raw sample after requesting a PC tone; sound onset and Bluetooth buffering are not hardware calibrated.'
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Recorded {sample_index} raw samples; diagnostics: {summary_path}")
    if serial_error is not None:
        raise SystemExit(f"Serial read failed on {args.port}: {serial_error}")


if __name__ == "__main__":
    main()
