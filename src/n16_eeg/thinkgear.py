"""Incremental parser for the NeuroSky ThinkGear serial stream.

Protocol reference:
https://developer.neurosky.com/docs/doku.php?id=thinkgear_communications_protocol
"""

from __future__ import annotations

from dataclasses import dataclass


SYNC = 0xAA
MAX_PAYLOAD = 169


@dataclass(frozen=True)
class ThinkGearPacket:
    raw_eeg: tuple[int, ...] = ()
    poor_signal: int | None = None
    attention: int | None = None
    meditation: int | None = None
    blink_strength: int | None = None


class ThinkGearParser:
    """Parse arbitrary byte chunks and return checksum-validated packets."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[ThinkGearPacket]:
        self._buffer.extend(chunk)
        packets: list[ThinkGearPacket] = []

        while True:
            start = self._find_sync()
            if start < 0:
                self._buffer[:] = self._buffer[-1:] if self._buffer[-1:] == b"\xAA" else b""
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 3:
                break

            payload_len = self._buffer[2]
            if payload_len > MAX_PAYLOAD:
                del self._buffer[0]
                continue
            total_len = 2 + 1 + payload_len + 1
            if len(self._buffer) < total_len:
                break

            payload = bytes(self._buffer[3 : 3 + payload_len])
            checksum = self._buffer[3 + payload_len]
            del self._buffer[:total_len]
            expected = (~(sum(payload) & 0xFF)) & 0xFF
            if checksum != expected:
                continue
            packets.append(self._parse_payload(payload))

        return packets

    def _find_sync(self) -> int:
        for index in range(max(0, len(self._buffer) - 1)):
            if self._buffer[index] == SYNC and self._buffer[index + 1] == SYNC:
                return index
        return -1

    @staticmethod
    def _parse_payload(payload: bytes) -> ThinkGearPacket:
        raw: list[int] = []
        poor_signal = attention = meditation = blink_strength = None
        index = 0

        while index < len(payload):
            while index < len(payload) and payload[index] == 0x55:
                index += 1
            if index >= len(payload):
                break

            code = payload[index]
            index += 1
            if code < 0x80:
                if index >= len(payload):
                    break
                value = payload[index]
                index += 1
                if code == 0x02:
                    poor_signal = value
                elif code == 0x04:
                    attention = value
                elif code == 0x05:
                    meditation = value
                elif code == 0x16:
                    blink_strength = value
                continue

            if index >= len(payload):
                break
            length = payload[index]
            index += 1
            value = payload[index : index + length]
            index += length
            if len(value) != length:
                break
            if code == 0x80 and length == 2:
                raw.append(int.from_bytes(value, byteorder="big", signed=True))

        return ThinkGearPacket(tuple(raw), poor_signal, attention, meditation, blink_strength)
