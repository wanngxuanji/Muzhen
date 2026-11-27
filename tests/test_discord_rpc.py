from __future__ import annotations

import json
import struct
import threading
import time
from typing import Any

from muzhen.core.discord_rpc import (
    DISCORD_CLIENT_ID,
    DISCORD_GITHUB_BUTTON_LABEL,
    DISCORD_GITHUB_BUTTON_URL,
    DISCORD_LARGE_IMAGE_KEY,
    DiscordRpcClient,
    PresenceState,
    build_activity_payload,
    encode_ipc_frame,
)


class _FakeIpcTransport:
    def __init__(self, *, fail_after_writes: int | None = None) -> None:
        self._lock = threading.Lock()
        self.frames: list[tuple[int, dict[str, Any]]] = []
        self.closed = False
        self._fail_after_writes = fail_after_writes
        self._pending_reply: bytes = b""
        self._read_pos = 0

    def write(self, data: bytes) -> None:
        with self._lock:
            if self.closed:
                raise OSError("transport closed")
            if self._fail_after_writes is not None and len(self.frames) >= self._fail_after_writes:
                raise OSError("simulated broken pipe")
            opcode, length = struct.unpack("<II", data[:8])
            payload = json.loads(data[8 : 8 + length].decode("utf-8")) if length else {}
            self.frames.append((opcode, payload))
            if opcode == 0:
                reply = encode_ipc_frame(1, {"cmd": "DISPATCH", "evt": "READY", "data": {}})
            elif opcode == 1:
                reply = encode_ipc_frame(
                    1,
                    {
                        "cmd": payload.get("cmd", ""),
                        "nonce": payload.get("nonce", ""),
                        "evt": None,
                        "data": {},
                    },
                )
            else:
                reply = b""
            self._pending_reply = reply
            self._read_pos = 0

    def read_exact(self, count: int) -> bytes:
        with self._lock:
            if self.closed:
                raise OSError("transport closed")
            end = self._read_pos + count
            if end > len(self._pending_reply):
                raise OSError("unexpected EOF")
            chunk = self._pending_reply[self._read_pos : end]
            self._read_pos = end
            return chunk

    def close(self) -> None:
        with self._lock:
            self.closed = True


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_build_activity_payload_covers_modes_privacy_and_github_button() -> None:
    idle = build_activity_payload(
        PresenceState(mode="idle", started_at=1700000000),
        hide_filename=True,
        show_github_button=True,
        app_version="2.1.0",
    )
    assert idle["details"] == "Idle"
    assert idle["state"] == "Ready to extract subtitles"
    assert idle["assets"]["large_image"] == DISCORD_LARGE_IMAGE_KEY
    assert idle["assets"]["large_text"] == "Muzhen v2.1.0"
    assert idle["timestamps"] == {"start": 1700000000}
    assert idle["buttons"] == [
        {"label": DISCORD_GITHUB_BUTTON_LABEL, "url": DISCORD_GITHUB_BUTTON_URL}
    ]

    preview_hidden = build_activity_payload(
        PresenceState(
            mode="preview",
            filename="Secret_EP01.mp4",
            width=1920,
            height=1080,
            fps=25.0,
            started_at=1700000000,
        ),
        hide_filename=True,
        show_github_button=False,
    )
    assert preview_hidden["details"] == "Previewing video"
    assert preview_hidden["state"] == "1920×1080 · 25 fps"
    assert "Secret_EP01" not in json.dumps(preview_hidden)
    assert "buttons" not in preview_hidden

    preview_shown = build_activity_payload(
        PresenceState(
            mode="preview",
            filename="Episode_01.mp4",
            width=1920,
            height=1080,
            fps=23.976,
            started_at=1700000000,
        ),
        hide_filename=False,
        show_github_button=True,
    )
    assert preview_shown["details"] == "Previewing Episode_01.mp4"
    assert preview_shown["state"] == "1920×1080 · 23.98 fps"

    ext_scan = build_activity_payload(
        PresenceState(
            mode="extracting",
            filename="Secret_EP01.mp4",
            provider="directml",
            ocr_model="accurate",
            stage="scan",
            progress_current=42,
            progress_total=100,
            started_at=1700000100,
        ),
        hide_filename=True,
        show_github_button=True,
    )
    assert ext_scan["details"] == "Extracting subtitles (GPU · Precision)"
    assert ext_scan["state"] == "Scanning 42%"
    assert ext_scan["timestamps"] == {"start": 1700000100}

    ext_ocr_named = build_activity_payload(
        PresenceState(
            mode="extracting",
            filename="Episode_01.mp4",
            provider="cpu",
            ocr_model="fast",
            stage="ocr",
            progress_current=18,
            progress_total=64,
            started_at=1700000100,
        ),
        hide_filename=False,
        show_github_button=True,
    )
    assert ext_ocr_named["details"] == "Extracting Episode_01.mp4"
    assert ext_ocr_named["state"] == "CPU · Turbo · OCR 18/64 cues"

    review_flagged = build_activity_payload(
        PresenceState(
            mode="reviewing",
            filename="Episode_01.mp4",
            total_cues=142,
            flagged_cues=3,
            started_at=1700000000,
        ),
        hide_filename=True,
        show_github_button=True,
    )
    assert review_flagged["details"] == "Editing subtitles"
    assert review_flagged["state"] == "142 cues (3 flagged)"

    review_clean_named = build_activity_payload(
        PresenceState(
            mode="reviewing",
            filename="A" * 200 + ".mp4",
            total_cues=22,
            flagged_cues=0,
            started_at=1700000000,
        ),
        hide_filename=False,
        show_github_button=False,
    )
    assert len(review_clean_named["details"]) <= 128
    assert review_clean_named["details"].endswith("…")
    assert review_clean_named["state"] == "22 cues"


def test_discord_rpc_client_handshake_throttling_toggles_and_reconnect() -> None:
    transports: list[_FakeIpcTransport] = []
    allow_connect = {"value": False}
    fake_time = {"now": 100.0}

    def factory() -> _FakeIpcTransport | None:
        if not allow_connect["value"]:
            return None
        tr = _FakeIpcTransport()
        transports.append(tr)
        return tr

    client = DiscordRpcClient(
        app_version="2.1.0",
        enabled=True,
        hide_filename=True,
        show_github_button=True,
        progress_throttle_sec=15.0,
        reconnect_interval_sec=0.05,
        transport_factory=factory,
        clock_fn=lambda: fake_time["now"],
    )
    client.start()
    time.sleep(0.05)
    assert transports == []

    allow_connect["value"] = True
    fake_time["now"] += 1.0
    _wait_until(lambda: len(transports) == 1 and len(transports[0].frames) >= 2)

    tr0 = transports[0]
    assert tr0.frames[0] == (0, {"v": 1, "client_id": DISCORD_CLIENT_ID})
    assert tr0.frames[1][0] == 1
    assert tr0.frames[1][1]["cmd"] == "SET_ACTIVITY"
    assert tr0.frames[1][1]["args"]["activity"]["details"] == "Idle"

    client.update(
        PresenceState(
            mode="extracting",
            filename="EP01.mp4",
            provider="directml",
            ocr_model="accurate",
            stage="scan",
            progress_current=10,
            progress_total=100,
            started_at=1700000000,
        ),
        urgent=True,
    )
    _wait_until(
        lambda: tr0.frames[-1][1].get("args", {}).get("activity", {}).get("state")
        == "Scanning 10%"
    )
    count_after_10 = len(tr0.frames)

    client.update(
        PresenceState(
            mode="extracting",
            filename="EP01.mp4",
            provider="directml",
            ocr_model="accurate",
            stage="scan",
            progress_current=20,
            progress_total=100,
            started_at=1700000000,
        ),
        urgent=False,
    )
    time.sleep(0.06)
    assert len(tr0.frames) == count_after_10

    fake_time["now"] += 16.0
    _wait_until(
        lambda: tr0.frames[-1][1].get("args", {}).get("activity", {}).get("state")
        == "Scanning 20%"
    )

    client.configure(hide_filename=False, show_github_button=False)
    _wait_until(
        lambda: tr0.frames[-1][1].get("args", {}).get("activity", {}).get("details")
        == "Extracting EP01.mp4"
    )
    assert "buttons" not in tr0.frames[-1][1]["args"]["activity"]

    client.configure(enabled=False)
    _wait_until(lambda: tr0.closed is True)
    clear_frames = [
        payload
        for op, payload in tr0.frames
        if op == 1 and payload.get("cmd") == "SET_ACTIVITY" and payload["args"]["activity"] is None
    ]
    assert len(clear_frames) == 1

    client.configure(enabled=True)
    _wait_until(lambda: len(transports) == 2 and len(transports[1].frames) >= 2)
    tr1 = transports[1]
    assert tr1.frames[1][1]["args"]["activity"]["details"] == "Extracting EP01.mp4"

    client.stop()
    assert client.is_running is False
    assert tr1.closed is True
    stop_clears = [
        payload
        for op, payload in tr1.frames
        if op == 1 and payload.get("cmd") == "SET_ACTIVITY" and payload["args"]["activity"] is None
    ]
    assert len(stop_clears) == 1
