"""Pure-Python Discord Rich Presence client over local IPC Named Pipe / Unix socket."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import os
import socket
import struct
import sys
import threading
import time
from typing import Any, Protocol
import uuid

DISCORD_CLIENT_ID = "1575985722450518016"
DISCORD_LARGE_IMAGE_KEY = "muzhen_logo"
DISCORD_GITHUB_BUTTON_LABEL = "View on GitHub"
DISCORD_GITHUB_BUTTON_URL = "https://github.com/wanngxuanji/Muzhen"

_OP_HANDSHAKE = 0
_OP_FRAME = 1
_OP_CLOSE = 2
_OP_PING = 3
_OP_PONG = 4

_MAX_FIELD_LEN = 128
_MAX_PAYLOAD_BYTES = 1024 * 1024
_DEFAULT_PROGRESS_THROTTLE_SEC = 15.0
_DEFAULT_RECONNECT_INTERVAL_SEC = 15.0


class DiscordIpcTransport(Protocol):
    """Minimal byte transport interface for Discord local IPC communication."""

    def write(self, data: bytes) -> None:
        ...

    def read_exact(self, count: int) -> bytes:
        ...

    def close(self) -> None:
        ...


class _WindowsPipeTransport:
    def __init__(self, pipe_path: str) -> None:
        self._handle = open(pipe_path, "r+b", buffering=0)

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        total = 0
        while total < len(view):
            written = self._handle.write(view[total:])
            if not written:
                raise OSError("Discord IPC pipe closed during write")
            total += written

    def read_exact(self, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            part = self._handle.read(count - len(chunks))
            if not part:
                raise OSError("Discord IPC pipe closed during read")
            chunks.extend(part)
        return bytes(chunks)

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass


class _UnixSocketTransport:
    def __init__(self, sock_path: str, timeout: float = 2.0) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(sock_path)
        except BaseException:
            sock.close()
            raise
        self._sock = sock

    def write(self, data: bytes) -> None:
        self._sock.sendall(data)

    def read_exact(self, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            part = self._sock.recv(count - len(chunks))
            if not part:
                raise OSError("Discord IPC socket closed during read")
            chunks.extend(part)
        return bytes(chunks)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def _open_default_transport() -> DiscordIpcTransport | None:
    if sys.platform == "win32":
        for idx in range(10):
            pipe_path = rf"\\.\pipe\discord-ipc-{idx}"
            try:
                return _WindowsPipeTransport(pipe_path)
            except OSError:
                continue
        return None

    base_dirs = [
        d
        for d in (
            os.environ.get("XDG_RUNTIME_DIR"),
            os.environ.get("TMPDIR"),
            os.environ.get("TMP"),
            os.environ.get("TEMP"),
            "/tmp",
        )
        if d
    ]
    for base in dict.fromkeys(base_dirs):
        for idx in range(10):
            sock_path = os.path.join(base, f"discord-ipc-{idx}")
            if not os.path.exists(sock_path):
                continue
            try:
                return _UnixSocketTransport(sock_path)
            except OSError:
                continue
    return None


def encode_ipc_frame(opcode: int, payload: dict[str, Any]) -> bytes:
    """Encode a Discord IPC binary frame with little-endian opcode and byte length."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack("<II", int(opcode), len(body)) + body


def read_ipc_frame(transport: DiscordIpcTransport) -> tuple[int, dict[str, Any]]:
    """Read and decode a single Discord IPC frame from ``transport``."""
    header = transport.read_exact(8)
    opcode, length = struct.unpack("<II", header)
    if length > _MAX_PAYLOAD_BYTES:
        raise ValueError(f"Discord IPC frame too large: {length}")
    raw = transport.read_exact(length) if length > 0 else b"{}"
    parsed = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        return (opcode, {})
    return (opcode, parsed)


def _clamp_field(text: str, max_len: int = _MAX_FIELD_LEN) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) < 2:
        cleaned = (cleaned + "  ")[:2]
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


@dataclass(frozen=True, slots=True)
class PresenceState:
    """Immutable snapshot of Muzhen workbench activity for Discord Rich Presence."""

    mode: str = "idle"
    filename: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    provider: str = "cpu"
    ocr_model: str = "accurate"
    stage: str = ""
    progress_current: int = 0
    progress_total: int = 0
    total_cues: int = 0
    flagged_cues: int = 0
    started_at: int = 0


def build_activity_payload(
    presence: PresenceState,
    *,
    hide_filename: bool = True,
    show_github_button: bool = True,
    app_version: str = "2.1.0",
) -> dict[str, Any]:
    """Build a Discord ``SET_ACTIVITY`` activity object from ``presence`` and privacy settings."""
    mode = presence.mode.strip().lower()
    fname = presence.filename.strip()
    show_name = (not hide_filename) and bool(fname)

    if mode == "preview":
        details = f"Previewing {fname}" if show_name else "Previewing video"
        if presence.width > 0 and presence.height > 0 and presence.fps > 0:
            fps_val = float(presence.fps)
            fps_str = (
                f"{fps_val:.0f}"
                if abs(fps_val - round(fps_val)) < 0.01
                else f"{fps_val:.2f}"
            )
            state_text = f"{presence.width}×{presence.height} · {fps_str} fps"
        else:
            state_text = "Ready to extract subtitles"
    elif mode == "extracting":
        provider_label = "GPU" if presence.provider == "directml" else "CPU"
        model_label = "Precision" if presence.ocr_model == "accurate" else "Turbo"
        if presence.stage == "scan" and presence.progress_total > 0:
            pct = max(
                0,
                min(
                    100,
                    int(round(100.0 * presence.progress_current / presence.progress_total)),
                ),
            )
            progress_str = f"Scanning {pct}%"
        elif presence.stage == "ocr" and presence.progress_total > 0:
            progress_str = f"OCR {presence.progress_current}/{presence.progress_total} cues"
        else:
            progress_str = "Starting extraction"
        if show_name:
            details = f"Extracting {fname}"
            state_text = f"{provider_label} · {model_label} · {progress_str}"
        else:
            details = f"Extracting subtitles ({provider_label} · {model_label})"
            state_text = progress_str
    elif mode == "reviewing":
        details = f"Editing {fname}" if show_name else "Editing subtitles"
        if presence.flagged_cues > 0:
            state_text = f"{presence.total_cues} cues ({presence.flagged_cues} flagged)"
        else:
            state_text = f"{presence.total_cues} cues"
    else:
        details = "Idle"
        state_text = "Ready to extract subtitles"

    activity: dict[str, Any] = {
        "details": _clamp_field(details),
        "state": _clamp_field(state_text),
        "assets": {
            "large_image": DISCORD_LARGE_IMAGE_KEY,
            "large_text": _clamp_field(f"Muzhen v{app_version}"),
        },
    }
    if presence.started_at > 0:
        activity["timestamps"] = {"start": int(presence.started_at)}
    if show_github_button:
        activity["buttons"] = [
            {
                "label": DISCORD_GITHUB_BUTTON_LABEL,
                "url": DISCORD_GITHUB_BUTTON_URL,
            }
        ]
    return activity


class DiscordRpcClient:
    """Non-blocking background Discord Rich Presence client with throttling and auto-reconnect."""

    def __init__(
        self,
        *,
        client_id: str | None = None,
        app_version: str = "2.1.0",
        enabled: bool = True,
        hide_filename: bool = True,
        show_github_button: bool = True,
        progress_throttle_sec: float = _DEFAULT_PROGRESS_THROTTLE_SEC,
        reconnect_interval_sec: float = _DEFAULT_RECONNECT_INTERVAL_SEC,
        transport_factory: Callable[[], DiscordIpcTransport | None] | None = None,
        clock_fn: Callable[[], float] | None = None,
    ) -> None:
        env_client_id = os.environ.get("MUZHEN_DISCORD_CLIENT_ID", "").strip()
        self._client_id = (
            client_id.strip()
            if isinstance(client_id, str) and client_id.strip()
            else (env_client_id or DISCORD_CLIENT_ID)
        )
        self._app_version = app_version
        self._enabled: bool = bool(enabled)
        self._hide_filename: bool = bool(hide_filename)
        self._show_github_button: bool = bool(show_github_button)
        self._progress_throttle_sec = max(0.0, float(progress_throttle_sec))
        self._reconnect_interval_sec = max(0.05, float(reconnect_interval_sec))
        self._transport_factory = (
            transport_factory if transport_factory is not None else _open_default_transport
        )
        self._clock = clock_fn if clock_fn is not None else time.monotonic

        self._cond = threading.Condition()
        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._transport: DiscordIpcTransport | None = None

        self._presence: PresenceState = PresenceState(started_at=int(time.time()))
        self._dirty: bool = True
        self._urgent: bool = True
        self._last_sent_payload: dict[str, Any] | None = None
        self._last_sent_mode: str = ""
        self._last_sent_monotonic: float = -1e9
        self._next_connect_monotonic: float = 0.0

    @property
    def is_running(self) -> bool:
        """Return True while the background IPC worker thread is active."""
        return self._running

    @property
    def enabled(self) -> bool:
        """Return True when Discord Rich Presence publishing is enabled."""
        return self._enabled

    @property
    def hide_filename(self) -> bool:
        """Return True when video file names are hidden from Discord presence."""
        return self._hide_filename

    @property
    def show_github_button(self) -> bool:
        """Return True when the View on GitHub button is included in presence."""
        return self._show_github_button

    @property
    def current_presence(self) -> PresenceState:
        """Return the latest queued presence state snapshot."""
        with self._cond:
            return self._presence

    def current_activity_payload(self) -> dict[str, Any]:
        """Build the activity payload corresponding to the current state and privacy flags."""
        with self._cond:
            return build_activity_payload(
                self._presence,
                hide_filename=self._hide_filename,
                show_github_button=self._show_github_button,
                app_version=self._app_version,
            )

    def start(self) -> None:
        """Start the background Discord IPC worker thread (idempotent)."""
        with self._cond:
            if self._running:
                return
            self._running = True
            self._dirty = True
            self._urgent = True
            thread = threading.Thread(
                target=self._worker_loop,
                name="muzhen-discord-rpc",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        """Clear presence if connected, close IPC transport, and join worker thread."""
        with self._cond:
            was_running = self._running
            self._running = False
            self._cond.notify_all()
            thread = self._thread
            self._thread = None
        if was_running and thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def configure(
        self,
        *,
        enabled: bool | None = None,
        hide_filename: bool | None = None,
        show_github_button: bool | None = None,
    ) -> None:
        """Update RPC enabled/privacy toggles and wake worker immediately."""
        with self._cond:
            changed = False
            if enabled is not None and bool(enabled) != self._enabled:
                self._enabled = bool(enabled)
                changed = True
                if self._enabled:
                    self._next_connect_monotonic = 0.0
            if hide_filename is not None and bool(hide_filename) != self._hide_filename:
                self._hide_filename = bool(hide_filename)
                changed = True
            if (
                show_github_button is not None
                and bool(show_github_button) != self._show_github_button
            ):
                self._show_github_button = bool(show_github_button)
                changed = True
            if changed:
                self._dirty = True
                self._urgent = True
                self._cond.notify_all()

    def update(self, presence: PresenceState, *, urgent: bool | None = None) -> None:
        """Publish a new presence state snapshot to the background worker."""
        with self._cond:
            prev_mode = self._presence.mode
            self._presence = presence
            self._dirty = True
            is_urgent = (presence.mode != prev_mode) if urgent is None else bool(urgent)
            if is_urgent:
                self._urgent = True
            self._cond.notify_all()

    def _disconnect(self, *, clear_activity: bool) -> None:
        transport = self._transport
        self._transport = None
        self._last_sent_payload = None
        self._last_sent_mode = ""
        if transport is None:
            return
        if clear_activity:
            try:
                self._send_set_activity(transport, None)
            except Exception:
                pass
        try:
            transport.write(encode_ipc_frame(_OP_CLOSE, {}))
        except Exception:
            pass
        transport.close()

    def _connect(self) -> bool:
        if not self._client_id:
            return False
        transport = self._transport_factory()
        if transport is None:
            return False
        try:
            transport.write(
                encode_ipc_frame(
                    _OP_HANDSHAKE,
                    {"v": 1, "client_id": self._client_id},
                )
            )
            opcode, reply = read_ipc_frame(transport)
            if opcode == _OP_CLOSE or reply.get("evt") == "ERROR":
                transport.close()
                return False
        except Exception:
            transport.close()
            return False
        self._transport = transport
        self._last_sent_payload = None
        self._last_sent_mode = ""
        return True

    def _send_set_activity(
        self,
        transport: DiscordIpcTransport,
        activity: dict[str, Any] | None,
    ) -> None:
        frame = encode_ipc_frame(
            _OP_FRAME,
            {
                "cmd": "SET_ACTIVITY",
                "args": {
                    "pid": os.getpid(),
                    "activity": activity,
                },
                "nonce": uuid.uuid4().hex,
            },
        )
        transport.write(frame)
        opcode, reply = read_ipc_frame(transport)
        if opcode == _OP_PING:
            transport.write(encode_ipc_frame(_OP_PONG, reply))
            opcode, reply = read_ipc_frame(transport)
        if opcode == _OP_CLOSE:
            raise OSError("Discord closed IPC connection")

    def _worker_loop(self) -> None:
        try:
            while True:
                with self._cond:
                    if not self._running:
                        break
                    enabled = self._enabled
                    presence = self._presence
                    hide_filename = self._hide_filename
                    show_github_button = self._show_github_button
                    dirty = self._dirty
                    urgent = self._urgent

                now = self._clock()
                if not enabled:
                    if self._transport is not None:
                        self._disconnect(clear_activity=True)
                    with self._cond:
                        self._dirty = False
                        self._urgent = False
                        if self._running and not self._enabled:
                            self._cond.wait(timeout=0.5)
                    continue

                if self._transport is None:
                    if now < self._next_connect_monotonic:
                        wait_sec = min(0.5, max(0.01, self._next_connect_monotonic - now))
                        with self._cond:
                            if self._running and not self._urgent:
                                self._cond.wait(timeout=wait_sec)
                        continue
                    if not self._connect():
                        self._next_connect_monotonic = self._clock() + self._reconnect_interval_sec
                        with self._cond:
                            self._urgent = False
                            if self._running:
                                self._cond.wait(timeout=min(0.5, self._reconnect_interval_sec))
                        continue
                    dirty = True
                    urgent = True

                payload = build_activity_payload(
                    presence,
                    hide_filename=hide_filename,
                    show_github_button=show_github_button,
                    app_version=self._app_version,
                )

                should_send = False
                wait_timeout = 0.5
                if dirty and payload != self._last_sent_payload:
                    elapsed = now - self._last_sent_monotonic
                    same_progress_mode = (
                        presence.mode == "extracting"
                        and self._last_sent_mode == "extracting"
                        and not urgent
                    )
                    if (not same_progress_mode) or (elapsed >= self._progress_throttle_sec):
                        should_send = True
                    else:
                        wait_timeout = min(
                            0.5,
                            max(0.01, self._progress_throttle_sec - elapsed),
                        )

                if should_send and self._transport is not None:
                    try:
                        self._send_set_activity(self._transport, payload)
                    except Exception:
                        self._disconnect(clear_activity=False)
                        self._next_connect_monotonic = self._clock() + self._reconnect_interval_sec
                    else:
                        self._last_sent_payload = payload
                        self._last_sent_mode = presence.mode
                        self._last_sent_monotonic = self._clock()
                        with self._cond:
                            if (
                                self._presence == presence
                                and self._hide_filename == hide_filename
                                and self._show_github_button == show_github_button
                            ):
                                self._dirty = False
                                self._urgent = False
                    continue

                with self._cond:
                    if self._running and (not self._dirty or payload == self._last_sent_payload or wait_timeout > 0):
                        if payload == self._last_sent_payload and self._presence == presence:
                            self._dirty = False
                            self._urgent = False
                        self._cond.wait(timeout=wait_timeout)
        finally:
            self._disconnect(clear_activity=True)
