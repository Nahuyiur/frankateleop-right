"""Low-latency V4L2 RGB capture for non-RealSense USB cameras."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class V4L2Capture:
    """Read the latest RGB frame from a V4L2 device without queueing stale data."""

    def __init__(
        self,
        name: str,
        device_path: str,
        dim: Tuple[int, int] = (640, 480),
        fps: int = 30,
        pixel_format: str = "MJPG",
        flip: bool = False,
        read_timeout_ms: int = 3000,
    ) -> None:
        path = Path(device_path)
        if not path.exists():
            raise RuntimeError(f"V4L2 camera device does not exist: {device_path}")
        if len(pixel_format) != 4:
            raise ValueError("V4L2 pixel_format must contain exactly four characters")

        self.name = str(name)
        self.device_path = str(path)
        self.dim = (int(dim[0]), int(dim[1]))
        self.fps = int(fps)
        self.pixel_format = str(pixel_format)
        self.flip = bool(flip)
        self.read_timeout_ms = int(read_timeout_ms)
        self.depth_enabled = False
        self.depth_scale = None
        self.intrinsics = None
        self.intrinsics_dict = None

        self._capture = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Failed to open V4L2 camera {self.device_path}")
        self._capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self.pixel_format),
        )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.dim[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.dim[1])
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._condition = threading.Condition()
        self._latest_bgr: Optional[np.ndarray] = None
        self._sequence = 0
        self._last_delivered_sequence = 0
        self._last_error = ""
        self._closed = False
        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"v4l2-{self.name}",
            daemon=True,
        )
        self._reader.start()

    def _reader_loop(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
            ok, frame = self._capture.read()
            if ok and isinstance(frame, np.ndarray) and frame.ndim == 3:
                with self._condition:
                    self._latest_bgr = frame
                    self._sequence += 1
                    self._last_error = ""
                    self._condition.notify_all()
                continue
            with self._condition:
                self._last_error = "V4L2 read returned no frame"
                self._condition.notify_all()
            time.sleep(0.02)

    def read(self):
        deadline = time.monotonic() + max(self.read_timeout_ms, 1) / 1000.0
        with self._condition:
            while self._sequence <= self._last_delivered_sequence:
                if self._closed:
                    raise RuntimeError(f"V4L2 camera {self.name} is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = f": {self._last_error}" if self._last_error else ""
                    raise RuntimeError(
                        f"V4L2 camera {self.name} did not return a frame within "
                        f"{self.read_timeout_ms} ms{detail}"
                    )
                self._condition.wait(remaining)
            frame_bgr = self._latest_bgr
            self._last_delivered_sequence = self._sequence
        if frame_bgr is None:
            raise RuntimeError(f"V4L2 camera {self.name} returned an empty frame")
        image_rgb = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)
        if self.flip:
            image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_180)
        return image_rgb, None

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._capture.release()
        self._reader.join(timeout=1.0)

    def metadata(self):
        return {
            "name": self.name,
            "backend": "v4l2",
            "device_path": self.device_path,
            "serial_number": None,
            "dim": list(self.dim),
            "fps": self.fps,
            "pixel_format": self.pixel_format,
            "depth": False,
            "align_depth": False,
            "flip": self.flip,
            "read_timeout_ms": self.read_timeout_ms,
            "depth_scale": None,
            "intrinsics": None,
        }

    def __repr__(self) -> str:
        return (
            f"V4L2Capture(name={self.name!r}, device_path={self.device_path!r}, "
            f"dim={self.dim!r}, fps={self.fps})"
        )
