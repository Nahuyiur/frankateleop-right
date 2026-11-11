#!/usr/bin/env python3
"""Serve a frankateleop V4L2 or RealSense camera as low-latency MJPEG."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple, Type

import cv2
import numpy as np


class FisheyeRectifier:
    """Project an equidistant fisheye image into a human-friendly pinhole view."""

    def __init__(self, source_fov_deg: float, output_fov_deg: float) -> None:
        if not 100.0 <= source_fov_deg <= 220.0:
            raise ValueError("source fisheye FOV must be within [100, 220] degrees")
        if not 45.0 <= output_fov_deg < source_fov_deg:
            raise ValueError("output FOV must be within [45, source FOV) degrees")
        self.source_fov_deg = float(source_fov_deg)
        self.output_fov_deg = float(output_fov_deg)
        self._shape: tuple[int, int] | None = None
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None

    def _build_maps(self, height: int, width: int) -> None:
        # The DAS calibration format identifies this lens as equidistant. Map
        # a virtual pinhole camera ray to source radius r=f*theta, using the
        # advertised horizontal fisheye FOV and preserving the full output size.
        center_x = (width - 1.0) * 0.5
        center_y = (height - 1.0) * 0.5
        output_focal = (width * 0.5) / np.tan(np.deg2rad(self.output_fov_deg) * 0.5)
        source_pixels_per_rad = (width * 0.5) / np.deg2rad(self.source_fov_deg * 0.5)
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        pinhole_x = (grid_x - center_x) / output_focal
        pinhole_y = (grid_y - center_y) / output_focal
        pinhole_radius = np.hypot(pinhole_x, pinhole_y)
        theta = np.arctan(pinhole_radius)
        source_radius = theta * source_pixels_per_rad
        scale = np.divide(
            source_radius,
            pinhole_radius,
            out=np.full_like(source_radius, source_pixels_per_rad / output_focal),
            where=pinhole_radius > 1e-8,
        )
        self._map_x = (center_x + pinhole_x * scale).astype(np.float32)
        self._map_y = (center_y + pinhole_y * scale).astype(np.float32)
        self._shape = (height, width)

    def apply(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        if self._shape != (height, width):
            self._build_maps(height, width)
        assert self._map_x is not None and self._map_y is not None
        return cv2.remap(
            image_rgb,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def _load_capture_class(module_name: str, class_name: str) -> Type:
    """Load one proven frankateleop driver without importing optional cameras."""
    repo = Path(os.environ.get("FRANKA_REPO", "/home/pnp/frankateleop"))
    source = repo / f"franka_capture/cameras/{module_name}.py"
    if not source.is_file():
        raise RuntimeError(f"frankateleop camera driver is missing: {source}")
    spec = importlib.util.spec_from_file_location(f"pico_frankateleop_{module_name}", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frankateleop camera driver: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _first_realsense_serial() -> str:
    import pyrealsense2 as rs

    devices = rs.context().query_devices()
    if len(devices) == 0:
        raise RuntimeError("no Intel RealSense camera is connected")
    return str(devices[0].get_info(rs.camera_info.serial_number))


def _reset_realsense(serial: str) -> None:
    """Reset one enumerated RealSense so a wedged UVC stream can re-enumerate."""
    import pyrealsense2 as rs

    for device in rs.context().query_devices():
        if str(device.get_info(rs.camera_info.serial_number)) == serial:
            device.hardware_reset()
            return
    raise RuntimeError(f"RealSense serial {serial} is not enumerated")


class PicoRealSenseCapture:
    """Color-only RealSense capture using the USB2-compatible YUYV profile."""

    def __init__(self, name: str, serial_number: str, dim: tuple[int, int], fps: int,
                 read_timeout_ms: int) -> None:
        import pyrealsense2 as rs

        self.name = name
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        # This D435if enumerates at USB 2.1; BGR8 can start but never produces
        # frames on that transport, while YUYV is reliably delivered.
        config.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.yuyv, fps)
        self.read_timeout_ms = read_timeout_ms
        self._started = False
        try:
            self._pipeline.start(config)
            self._started = True
        except Exception:
            self.close()
            raise

    def read(self) -> tuple[np.ndarray, None]:
        frames = self._pipeline.wait_for_frames(self.read_timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"Camera {self.name} did not return a color frame")
        packed = np.asarray(color_frame.get_data())
        # pyrealsense2 exposes YUYV differently across builds: either as
        # HxWx2 uint8 bytes or as HxW uint16 packed pixels. OpenCV expects the
        # former, so reinterpret (do not numerically convert) the packed form.
        if packed.ndim == 2 and packed.dtype == np.uint16:
            yuyv = np.ascontiguousarray(packed).view(np.uint8).reshape(
                packed.shape[0], packed.shape[1], 2
            )
        elif packed.ndim == 3 and packed.shape[2] == 2:
            yuyv = np.ascontiguousarray(packed, dtype=np.uint8)
        else:
            raise RuntimeError(
                f"Camera {self.name} returned unexpected YUYV buffer "
                f"shape={packed.shape} dtype={packed.dtype}"
            )
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2), None

    def close(self) -> None:
        if self._started:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._started = False


class LatestJpeg:
    def __init__(
        self,
        name: str,
        backend: str,
        device: str,
        serial: str,
        width: int,
        height: int,
        fps: int,
        rectify_fisheye: bool = False,
        fisheye_fov_deg: float = 160.0,
        output_fov_deg: float = 95.0,
        jpeg_quality: int = 100,
    ) -> None:
        self.name = name
        self.backend = backend
        self.device = device
        self.serial = serial
        self.active_device = device if backend == "v4l2" else serial
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")
        self.rectifier = (
            FisheyeRectifier(fisheye_fov_deg, output_fov_deg)
            if rectify_fisheye
            else None
        )
        self.condition = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.sequence = 0
        self.captured_monotonic = 0.0
        self.status = "starting"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)

    def _capture_loop(self) -> None:
        failures = 0
        last_realsense_reset = 0.0
        while not self.stop_event.is_set():
            camera = None
            try:
                if self.backend == "v4l2":
                    capture_class = _load_capture_class("v4l2", "V4L2Capture")
                    camera = capture_class(
                        name=self.name,
                        device_path=self.device,
                        dim=(self.width, self.height),
                        fps=self.fps,
                        pixel_format="MJPG",
                        read_timeout_ms=2000,
                    )
                    self.active_device = self.device
                else:
                    serial = self.serial if self.serial != "auto" else _first_realsense_serial()
                    camera = PicoRealSenseCapture(
                        name=self.name,
                        serial_number=serial,
                        dim=(self.width, self.height),
                        fps=self.fps,
                        read_timeout_ms=2000,
                    )
                    self.active_device = serial
                with self.condition:
                    self.status = "streaming"
                while not self.stop_event.is_set():
                    try:
                        rgb, _depth = camera.read()
                    except RuntimeError as exc:
                        raise RuntimeError(str(exc)) from exc
                    if self.rectifier is not None:
                        rgb = self.rectifier.apply(rgb)
                    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                    ok, encoded = cv2.imencode(
                        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                    )
                    if not ok:
                        raise RuntimeError("OpenCV JPEG encoding failed")
                    with self.condition:
                        self.jpeg = encoded.tobytes()
                        self.sequence += 1
                        self.captured_monotonic = time.monotonic()
                        self.status = "streaming"
                        self.condition.notify_all()
                    failures = 0
            except Exception as exc:
                failures += 1
                with self.condition:
                    self.status = f"reconnecting: {type(exc).__name__}: {exc}"
                    self.condition.notify_all()
                now = time.monotonic()
                if (
                    self.backend == "realsense"
                    and failures >= 3
                    and now - last_realsense_reset >= 15.0
                ):
                    try:
                        serial = self.active_device
                        if not serial or serial == "auto":
                            serial = _first_realsense_serial()
                        _reset_realsense(serial)
                        last_realsense_reset = now
                        with self.condition:
                            self.status = f"resetting RealSense {serial} after {failures} failures"
                    except Exception as reset_exc:
                        with self.condition:
                            self.status = (
                                f"reconnecting: {type(exc).__name__}: {exc}; "
                                f"reset failed: {type(reset_exc).__name__}: {reset_exc}"
                            )
                    self.stop_event.wait(5.0)
                else:
                    self.stop_event.wait(1.0)
            finally:
                if camera is not None:
                    try:
                        camera.close()
                    except Exception:
                        # A disconnected/wedged RealSense may already have stopped
                        # its pipeline. Cleanup must never kill the retry thread.
                        pass

    def wait_for_frame(self, after_sequence: int, timeout: float) -> Tuple[int, Optional[bytes]]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= after_sequence and not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self.condition.wait(remaining)
            return self.sequence, self.jpeg

    def health(self) -> dict:
        with self.condition:
            age_ms = (
                (time.monotonic() - self.captured_monotonic) * 1000.0
                if self.captured_monotonic
                else None
            )
            return {
                "ok": self.jpeg is not None and age_ms is not None and age_ms < 3000.0,
                "name": self.name,
                "backend": self.backend,
                "device": self.active_device,
                "status": self.status,
                "sequence": self.sequence,
                "frame_age_ms": age_ms,
                "view": "rectilinear" if self.rectifier is not None else "native",
                "source_fov_deg": (
                    self.rectifier.source_fov_deg if self.rectifier is not None else None
                ),
                "output_fov_deg": (
                    self.rectifier.output_fov_deg if self.rectifier is not None else None
                ),
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "jpeg_quality": self.jpeg_quality,
            }


class CameraServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, frame_source: LatestJpeg):
        self.frame_source = frame_source
        super().__init__(address, CameraHandler)


class CameraHandler(BaseHTTPRequestHandler):
    server: CameraServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            payload = json.dumps(self.server.frame_source.health()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/snapshot.jpg":
            _sequence, jpeg = self.server.frame_source.wait_for_frame(-1, 2.0)
            if jpeg is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "camera frame unavailable")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return
        if self.path != "/stream.mjpg":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        sequence = -1
        try:
            while True:
                new_sequence, jpeg = self.server.frame_source.wait_for_frame(sequence, 2.0)
                if new_sequence <= sequence or jpeg is None:
                    continue
                sequence = new_sequence
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Franka camera MJPEG bridge")
    parser.add_argument("--name", required=True)
    parser.add_argument("--backend", choices=("v4l2", "realsense"), default="v4l2")
    parser.add_argument("--device", default="")
    parser.add_argument("--serial", default="auto")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--jpeg-quality", type=int, default=100,
        help="MJPEG JPEG quality 1-100 (100 is highest quality; still lossy)",
    )
    parser.add_argument("--rectify-fisheye", action="store_true")
    parser.add_argument("--fisheye-fov-deg", type=float, default=160.0)
    parser.add_argument("--output-fov-deg", type=float, default=95.0)
    args = parser.parse_args()
    if args.backend == "v4l2" and not args.device:
        parser.error("--device is required for the v4l2 backend")

    source = LatestJpeg(
        args.name,
        args.backend,
        args.device,
        args.serial,
        args.width,
        args.height,
        args.fps,
        args.rectify_fisheye,
        args.fisheye_fov_deg,
        args.output_fov_deg,
        args.jpeg_quality,
    )
    source.start()
    server = CameraServer((args.bind, args.port), source)
    print(
        f"Camera stream ({args.backend}): {args.device or args.serial} -> "
        f"http://{args.bind}:{args.port}/stream.mjpg",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        source.close()


if __name__ == "__main__":
    main()
