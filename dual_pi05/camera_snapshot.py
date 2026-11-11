#!/usr/bin/env python3
"""Read-only, on-demand wrist-camera snapshots. Never starts a robot service."""
import argparse
import base64
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        status = 200
        try:
            if self.path == "/health":
                result = {"ok": True, "role": ARGS.role, "device": ARGS.device, "robot_commands_sent": False}
            elif self.path == "/snapshot":
                with LOCK:
                    device = str(Path(ARGS.device).resolve(strict=True))
                    busy = subprocess.run(["fuser", device], capture_output=True, timeout=2)
                    if busy.returncode == 0:
                        raise RuntimeError("Camera is in use by another process: " + device)
                    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        frame = None
                        for _ in range(4):
                            ok, frame = cap.read()
                            if not ok:
                                raise RuntimeError("Camera frame unavailable")
                        captured = time.time()
                    finally:
                        cap.release()
                    h, w = frame.shape[:2]
                    side = min(h, w)
                    square = frame[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
                    resized = cv2.resize(square, (384, 384), interpolation=cv2.INTER_AREA)
                    ok, jpeg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise RuntimeError("JPEG encode failed")
                    result = {"image": {"shape": [384, 384, 3], "encoding": "jpeg_base64",
                                        "color_space": "rgb", "data": base64.b64encode(jpeg).decode()},
                              "role": ARGS.role, "captured_at": captured, "source_shape": [h, w, 3],
                              "preprocessing": "center square then resize 384; backend crops 224",
                              "training_camera_projection_validated": False}
            else:
                status, result = 404, {"error": "not_found"}
        except Exception as error:
            status, result = 503, {"error": str(error)}
        body = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--role", choices=["left_wrist", "right_wrist"], required=True)
    ARGS = parser.parse_args()
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()
