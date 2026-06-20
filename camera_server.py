#!/usr/bin/env python3
"""
Simple Raspberry Pi camera server that exposes an MJPEG stream at /video.
Run this file on the Pi, not on the Mac.
"""

from __future__ import annotations

import os
import time

import cv2
from flask import Flask, Response
from picamera2 import Picamera2


app = Flask(__name__)
COLOR_MODE = os.getenv("PI_COLOR_MODE", "bgr_to_rgb").lower()

picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()
time.sleep(1)


def prepare_frame(frame):
    if COLOR_MODE == "raw":
        return frame
    if COLOR_MODE == "rgb_to_bgr":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if COLOR_MODE == "bgr_to_rgb":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(
        "Unsupported PI_COLOR_MODE. Use raw, rgb_to_bgr, or bgr_to_rgb."
    )


def generate_frames():
    while True:
        frame = picam2.capture_array()
        frame = prepare_frame(frame)
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return f"Pi camera server running. Open /video. color_mode={COLOR_MODE}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



