#!/usr/bin/env python3
"""
Watch a network camera stream, publish it to Overshoot, and save a photo
whenever a plate of food newly enters the frame.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import requests
from dotenv import load_dotenv


OVERSHOOT_BASE_URL = "https://api.overshoot.ai/v1"
DEFAULT_PROMPT = (
    "You are monitoring a fixed security-style camera. Decide whether a real "
    "plate of food is visible in the latest frame right now. Count partial "
    "plates as visible if food is clearly on them. Ignore empty plates, bowls, "
    "cups, printed pictures of food, and food advertisements. Use the short "
    "video clip only for motion context. Return JSON only with this exact "
    'schema: {"plate_of_food_visible": true, "confidence": 0.0, "reason": "short phrase"}'
)


class OvershootError(RuntimeError):
    """Raised when the Overshoot API returns an unexpected response."""


def parse_json_fragment(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in model response: {text!r}")
    return json.loads(text[start : end + 1])


@dataclass
class DetectionResult:
    plate_of_food_visible: bool
    confidence: float
    reason: str
    raw_response: str


class OvershootClient:
    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        self.api_key = api_key
        self.requested_model = model
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.ok:
            return
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except Exception:
            pass
        raise OvershootError(f"Overshoot API error {response.status_code}: {detail}")

    def list_ready_models(self) -> list[str]:
        response = self.session.get(f"{OVERSHOOT_BASE_URL}/models", timeout=20)
        self._raise_for_status(response)
        payload = response.json()
        return [
            item["id"]
            for item in payload.get("data", [])
            if item.get("status") == "ready"
        ]

    def choose_model(self) -> str:
        models = self.list_ready_models()
        if not models:
            raise OvershootError("Overshoot returned no ready models.")
        if self.requested_model:
            if self.requested_model not in models:
                raise OvershootError(
                    f"Requested model {self.requested_model!r} is not ready. "
                    f"Ready models: {', '.join(models)}"
                )
            return self.requested_model

        preferred = [
            "Qwen/Qwen3.5-9B",
            "google/gemma-4-E4B-it",
        ]
        for candidate in preferred:
            if candidate in models:
                return candidate
        return models[0]

    def create_stream(self) -> Dict[str, Any]:
        response = self.session.post(f"{OVERSHOOT_BASE_URL}/streams", timeout=20)
        self._raise_for_status(response)
        return response.json()

    def keepalive(self, stream_id: str) -> Dict[str, Any]:
        response = self.session.post(
            f"{OVERSHOOT_BASE_URL}/streams/{stream_id}/keepalive",
            timeout=20,
        )
        self._raise_for_status(response)
        return response.json()

    def get_stream(self, stream_id: str) -> Dict[str, Any]:
        response = self.session.get(
            f"{OVERSHOOT_BASE_URL}/streams/{stream_id}",
            timeout=20,
        )
        self._raise_for_status(response)
        return response.json()

    def delete_stream(self, stream_id: str) -> None:
        response = self.session.delete(
            f"{OVERSHOOT_BASE_URL}/streams/{stream_id}",
            timeout=20,
        )
        self._raise_for_status(response)

    def detect_plate_of_food(
        self,
        *,
        stream_id: str,
        model: str,
        prompt: str,
        window_ms: int,
        max_completion_tokens: int,
    ) -> DetectionResult:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": (
                                    f"ovs://streams/{stream_id}"
                                    f"?start_offset_ms=-{window_ms}&max_fps=1"
                                )
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"ovs://streams/{stream_id}?frame_index=-1"
                            },
                        },
                    ],
                }
            ],
            "max_completion_tokens": max_completion_tokens,
        }
        response = self.session.post(
            f"{OVERSHOOT_BASE_URL}/chat/completions",
            json=payload,
            timeout=45,
        )
        self._raise_for_status(response)
        content = response.json()["choices"][0]["message"]["content"]
        data = parse_json_fragment(content)
        return DetectionResult(
            plate_of_food_visible=bool(data.get("plate_of_food_visible")),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            reason=str(data.get("reason", "")).strip(),
            raw_response=content,
        )


class CameraFeed:
    def __init__(self, stream_url: str, max_width: int = 960) -> None:
        self.stream_url = stream_url
        self.max_width = max_width
        self._lock = threading.Lock()
        self._latest_frame: Optional[Any] = None
        self._latest_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _resize(self, frame: Any) -> Any:
        height, width = frame.shape[:2]
        if width <= self.max_width:
            return frame
        scale = self.max_width / width
        return cv2.resize(frame, (self.max_width, int(height * scale)))

    def _run(self) -> None:
        while not self._stop.is_set():
            capture = cv2.VideoCapture(self.stream_url)
            if not capture.isOpened():
                print(f"Could not open stream {self.stream_url}; retrying in 2s.")
                time.sleep(2)
                continue

            try:
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        print("Camera read failed; reconnecting in 1s.")
                        time.sleep(1)
                        break
                    frame = self._resize(frame)
                    with self._lock:
                        self._latest_frame = frame.copy()
                        self._latest_at = time.time()
            finally:
                capture.release()

    def wait_for_frame(self, timeout: float = 15.0) -> Any:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.get_latest_frame()
            if frame is not None:
                return frame
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for first camera frame.")

    def get_latest_frame(self) -> Optional[Any]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def latest_age_seconds(self) -> float:
        with self._lock:
            if not self._latest_at:
                return float("inf")
            return time.time() - self._latest_at


class LiveKitPublisher:
    def __init__(self) -> None:
        self.rtc = None
        self.room = None
        self.source = None

    async def connect(self, url: str, token: str, width: int, height: int) -> None:
        try:
            from livekit import rtc
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency `livekit`. Install it with `pip install livekit`."
            ) from exc

        self.rtc = rtc
        self.room = rtc.Room()
        await self.room.connect(url, token)

        self.source = rtc.VideoSource(width, height)
        track = rtc.LocalVideoTrack.create_video_track("camera", self.source)

        publish_options = None
        try:
            publish_options = rtc.TrackPublishOptions()
            publish_options.source = rtc.TrackSource.SOURCE_CAMERA
        except Exception:
            publish_options = None

        if publish_options is None:
            await self.room.local_participant.publish_track(track)
        else:
            await self.room.local_participant.publish_track(track, publish_options)

    def push_frame(self, frame_bgr: Any) -> None:
        if self.source is None or self.rtc is None:
            raise RuntimeError("LiveKit publisher is not connected.")

        bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        height, width = bgra.shape[:2]
        payload = bgra.tobytes()

        constructors = [
            lambda: self.rtc.VideoFrame(
                width, height, self.rtc.VideoBufferType.BGRA, payload
            ),
            lambda: self.rtc.VideoFrame(
                width=width,
                height=height,
                type=self.rtc.VideoBufferType.BGRA,
                data=payload,
            ),
        ]

        last_error: Optional[Exception] = None
        for build in constructors:
            try:
                frame = build()
                self.source.capture_frame(frame)
                return
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"Unable to build LiveKit frame: {last_error}")

    async def close(self) -> None:
        if self.room is not None:
            await self.room.disconnect()


async def keep_stream_alive(
    client: OvershootClient,
    stream_id: str,
    stop_event: asyncio.Event,
    interval_seconds: float,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            try:
                lease = client.keepalive(stream_id)
                expires = lease.get("expires_at_ms")
                if expires:
                    print(
                        "Sent keepalive; stream expires at "
                        f"{datetime.fromtimestamp(expires / 1000).isoformat()}."
                    )
            except Exception as exc:
                print(f"Keepalive failed: {exc}")


async def publish_camera_frames(
    feed: CameraFeed,
    publisher: LiveKitPublisher,
    stop_event: asyncio.Event,
    fps: float,
) -> None:
    frame_interval = 1.0 / fps
    while not stop_event.is_set():
        frame = feed.get_latest_frame()
        if frame is not None:
            publisher.push_frame(frame)
        await asyncio.sleep(frame_interval)


def save_snapshot(frame: Any, output_dir: Path, detection: DetectionResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"plate_{timestamp}.jpg"
    meta_path = output_dir / f"plate_{timestamp}.json"

    cv2.imwrite(str(image_path), frame)
    meta_path.write_text(
        json.dumps(
            {
                "captured_at": datetime.now().isoformat(),
                "confidence": detection.confidence,
                "reason": detection.reason,
                "raw_response": detection.raw_response,
            },
            indent=2,
        )
    )
    return image_path


async def detection_loop(
    *,
    client: OvershootClient,
    stream_id: str,
    model: str,
    feed: CameraFeed,
    output_dir: Path,
    stop_event: asyncio.Event,
    prompt: str,
    detection_interval: float,
    window_ms: int,
    min_confidence: float,
    cooldown_seconds: float,
    enter_confirmations: int,
    exit_confirmations: int,
    max_completion_tokens: int,
) -> None:
    last_capture_at = 0.0
    plate_present = False
    visible_streak = 0
    absent_streak = 0

    while not stop_event.is_set():
        if feed.latest_age_seconds() > 5:
            print("Camera feed looks stale; skipping detection.")
            await asyncio.sleep(detection_interval)
            continue

        try:
            stream_state = client.get_stream(stream_id)
            last_frame_index = stream_state.get("last_frame_index")
            if last_frame_index is None:
                print("Overshoot stream has not received frames yet.")
                await asyncio.sleep(detection_interval)
                continue

            detection = client.detect_plate_of_food(
                stream_id=stream_id,
                model=model,
                prompt=prompt,
                window_ms=window_ms,
                max_completion_tokens=max_completion_tokens,
            )
        except Exception as exc:
            print(f"Detection request failed: {exc}")
            await asyncio.sleep(detection_interval)
            continue

        visible_now = (
            detection.plate_of_food_visible and detection.confidence >= min_confidence
        )
        print(
            "Detection:",
            json.dumps(
                {
                    "plate_of_food_visible": detection.plate_of_food_visible,
                    "confidence": round(detection.confidence, 3),
                    "reason": detection.reason,
                }
            ),
        )

        if visible_now:
            visible_streak += 1
            absent_streak = 0
        else:
            visible_streak = 0
            absent_streak += 1

        if (
            not plate_present
            and visible_streak >= enter_confirmations
            and (time.time() - last_capture_at) >= cooldown_seconds
        ):
            frame = feed.get_latest_frame()
            if frame is not None:
                image_path = save_snapshot(frame, output_dir, detection)
                plate_present = True
                last_capture_at = time.time()
                print(f"Saved snapshot to {image_path}")
        elif plate_present and absent_streak >= exit_confirmations:
            plate_present = False
            print("Frame is clear again; ready for the next plate.")

        await asyncio.sleep(detection_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a VLC-compatible camera stream, publish it to Overshoot, "
            "and save a JPEG every time a plate of food enters the frame."
        )
    )
    parser.add_argument(
        "--stream-url",
        default=os.getenv("CAMERA_STREAM_URL"),
        help="RTSP/HTTP stream URL you can already open in VLC.",
    )
    parser.add_argument(
        "--output-dir",
        default="captures",
        help="Folder where snapshots and JSON sidecars will be written.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OVERSHOOT_MODEL"),
        help="Optional ready model ID from Overshoot. If omitted, one is selected.",
    )
    parser.add_argument(
        "--publish-fps",
        type=float,
        default=4.0,
        help="How fast frames are sent into Overshoot.",
    )
    parser.add_argument(
        "--detection-interval",
        type=float,
        default=1.5,
        help="Seconds between plate-detection checks.",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=2000,
        help="How much recent video context to include in each detection request.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum model confidence required before a plate counts as visible.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=4.0,
        help="Minimum time between saved snapshots.",
    )
    parser.add_argument(
        "--enter-confirmations",
        type=int,
        default=2,
        help="How many consecutive positive detections are required to trigger.",
    )
    parser.add_argument(
        "--exit-confirmations",
        type=int,
        default=2,
        help="How many consecutive negative detections mark the frame as clear.",
    )
    parser.add_argument(
        "--keepalive-seconds",
        type=float,
        default=15.0,
        help="How often to renew the Overshoot stream lease.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=120,
        help="Cap the detector response size to keep calls cheap.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Custom detector prompt. Must still return JSON.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    api_key = os.getenv("OVERSHOOT_API_KEY")
    if not api_key:
        raise RuntimeError("Set OVERSHOOT_API_KEY before starting the watcher.")
    if not args.stream_url:
        raise RuntimeError("Provide --stream-url or set CAMERA_STREAM_URL.")

    feed = CameraFeed(args.stream_url)
    client = OvershootClient(api_key=api_key, model=args.model)
    publisher = LiveKitPublisher()
    stop_event = asyncio.Event()
    tasks: list[asyncio.Task[Any]] = []
    stream_id: Optional[str] = None

    try:
        feed.start()
        first_frame = feed.wait_for_frame()
        height, width = first_frame.shape[:2]

        model = client.choose_model()
        stream = client.create_stream()
        stream_id = stream["id"]
        publish = stream["publish"]

        print(f"Using model: {model}")
        print(f"Created Overshoot stream: {stream_id}")

        await publisher.connect(
            publish["url"],
            publish["token"],
            width=width,
            height=height,
        )

        loop = asyncio.get_running_loop()
        for signame in ("SIGINT", "SIGTERM"):
            try:
                loop.add_signal_handler(
                    getattr(signal, signame),
                    stop_event.set,
                )
            except NotImplementedError:
                pass

        tasks = [
            asyncio.create_task(
                keep_stream_alive(
                    client,
                    stream_id,
                    stop_event,
                    interval_seconds=args.keepalive_seconds,
                )
            ),
            asyncio.create_task(
                publish_camera_frames(
                    feed,
                    publisher,
                    stop_event,
                    fps=args.publish_fps,
                )
            ),
            asyncio.create_task(
                detection_loop(
                    client=client,
                    stream_id=stream_id,
                    model=model,
                    feed=feed,
                    output_dir=Path(args.output_dir),
                    stop_event=stop_event,
                    prompt=args.prompt,
                    detection_interval=args.detection_interval,
                    window_ms=args.window_ms,
                    min_confidence=args.min_confidence,
                    cooldown_seconds=args.cooldown_seconds,
                    enter_confirmations=args.enter_confirmations,
                    exit_confirmations=args.exit_confirmations,
                    max_completion_tokens=args.max_completion_tokens,
                )
            ),
        ]

        await asyncio.gather(*tasks)
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await publisher.close()
        feed.stop()
        if stream_id:
            try:
                client.delete_stream(stream_id)
                print(f"Deleted Overshoot stream: {stream_id}")
            except Exception as exc:
                print(f"Failed to delete stream {stream_id}: {exc}")


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
