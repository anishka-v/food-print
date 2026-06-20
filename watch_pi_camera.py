#!/usr/bin/env python3
"""
Watch a Raspberry Pi MJPEG stream, publish frames to Overshoot, and save a
photo when motion happens and Overshoot confirms a plate of food is present.
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
DEFAULT_URL = "http://10.0.0.217:8080/video"
DEFAULT_PROMPT = (
    "You are monitoring a fixed camera. Decide whether a real plate of food is "
    "visible in the current scene. Ignore shadows, lighting changes, "
    "reflections, screens, posters, empty plates, bowls, cups, and empty "
    "motion. Return JSON only with this exact "
    'schema: {"plate_of_food_visible": true, "confidence": 0.0, "reason": "short phrase"}'
)


def parse_json_fragment(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not extract JSON from model response: {text!r}")
    return json.loads(text[start : end + 1])


class OvershootError(RuntimeError):
    pass


@dataclass
class DetectionResult:
    plate_of_food_visible: bool
    confidence: float
    reason: str
    raw_response: str


class OvershootClient:
    def __init__(self, api_key: str, requested_model: Optional[str]) -> None:
        self.requested_model = requested_model
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

        for candidate in ("Qwen/Qwen3.5-9B", "google/gemma-4-E4B-it"):
            if candidate in models:
                return candidate
        return models[0]

    def create_stream(self) -> Dict[str, Any]:
        response = self.session.post(f"{OVERSHOOT_BASE_URL}/streams", timeout=20)
        self._raise_for_status(response)
        return response.json()

    def keepalive(self, stream_id: str) -> None:
        response = self.session.post(
            f"{OVERSHOOT_BASE_URL}/streams/{stream_id}/keepalive",
            timeout=20,
        )
        self._raise_for_status(response)

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
    def __init__(self, url: str, max_width: int = 960) -> None:
        self.url = url
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
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                print(f"Could not connect to Pi camera stream at {self.url}; retrying.")
                time.sleep(2)
                continue
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print("Failed to read Pi camera frame; reconnecting.")
                        time.sleep(1)
                        break
                    frame = self._resize(frame)
                    with self._lock:
                        self._latest_frame = frame.copy()
                        self._latest_at = time.time()
            finally:
                cap.release()

    def wait_for_frame(self, timeout: float = 15.0) -> Any:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.get_latest_frame()
            if frame is not None:
                return frame
            time.sleep(0.1)
        raise TimeoutError("Timed out waiting for Pi camera frames.")

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

        raise RuntimeError(f"Unable to create LiveKit frame: {last_error}")

    async def close(self) -> None:
        if self.room is not None:
            await self.room.disconnect()


def save_photo(frame: Any, capture_dir: Path, detection: DetectionResult) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_path = capture_dir / f"plate_{timestamp}.jpg"
    meta_path = capture_dir / f"plate_{timestamp}.json"
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


async def keep_stream_alive(
    client: OvershootClient,
    stream_id: str,
    stop_event: asyncio.Event,
    keepalive_seconds: float,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=keepalive_seconds)
        except asyncio.TimeoutError:
            try:
                client.keepalive(stream_id)
            except Exception as exc:
                print(f"Keepalive failed: {exc}")


async def publish_frames(
    feed: CameraFeed,
    publisher: LiveKitPublisher,
    stop_event: asyncio.Event,
    publish_fps: float,
) -> None:
    interval = 1.0 / publish_fps
    while not stop_event.is_set():
        frame = feed.get_latest_frame()
        if frame is not None:
            publisher.push_frame(frame)
        await asyncio.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Watch a Pi MJPEG stream, use motion as a cheap gate, and use "
            "Overshoot to confirm whether the motion contains a plate of food."
        )
    )
    parser.add_argument(
        "--url",
        default=os.getenv("PI_CAMERA_URL", DEFAULT_URL),
        help="Pi camera MJPEG URL.",
    )
    parser.add_argument(
        "--capture-dir",
        default="captures",
        help="Directory for saved photos.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OVERSHOOT_MODEL"),
        help="Optional ready Overshoot model ID.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=5.0,
        help="Minimum time between saved photos.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=int,
        default=25,
        help="Pixel-difference threshold applied before counting motion.",
    )
    parser.add_argument(
        "--min-changed-pixels",
        type=int,
        default=6000,
        help="Minimum changed pixels required before querying Overshoot.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=21,
        help="Gaussian blur kernel size. Must be odd.",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Display the live camera window locally.",
    )
    parser.add_argument(
        "--publish-fps",
        type=float,
        default=4.0,
        help="How fast frames are sent into Overshoot.",
    )
    parser.add_argument(
        "--keepalive-seconds",
        type=float,
        default=15.0,
        help="How often to renew the Overshoot stream lease.",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=2000,
        help="How much recent video context to send to Overshoot.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum Overshoot confidence required to count as a plate.",
    )
    parser.add_argument(
        "--enter-confirmations",
        type=int,
        default=2,
        help="Consecutive positive detections required before saving a plate.",
    )
    parser.add_argument(
        "--exit-confirmations",
        type=int,
        default=2,
        help="Consecutive negative detections required before a plate is considered gone.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=120,
        help="Cap the Overshoot response size.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Custom Overshoot prompt. Must still return JSON.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    api_key = os.getenv("OVERSHOOT_API_KEY")
    if not api_key:
        raise RuntimeError("Set OVERSHOOT_API_KEY before running the watcher.")

    blur_kernel = args.blur_kernel if args.blur_kernel % 2 == 1 else args.blur_kernel + 1
    capture_dir = Path(args.capture_dir)

    feed = CameraFeed(args.url)
    client = OvershootClient(api_key=api_key, requested_model=args.model)
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

        print(f"Connected to Pi camera stream at {args.url}")
        print(f"Using Overshoot model: {model}")
        print(f"Created Overshoot stream: {stream_id}")

        await publisher.connect(
            publish["url"],
            publish["token"],
            width=width,
            height=height,
        )

        tasks = [
            asyncio.create_task(
                keep_stream_alive(
                    client,
                    stream_id,
                    stop_event,
                    args.keepalive_seconds,
                )
            ),
            asyncio.create_task(
                publish_frames(
                    feed,
                    publisher,
                    stop_event,
                    args.publish_fps,
                )
            ),
        ]

        loop = asyncio.get_running_loop()
        for signame in ("SIGINT", "SIGTERM"):
            try:
                loop.add_signal_handler(getattr(signal, signame), stop_event.set)
            except NotImplementedError:
                pass

        prev_gray = None
        last_capture_time = 0.0
        last_overshoot_check = 0.0
        plate_present = False
        visible_streak = 0
        absent_streak = 0

        print("Watching for motion. Press q to quit.")

        while not stop_event.is_set():
            frame = feed.get_latest_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

            if prev_gray is None:
                prev_gray = gray
                await asyncio.sleep(0.05)
                continue

            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(
                diff,
                args.motion_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            changed_pixels = cv2.countNonZero(thresh)
            now = time.time()

            if args.show_window:
                cv2.imshow("Pi Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            should_check = (
                changed_pixels > args.min_changed_pixels
                and now - last_overshoot_check > 1.0
                and feed.latest_age_seconds() < 5
            )

            if should_check:
                last_overshoot_check = now
                try:
                    stream_state = client.get_stream(stream_id)
                    if stream_state.get("last_frame_index") is None:
                        print("Overshoot has not received frames yet.")
                    else:
                        detection = client.detect_plate_of_food(
                            stream_id=stream_id,
                            model=model,
                            prompt=args.prompt,
                            window_ms=args.window_ms,
                            max_completion_tokens=args.max_completion_tokens,
                        )
                        print(
                            "Detection:",
                            json.dumps(
                                {
                                    "plate_of_food_visible": detection.plate_of_food_visible,
                                    "confidence": round(detection.confidence, 3),
                                    "reason": detection.reason,
                                    "changed_pixels": int(changed_pixels),
                                }
                            ),
                        )
                        visible_now = (
                            detection.plate_of_food_visible
                            and detection.confidence >= args.min_confidence
                        )

                        if visible_now:
                            visible_streak += 1
                            absent_streak = 0
                        else:
                            visible_streak = 0
                            absent_streak += 1

                        if (
                            not plate_present
                            and visible_streak >= args.enter_confirmations
                            and now - last_capture_time > args.cooldown_seconds
                        ):
                            path = save_photo(frame, capture_dir, detection)
                            last_capture_time = now
                            plate_present = True
                            print(f"Saved photo: {path}")
                        elif plate_present and absent_streak >= args.exit_confirmations:
                            plate_present = False
                            print("Plate cleared from frame; ready for the next plate.")
                except Exception as exc:
                    print(f"Overshoot detection failed: {exc}")

            prev_gray = gray
            await asyncio.sleep(0.05)
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await publisher.close()
        feed.stop()
        if args.show_window:
            cv2.destroyAllWindows()
        if stream_id:
            try:
                client.delete_stream(stream_id)
                print(f"Deleted Overshoot stream: {stream_id}")
            except Exception as exc:
                print(f"Failed to delete Overshoot stream {stream_id}: {exc}")


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
