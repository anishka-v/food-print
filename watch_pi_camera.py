#!/usr/bin/env python3
"""
Watch a Raspberry Pi MJPEG stream, publish frames to Overshoot, and save a
photo when motion happens and Overshoot confirms a plate of food is present.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re
import signal
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import requests
from dotenv import load_dotenv
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional at runtime
    genai = None


OVERSHOOT_BASE_URL = "https://api.overshoot.ai/v1"
DEFAULT_URL = "http://10.0.0.217:8080/video"
DEFAULT_PROMPT = (
    "You are monitoring a fixed camera. Decide whether a real plate of food is "
    "fully visible in the current scene. Only return true when a real plate "
    "and its food are clearly in frame, not cut off by the image edges. "
    "Return false if the plate is only partially visible, entering the frame, leaving the frame, "
    "or too cropped to inspect reliably. Ignore shadows, lighting changes, "
    "reflections, screens, posters, empty plates, bowls, cups, and empty "
    "motion. Return JSON only with this exact "
    'schema: {"plate_of_food_visible": true, "confidence": 0.0, "reason": "short phrase"}'
)
LEFTOVER_ANALYSIS_PROMPT = (
    "You are analyzing a dining hall plate after a student has finished eating. "
    "Identify the foods visibly left on the plate right now and estimate their "
    "relative remaining amounts. Use generic observed food names by default. "
    "Do not try to map foods to the dining hall menu unless the exact dish is visually unmistakable. "
    "List separate visible leftovers as separate food_items whenever possible. "
    "For example, if the plate shows banana slices and bread with spread, return two items such as "
    "\"banana slices\" and \"bread with spread\" rather than guessing a menu item. "
    "Treat composite foods as one item only when they are clearly one visible item. Return JSON only with this exact schema: "
    '{"plate_has_leftovers": true, "summary": "short phrase", "food_items": '
    '[{"name": "food name", "relative_amount_label": "none|minimal|some|most", '
    '"relative_amount_pct": 0, "notes": "short phrase"}]}. '
    "Only include foods that are actually still visible on the plate."
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


class GeminiAnalysisError(RuntimeError):
    pass


@dataclass
class DetectionResult:
    plate_of_food_visible: bool
    confidence: float
    reason: str
    raw_response: str


@dataclass
class LeftoverItem:
    name: str
    relative_amount_label: str
    relative_amount_pct: float
    notes: str


@dataclass
class LeftoverAnalysis:
    plate_has_leftovers: bool
    summary: str
    food_items: list[LeftoverItem]
    raw_response: str


GENERIC_ITEM_KEYWORDS = [
    ("cream cheese", "Cream Cheese"),
    ("whipped cream", "Whipped Cream"),
    ("yogurt", "Yogurt"),
    ("banana", "Banana"),
    ("bread", "Bread"),
    ("bagel", "Bagel"),
    ("roll", "Roll"),
    ("croissant", "Croissant"),
    ("cake", "Cake"),
    ("pie", "Pie"),
    ("cookie", "Cookie"),
    ("soup", "Soup"),
    ("rice", "Rice"),
    ("pasta", "Pasta"),
    ("noodle", "Noodles"),
    ("salad", "Salad"),
    ("pizza", "Pizza"),
    ("fruit", "Fruit"),
]


def normalize_observed_leftovers(
    analysis: LeftoverAnalysis,
    menu_items: Optional[list[str]] = None,
) -> LeftoverAnalysis:
    allowed_exact = {item.lower(): item for item in (menu_items or [])}
    normalized_items: list[LeftoverItem] = []

    for item in analysis.food_items:
        raw_name = item.name.strip()
        if not raw_name:
            continue

        exact_menu_name = allowed_exact.get(raw_name.lower())
        if exact_menu_name is not None:
            normalized_items.append(
                LeftoverItem(
                    name=exact_menu_name,
                    relative_amount_label=item.relative_amount_label,
                    relative_amount_pct=item.relative_amount_pct,
                    notes=item.notes,
                )
            )
            continue

        raw_parts = [
            part.strip(" .")
            for part in re.split(r",| and ", raw_name, flags=re.IGNORECASE)
            if part.strip(" .")
        ]
        parts = raw_parts or [raw_name]
        part_pct = item.relative_amount_pct / max(1, len(parts))

        for part in parts:
            lowered = part.lower()
            final_name = part.strip().title()
            for keyword, canonical in GENERIC_ITEM_KEYWORDS:
                if keyword in lowered:
                    final_name = canonical
                    break

            normalized_items.append(
                LeftoverItem(
                    name=final_name,
                    relative_amount_label=item.relative_amount_label,
                    relative_amount_pct=part_pct,
                    notes=item.notes,
                )
            )

    collapsed: Dict[str, LeftoverItem] = {}
    for item in normalized_items:
        existing = collapsed.get(item.name)
        if existing is None:
            collapsed[item.name] = item
            continue
        collapsed[item.name] = LeftoverItem(
            name=item.name,
            relative_amount_label=existing.relative_amount_label,
            relative_amount_pct=min(100.0, existing.relative_amount_pct + item.relative_amount_pct),
            notes="; ".join(note for note in (existing.notes, item.notes) if note),
        )

    return LeftoverAnalysis(
        plate_has_leftovers=analysis.plate_has_leftovers,
        summary=analysis.summary,
        food_items=list(collapsed.values()),
        raw_response=analysis.raw_response,
    )


class GeminiLeftoverAnalyzer:
    def __init__(self, api_key: str, model_name: str) -> None:
        if genai is None:
            raise GeminiAnalysisError(
                "google-generativeai is not installed. Install requirements first."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def analyze_image(
        self,
        *,
        dining_hall: str,
        active_meal: str,
        menu_items: list[str],
        frame_bgr: Any,
    ) -> LeftoverAnalysis:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        prompt = (
            "You are analyzing a single dining hall plate image after eating. "
            "Describe only the foods visibly left on the plate right now and estimate their relative remaining amounts. "
            "Use generic observed food names by default, such as banana slices, bread, yogurt, cream cheese, cake, rice, noodles, soup, or salad. "
            "Do not try to match foods to the dining hall menu unless the exact dish is visually unmistakable. "
            "Do not infer or guess a menu item from weak evidence. "
            "If the food is ambiguous, keep the generic observed label. "
            "List separate visible leftovers as separate food_items whenever possible. "
            "For example, if the image shows banana slices and bread with spread, return separate items like "
            "\"banana slices\" and \"bread with spread\". "
            "If multiple visible parts clearly belong to one composed item, you may describe them as one observed leftover. "
            f"Dining hall: {dining_hall}. Meal: {active_meal}. "
            "Return JSON only with this exact schema: "
            '{"plate_has_leftovers": true, "summary": "short phrase", "food_items": '
            '[{"name": "generic observed leftover or exact menu item if visually obvious", "relative_amount_label": "none|minimal|some|most", '
            '"relative_amount_pct": 0, "notes": "short phrase"}]}. '
            "Known same-day menu items for reference only: "
            + ", ".join(menu_items or ["none"])
            + ". "
            "Do not force the output to match this menu. The menu is only a hint for exact obvious dishes."
        )

        response = self.model.generate_content([prompt, image])
        text = getattr(response, "text", "") or ""
        if not text:
            raise GeminiAnalysisError("Gemini returned an empty leftover-analysis response.")

        data = parse_json_fragment(text)
        items: list[LeftoverItem] = []
        for item in data.get("food_items", []):
            items.append(
                LeftoverItem(
                    name=str(item.get("name", "")).strip() or "Unknown",
                    relative_amount_label=str(
                        item.get("relative_amount_label", "some")
                    ).strip()
                    or "some",
                    relative_amount_pct=max(
                        0.0,
                        min(100.0, float(item.get("relative_amount_pct", 0.0))),
                    ),
                    notes=str(item.get("notes", "")).strip(),
                )
            )

        return LeftoverAnalysis(
            plate_has_leftovers=bool(data.get("plate_has_leftovers", bool(items))),
            summary=str(data.get("summary", "")).strip(),
            food_items=items,
            raw_response=text,
        )


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

    def analyze_plate_leftovers(
        self,
        *,
        stream_id: str,
        model: str,
        prompt: str,
        window_ms: int,
        max_completion_tokens: int,
    ) -> LeftoverAnalysis:
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
        items = []
        for item in data.get("food_items", []):
            items.append(
                LeftoverItem(
                    name=str(item.get("name", "")).strip() or "Unknown",
                    relative_amount_label=str(item.get("relative_amount_label", "some")).strip() or "some",
                    relative_amount_pct=max(
                        0.0,
                        min(100.0, float(item.get("relative_amount_pct", 0.0))),
                    ),
                    notes=str(item.get("notes", "")).strip(),
                )
            )
        return LeftoverAnalysis(
            plate_has_leftovers=bool(data.get("plate_has_leftovers", bool(items))),
            summary=str(data.get("summary", "")).strip(),
            food_items=items,
            raw_response=content,
        )


class PlateEventStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS plate_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                dining_hall TEXT NOT NULL,
                capture_path TEXT NOT NULL,
                detection_confidence REAL NOT NULL,
                detection_reason TEXT NOT NULL,
                analysis_summary TEXT NOT NULL,
                analysis_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS leftover_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                food_name TEXT NOT NULL,
                relative_amount_label TEXT NOT NULL,
                relative_amount_pct REAL NOT NULL,
                notes TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES plate_events(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def save_plate_event(
        self,
        *,
        dining_hall: str,
        capture_path: str,
        detection: DetectionResult,
        analysis: LeftoverAnalysis,
    ) -> int:
        captured_at = datetime.now().isoformat()
        payload = {
            "captured_at": captured_at,
            "dining_hall": dining_hall,
            "capture_path": capture_path,
            "detection_confidence": detection.confidence,
            "detection_reason": detection.reason,
            "analysis_summary": analysis.summary,
            "food_items": [
                {
                    "name": item.name,
                    "relative_amount_label": item.relative_amount_label,
                    "relative_amount_pct": item.relative_amount_pct,
                    "notes": item.notes,
                }
                for item in analysis.food_items
            ],
            "raw_detection_response": detection.raw_response,
            "raw_analysis_response": analysis.raw_response,
        }
        cursor = self.conn.execute(
            """
            INSERT INTO plate_events (
                captured_at,
                dining_hall,
                capture_path,
                detection_confidence,
                detection_reason,
                analysis_summary,
                analysis_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                captured_at,
                dining_hall,
                capture_path,
                detection.confidence,
                detection.reason,
                analysis.summary,
                json.dumps(payload),
            ),
        )
        event_id = cursor.lastrowid
        self.conn.executemany(
            """
            INSERT INTO leftover_items (
                event_id,
                food_name,
                relative_amount_label,
                relative_amount_pct,
                notes
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    event_id,
                    item.name,
                    item.relative_amount_label,
                    item.relative_amount_pct,
                    item.notes,
                )
                for item in analysis.food_items
            ],
        )
        self.conn.commit()
        return int(event_id)

    def close(self) -> None:
        self.conn.close()


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


def plate_looks_centered(frame: Any) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, min(frame.shape[:2]) // 4),
        param1=90,
        param2=28,
        minRadius=max(30, min(frame.shape[:2]) // 10),
        maxRadius=max(80, min(frame.shape[:2]) // 2),
    )
    if circles is None:
        return True

    height, width = frame.shape[:2]
    best_circle = max(circles[0], key=lambda circle: circle[2])
    x, y, r = [float(value) for value in best_circle]
    margin = r * 0.18
    return (
        x - r > margin
        and y - r > margin
        and x + r < width - margin
        and y + r < height - margin
    )


def normalize_hall_slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def normalize_menu_name(name: str) -> str:
    cleaned = name.lower().strip()
    for token in ("(vegan)", "(vegetarian)", "halal", "vegan", "base - "):
        cleaned = cleaned.replace(token, "")
    cleaned = " ".join(cleaned.replace("/", " ").replace("-", " ").split())
    return cleaned


GENERIC_MENU_TOKENS = {
    "a",
    "an",
    "and",
    "baked",
    "bite",
    "bites",
    "bread",
    "cake",
    "cream",
    "dish",
    "dollop",
    "food",
    "left",
    "leftover",
    "leftovers",
    "of",
    "on",
    "piece",
    "pieces",
    "plate",
    "portion",
    "remaining",
    "remains",
    "sauce",
    "side",
    "slice",
    "slices",
    "small",
    "some",
    "spread",
    "topping",
    "white",
}


def menu_name_tokens(name: str) -> set[str]:
    return {
        token
        for token in normalize_menu_name(name).split()
        if len(token) > 1 and token not in GENERIC_MENU_TOKENS
    }


def match_menu_item_name(observed_name: str, lookup: Dict[str, str]) -> Optional[str]:
    normalized = normalize_menu_name(observed_name)
    if not normalized:
        return None

    exact_match = lookup.get(normalized)
    if exact_match is not None:
        return exact_match

    observed_tokens = menu_name_tokens(observed_name)
    if len(observed_tokens) < 2:
        return None

    best_match: Optional[str] = None
    best_score = 0.0
    for canonical_norm, canonical_name in lookup.items():
        canonical_tokens = menu_name_tokens(canonical_name)
        if len(canonical_tokens) < 2:
            continue

        overlap = observed_tokens & canonical_tokens
        if len(overlap) < 2:
            continue

        coverage = len(overlap) / len(observed_tokens)
        precision = len(overlap) / len(canonical_tokens)
        score = (coverage * 0.7) + (precision * 0.3)
        if coverage >= 0.6 and score > best_score:
            best_match = canonical_name
            best_score = score

    return best_match


def load_menu_context(menu_dir: str, dining_hall: str) -> Dict[str, Any]:
    slug = normalize_hall_slug(dining_hall)
    pattern = os.path.join(menu_dir, f"{slug}*.json")
    matches = glob.glob(pattern)
    if not matches:
        return {}
    latest = max(matches, key=os.path.getmtime)
    try:
        with open(latest, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    payload["_source_file"] = latest
    return payload


def infer_service_meal(now: Optional[datetime] = None) -> str:
    current = now or datetime.now()
    minutes = current.hour * 60 + current.minute
    if minutes < 10 * 60 + 30:
        return "breakfast"
    if minutes < 16 * 60:
        return "lunch"
    return "dinner"


def menu_item_lookup(
    menu_context: Dict[str, Any],
    allowed_meals: Optional[list[str]] = None,
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    meal_names = allowed_meals or ["breakfast", "lunch", "dinner"]
    for meal_name in meal_names:
        for item in menu_context.get(meal_name, []):
            lookup[normalize_menu_name(item)] = item
    return lookup


def meal_menu_items(
    menu_context: Dict[str, Any],
    active_meal: Optional[str] = None,
) -> list[str]:
    meal_names = [active_meal] if active_meal else ["breakfast", "lunch", "dinner"]
    items: list[str] = []
    for meal_name in meal_names:
        if not meal_name:
            continue
        items.extend(menu_context.get(meal_name, []))
    return items


def constrain_analysis_to_menu(
    analysis: LeftoverAnalysis,
    menu_context: Dict[str, Any],
    allowed_meals: Optional[list[str]] = None,
) -> LeftoverAnalysis:
    if not menu_context:
        return analysis

    lookup = menu_item_lookup(menu_context, allowed_meals)
    if not lookup:
        return analysis

    constrained_by_name: Dict[str, LeftoverItem] = {}
    for item in analysis.food_items:
        matched_name = match_menu_item_name(item.name, lookup)
        final_name = matched_name or item.name

        existing = constrained_by_name.get(final_name)
        if existing is None:
            constrained_by_name[final_name] = LeftoverItem(
                name=final_name,
                relative_amount_label=item.relative_amount_label,
                relative_amount_pct=item.relative_amount_pct,
                notes=item.notes,
            )
            continue

        combined_pct = min(100.0, existing.relative_amount_pct + item.relative_amount_pct)
        combined_notes = "; ".join(
            note for note in (existing.notes, item.notes) if note
        )
        constrained_by_name[final_name] = LeftoverItem(
            name=final_name,
            relative_amount_label=existing.relative_amount_label,
            relative_amount_pct=combined_pct,
            notes=combined_notes,
        )

    constrained_items = list(constrained_by_name.values())

    return LeftoverAnalysis(
        plate_has_leftovers=bool(constrained_items),
        summary=analysis.summary,
        food_items=constrained_items,
        raw_response=analysis.raw_response,
    )


def build_leftover_prompt(
    dining_hall: str,
    menu_context: Dict[str, Any],
    active_meal: Optional[str] = None,
) -> str:
    prompt = LEFTOVER_ANALYSIS_PROMPT + f" Dining hall: {dining_hall}."
    if not menu_context:
        return prompt

    menu_lines = []
    meal_names = [active_meal] if active_meal else ["breakfast", "lunch", "dinner"]
    for meal_name in meal_names:
        if not meal_name:
            continue
        items = menu_context.get(meal_name, [])
        if items:
            menu_lines.append(f"{meal_name.title()}: {', '.join(items[:25])}")
    if menu_lines:
        prompt += (
            " These known same-day menu items are reference only. Use them only if the item is visually obvious; otherwise keep a generic observed label. "
            f" Current service window: {active_meal or 'current service window'}. "
            + " Known menu items today: "
            + " | ".join(menu_lines)
        )
    return prompt


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
    parser.add_argument(
        "--dining-hall",
        default=os.getenv("DINING_HALL", "Crossroads"),
        help="Dining hall label stored with each plate event.",
    )
    parser.add_argument(
        "--db-path",
        default="foodprint.db",
        help="SQLite database path for plate leftovers.",
    )
    parser.add_argument(
        "--menu-dir",
        default="menus",
        help="Directory containing scraped dining hall menu JSON files.",
    )
    parser.add_argument(
        "--leftover-max-completion-tokens",
        type=int,
        default=300,
        help="Cap the Overshoot leftover-analysis response size.",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        help="Gemini model used for menu-item matching when GEMINI_API_KEY is set.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    api_key = os.getenv("OVERSHOOT_API_KEY")
    if not api_key:
        raise RuntimeError("Set OVERSHOOT_API_KEY before running the watcher.")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    blur_kernel = args.blur_kernel if args.blur_kernel % 2 == 1 else args.blur_kernel + 1
    capture_dir = Path(args.capture_dir)

    feed = CameraFeed(args.url)
    client = OvershootClient(api_key=api_key, requested_model=args.model)
    store = PlateEventStore(args.db_path)
    publisher = LiveKitPublisher()
    gemini_analyzer: Optional[GeminiLeftoverAnalyzer] = None
    stop_event = asyncio.Event()
    tasks: list[asyncio.Task[Any]] = []
    stream_id: Optional[str] = None
    menu_context = load_menu_context(args.menu_dir, args.dining_hall)
    if gemini_api_key:
        try:
            gemini_analyzer = GeminiLeftoverAnalyzer(gemini_api_key, args.gemini_model)
        except Exception as exc:
            print(f"Gemini leftover analyzer unavailable: {exc}")

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
        print(f"Storing plate events in {args.db_path} for {args.dining_hall}")
        if gemini_analyzer is not None:
            print(f"Using Gemini leftover analysis model: {args.gemini_model}")
        if menu_context:
            print(
                "Loaded menu context from "
                f"{Path(menu_context['_source_file']).name} for {args.dining_hall}"
            )

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
                        full_plate_now = visible_now and plate_looks_centered(frame)

                        if full_plate_now:
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
                            try:
                                active_meal = infer_service_meal()
                                leftover_prompt = build_leftover_prompt(
                                    args.dining_hall,
                                    menu_context,
                                    active_meal,
                                )
                                all_day_menu_items = meal_menu_items(menu_context, None)
                                if gemini_analyzer is not None:
                                    try:
                                        analysis = gemini_analyzer.analyze_image(
                                            dining_hall=args.dining_hall,
                                            active_meal=active_meal,
                                            menu_items=all_day_menu_items,
                                            frame_bgr=frame,
                                        )
                                    except Exception as exc:
                                        print(f"Gemini leftover analysis failed, using fallback: {exc}")
                                        analysis = client.analyze_plate_leftovers(
                                            stream_id=stream_id,
                                            model=model,
                                            prompt=leftover_prompt,
                                            window_ms=args.window_ms,
                                            max_completion_tokens=args.leftover_max_completion_tokens,
                                        )
                                else:
                                    analysis = client.analyze_plate_leftovers(
                                        stream_id=stream_id,
                                        model=model,
                                        prompt=leftover_prompt,
                                        window_ms=args.window_ms,
                                        max_completion_tokens=args.leftover_max_completion_tokens,
                                    )
                                analysis = normalize_observed_leftovers(
                                    analysis,
                                    all_day_menu_items,
                                )
                                event_id = store.save_plate_event(
                                    dining_hall=args.dining_hall,
                                    capture_path=str(path),
                                    detection=detection,
                                    analysis=analysis,
                                )
                                print(
                                    "Stored plate event:",
                                    json.dumps(
                                        {
                                            "event_id": event_id,
                                            "dining_hall": args.dining_hall,
                                            "summary": analysis.summary,
                                            "food_items": [
                                                {
                                                    "name": item.name,
                                                    "relative_amount_label": item.relative_amount_label,
                                                    "relative_amount_pct": item.relative_amount_pct,
                                                }
                                                for item in analysis.food_items
                                            ],
                                        }
                                    ),
                                )
                            except Exception as exc:
                                print(f"Leftover analysis/storage failed: {exc}")
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
        store.close()
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
