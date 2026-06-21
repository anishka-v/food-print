# foodprint

`foodprint` is a dining hall waste-tracking system for plate-level feedback.

It combines:
- a Raspberry Pi camera stream
- Overshoot for full-plate detection
- Gemini for leftover description
- Browserbase for daily menu scraping
- a FastAPI dashboard for dining hall staff

The goal is to show what students are actually leaving on their plates, meal by meal, so dining teams can make better decisions about portions, menus, and purchasing.

## What It Does

`foodprint` has four main parts:

1. `camera_server.py`
   Runs on the Raspberry Pi and exposes the camera as an MJPEG stream at `/video`.

2. `watch_pi_camera.py`
   Runs on a Mac or laptop, watches the Pi stream, uses Overshoot to decide when a full plate is in view, captures an image, sends the image to Gemini for leftover analysis, and stores the result in `foodprint.db`.

3. `browserbase_crossroads_menu_agent.mjs`
   Uses Browserbase + Stagehand to scrape Berkeley Dining menus into JSON files in `menus/`.

4. `dining_waste_tracker_gemini.py` + `staff_dashboard.html`
   Serves the staff dashboard and APIs for menu logs, waste trends, recent plate events, and waste by meal.

## Architecture

```text
Raspberry Pi Camera
  -> Flask MJPEG stream
  -> watch_pi_camera.py
     -> Overshoot: plate present / no plate
     -> Gemini: leftover description from image
     -> SQLite: foodprint.db
  -> FastAPI dashboard
     -> staff_dashboard.html
     -> dining hall analytics

Browserbase
  -> scrape Berkeley Dining menus
  -> menus/*.json
  -> dashboard menu log import
```

## Repository Layout

- [camera_server.py](/Users/anishka/Desktop/projects/eco-dining/camera_server.py)
- [watch_pi_camera.py](/Users/anishka/Desktop/projects/eco-dining/watch_pi_camera.py)
- [dining_waste_tracker_gemini.py](/Users/anishka/Desktop/projects/eco-dining/dining_waste_tracker_gemini.py)
- [staff_dashboard.html](/Users/anishka/Desktop/projects/eco-dining/staff_dashboard.html)
- [browserbase_crossroads_menu_agent.mjs](/Users/anishka/Desktop/projects/eco-dining/browserbase_crossroads_menu_agent.mjs)
- [requirements.txt](/Users/anishka/Desktop/projects/eco-dining/requirements.txt)
- [package.json](/Users/anishka/Desktop/projects/eco-dining/package.json)
- `menus/`
- `captures/`
- `foodprint.db`

## Requirements

### Python

Install:

```bash
python3 -m pip install -r requirements.txt
```

Key Python dependencies:
- `fastapi`
- `uvicorn`
- `opencv-python`
- `google-generativeai`
- `requests`
- `livekit`

### Node

Install:

```bash
npm install
```

Key Node dependencies:
- `@browserbasehq/stagehand`
- `zod`

## Environment Variables

Set the API keys you need before running the watcher or scraper:

```bash
export OVERSHOOT_API_KEY="your_overshoot_key"
export GEMINI_API_KEY="your_gemini_key"
export BROWSERBASE_API_KEY="your_browserbase_key"
```

Optional:

```bash
export GEMINI_MODEL="gemini-1.5-flash"
export DINING_HALL="Crossroads"
export FOODPRINT_DB_PATH="foodprint.db"
export BROWSERBASE_MODEL="google/gemini-2.5-flash"
```

## 1. Run the Pi Camera Server

Run this on the Raspberry Pi:

```bash
pip3 install flask picamera2 opencv-python
python3 camera_server.py
```

The stream should be available at:

```text
http://raspberrypi.local:8080/video
```

If colors look wrong, try setting:

```bash
PI_COLOR_MODE=raw python3 camera_server.py
```

Valid values:
- `raw`
- `rgb_to_bgr`
- `bgr_to_rgb`

## 2. Scrape the Dining Menu

Scrape Crossroads:

```bash
npm run scrape:crossroads-menu
```

Scrape all supported halls:

```bash
npm run scrape:all-menus
```

Output is written to `menus/*.json`.

## 3. Run the Plate Watcher

Run this on your laptop or Mac:

```bash
python3 watch_pi_camera.py \
  --url http://raspberrypi.local:8080/video \
  --dining-hall Crossroads \
  --db-path foodprint.db \
  --show-window
```

What it does:
- waits until a full plate is clearly in frame
- uses Overshoot only for plate detection
- captures an image into `captures/`
- sends the image to Gemini for leftover analysis
- normalizes leftovers into tracked items such as `Banana`, `Bread`, or `Yogurt`
- stores the event in `foodprint.db`

## 4. Run the Dashboard

Start the backend:

```bash
python3 dining_waste_tracker_gemini.py
```

Open the dashboard:

```text
http://localhost:8000/staff
```

The dashboard includes:
- daily menu log
- waste by meal
- daily trend graph
- dining hall menu waste
- recent plate events with thumbnails
- insights

## Current Detection Behavior

The watcher is intentionally conservative:

- Overshoot only answers: is a real full plate present?
- Gemini describes leftovers from the captured image
- menu matching is not forced
- if the exact dish is unclear, generic leftovers are preserved

Examples:
- `banana slices` -> `Banana`
- `bread with spread` -> `Bread`
- `white creamy substance` -> often `Yogurt` or another generic dairy-style label if that is all that is visually supported

This is deliberate. It is better to keep a generic truthful label than invent the wrong menu item.

## Data Storage

Captured events are stored in SQLite at `foodprint.db`.

Main tables:
- `plate_events`
- `leftover_items`

Captured images and metadata sidecars are stored in:
- `captures/*.jpg`
- `captures/*.json`

## Dashboard Data Rules

- `Daily Menu Log` stays based on scraped or manual menu items.
- `Dining Hall Menu Waste` switches to real DB-backed waste data as soon as at least one real plate event exists for the selected hall and window.
- The dashboard will stop showing test placeholder waste rows once real captured data exists.

## Common Commands

Run watcher:

```bash
python3 watch_pi_camera.py --url http://raspberrypi.local:8080/video --dining-hall Crossroads --db-path foodprint.db --show-window
```

Run dashboard:

```bash
python3 dining_waste_tracker_gemini.py
```

Scrape menus:

```bash
npm run scrape:all-menus
```

Clear the database:

```bash
sqlite3 foodprint.db "DELETE FROM leftover_items; DELETE FROM plate_events;"
```

## Notes

- The Pi camera stream must be stable before the watcher starts.
- If `watch_pi_camera.py` times out waiting for frames, confirm `http://raspberrypi.local:8080/video` works in a browser first.
- If Overshoot fails to connect, the issue is usually network or WebRTC connectivity, not the Pi stream.
- If `GEMINI_API_KEY` is missing, the watcher falls back to the generic Overshoot leftover prompt.

## Related Docs

- [PI_CAMERA_SETUP.md](/Users/anishka/Desktop/projects/eco-dining/PI_CAMERA_SETUP.md)
- [PERSON_SNAPSHOT_OVERSHOOT.md](/Users/anishka/Desktop/projects/eco-dining/PERSON_SNAPSHOT_OVERSHOOT.md)
- [BROWSERBASE_MENU_AGENT.md](/Users/anishka/Desktop/projects/eco-dining/BROWSERBASE_MENU_AGENT.md)
