# Person Snapshot Watcher

This script watches the same camera URL you can already open in VLC, publishes the frames to Overshoot, and saves a JPEG each time a person newly enters the frame.

## Environment

Create a `.env` file or export these variables:

```bash
OVERSHOOT_API_KEY=ovs-...
CAMERA_STREAM_URL=rtsp://...
OVERSHOOT_MODEL=Qwen/Qwen3.5-9B
```

`OVERSHOOT_MODEL` is optional. If omitted, the script asks Overshoot for the currently ready models and picks one automatically.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 person_snapshot_overshoot.py --output-dir captures
```

Useful flags:

```bash
python3 person_snapshot_overshoot.py \
  --stream-url rtsp://YOUR-PI-STREAM \
  --publish-fps 4 \
  --detection-interval 1.5 \
  --min-confidence 0.65 \
  --cooldown-seconds 4
```

## Output

- JPEG snapshots go into `captures/`
- A JSON file with model confidence and the raw Overshoot reply is written next to each image

## Notes

- Overshoot streams expire unless they are renewed. This script sends keepalives every 15 seconds.
- The detector only saves one image per entry event. A person lingering in frame should not create repeated images until the frame clears.
- Detection frequency affects API cost. If you want lower cost, increase `--detection-interval`.
