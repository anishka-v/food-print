# Pi Camera Motion Capture

This setup uses a small Flask server on the Raspberry Pi and a Mac watcher that:

- reads the MJPEG stream locally
- uses OpenCV motion detection as a cheap filter
- asks Overshoot whether the motion actually contains a person before saving a photo

## Pi setup

Install dependencies on the Pi:

```bash
pip3 install flask picamera2 opencv-python
```

Run the server on the Pi:

```bash
python3 camera_server.py
```

Test from the Mac in a browser:

```text
http://10.0.0.217:8080/video
```

## Mac setup

Install dependencies on the Mac:

```bash
pip3 install opencv-python requests livekit python-dotenv
```

Run the watcher:

```bash
export OVERSHOOT_API_KEY="ovs-..."
python3 watch_pi_camera.py --url http://10.0.0.217:8080/video --show-window
```

Photos are saved into `captures/`.
Each saved image also gets a JSON sidecar with the Overshoot confidence and response text.

## Tuning

- Increase `--min-changed-pixels` if small lighting shifts are causing too many Overshoot checks.
- Increase `--cooldown-seconds` if one person creates too many photos.
- Lower `--min-changed-pixels` if real motion is being missed before Overshoot gets a chance to check.
- Raise `--min-confidence` if you want fewer borderline person detections.
- Omit `--show-window` if you want it to run headless.
