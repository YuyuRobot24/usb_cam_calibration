# Stereo Camera Calibration GUI

Python/PySide6 desktop application for monocular and stereo calibration of UVC/USB cameras on a dental suction tool.

## Ubuntu system packages

```bash
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  v4l-utils \
  libgl1 \
  libglib2.0-0 \
  libxkbcommon-x11-0 \
  libxcb-cursor0 \
  libegl1
```

No `sudo` is required inside the project after this step.

## Create and activate the venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

This project intentionally uses `opencv-python-headless`, not `opencv-python`. It still includes `VideoCapture`, chessboard detection, calibration, rectification, and image-processing functions, but excludes OpenCV HighGUI windows. The application displays frames through Qt `QImage`/`QPixmap`.

If a full OpenCV wheel was previously installed, clean the environment:

```bash
python -m pip uninstall -y opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless
python -m pip install -r requirements.txt
```

If ArUco/ChArUco support is added later, replace the OpenCV dependency with `opencv-contrib-python-headless`; never install both headless wheels together.

## Check cameras and permissions

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

If the current user cannot access the device:

```bash
sudo usermod -aG video "$USER"
```

Log out and back in after the group change.

Useful capability check:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

The application has a probe button for camera indices, but it does not keep every camera open during discovery.

## Live preview latency

The live preview uses the latest received frame and throttles live chessboard detection so the GUI does not fall behind. Capture still re-detects corners on the original full-resolution frame before saving a sample.

Tune these values in `configs/default.yaml` if needed:

```yaml
capture:
  live_detection_interval_s: 0.25
  max_preview_lag_ms: 250
```

## Run

```bash
source .venv/bin/activate
python app.py
```

In the GUI, choose or type the output base directory, then press **Create save dir** before capturing. The application does not create a timestamped session directory just because it starts or because the output path changes.

Auto capture is enabled by default in `configs/default.yaml`:

```yaml
capture:
  auto_capture_default: true
```

Capture still requires a valid board configuration, a created save directory, and open camera stream.

## Test

```bash
source .venv/bin/activate
pytest -q
```

For a GUI smoke test on a machine without a display only:

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

Do not use offscreen mode for real camera calibration because the operator must inspect the images.

## Board presets

The labels `12-150-10`, `12-100-6`, and `12-70-5` are treated only as preset names. Their internal-corner counts and square sizes are intentionally blank until measured. Capture stays disabled until each selected board has:

- horizontal internal corners;
- vertical internal corners;
- square size in millimetres.

These are internal corners, not the number of black/white squares. Square size supports `0.001 mm` precision.

## Manual validation procedure

1. Enter the real board internal-corner counts and measured square size.
2. Choose the output base directory and press **Create save dir**.
3. Capture 30-50 diverse views per monocular camera at the final image settings.
4. Watch the ROS-style sample coverage bars for X, Y, scale, and skew. New frames that are too similar to accepted frames are rejected so the set becomes more diverse.
5. Cover centre, edges, corners, in-plane rotation, and board tilt.
6. For the 100 mm working distance, include roughly 70-130 mm when focus and field of view permit.
7. Calibrate each monocular camera and inspect OpenCV RMS, independently computed all-corner reprojection RMSE, per-image errors, and worst images.
8. Remove explicit outliers only after visual inspection, then recalibrate.
9. Capture paired stereo samples with the same board and image settings.
10. Run stereo calibration, inspect baseline, rectified vertical residual mean/median/P95/max, Sampson error, and valid ROIs.
11. Use rectification preview horizontal guide lines to visually inspect alignment.
12. Save results. The session directory contains YAML, JSON, CSV, original images, and a Markdown report.

Synthetic tests verify math and serialization behavior only; they do not claim real-camera accuracy.
