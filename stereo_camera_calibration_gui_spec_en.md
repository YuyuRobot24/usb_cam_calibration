# Stereo Camera Calibration GUI for a Dental Suction Tool

> Implementation specification for Codex. Place this file in an empty project directory and ask Codex to implement it stage by stage. Deliver a complete, runnable Python application rather than isolated examples or pseudocode.

## 1. Objective

Build a Python desktop GUI for calibrating the cameras mounted on a dental suction tool. The application must support:

1. **Monocular intrinsic calibration** for either the left or right camera.
2. **Stereo calibration** using paired left/right frames, previously calibrated intrinsics, stereo extrinsics, rectification, and quantitative error evaluation.

System assumptions:

- Maximum camera resolution: `1280 × 720`.
- Main useful viewing distance: approximately `100 mm`.
- Current chessboard labels: `12-150-10`, `12-100-6`, and `12-70-5`.
- Initial camera backend: ordinary UVC/USB cameras addressed by OpenCV device index.

The GUI must show live camera images and a permanently visible calibration-error panel below the control buttons. It must not report only a single, ambiguous “calibration error”.

## 2. Chessboard definitions must remain configurable

The precise meaning of the labels `12-150-10`, `12-100-6`, and `12-70-5` has not yet been confirmed. Do not infer the number of corners or square size from these strings.

Treat them as preset names. Each preset must contain editable fields:

- `inner_corners_cols`: number of horizontal **internal corners**;
- `inner_corners_rows`: number of vertical **internal corners**;
- `square_size_mm`: measured side length of one square in millimetres;
- optional notes, manufacturer, and measured total length.

If a preset contains missing values, disable frame capture and explain what must be entered. The UI must say “internal corners, not the number of black/white squares”. Allow at least `0.001 mm` input precision.

## 3. Technology

- Ubuntu 22.04 or 24.04
- Python 3.10+
- PySide6 for the GUI
- `opencv-python-headless` for image capture, corner detection, calibration, and rectification
- NumPy
- PyYAML
- pytest

Use `opencv-python-headless`, not `opencv-python`, by default. The headless package still includes `VideoCapture`, chessboard detection, calibration, remapping, and image-processing functions. It excludes OpenCV's own HighGUI windows. Therefore:

- never call `cv2.imshow`, `cv2.waitKey`, `cv2.namedWindow`, or `cv2.destroyAllWindows`;
- convert OpenCV BGR frames to RGB and display them in PySide6 using `QImage`/`QPixmap`;
- use Qt keyboard events and timers instead of OpenCV HighGUI;
- do not install `opencv-python` and `opencv-python-headless` in the same virtual environment.

## 4. Ubuntu virtual-environment setup

The generated `README.md` must include these commands.

### 4.1 Install system packages

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

Do not require `sudo` inside the project after this step.

### 4.2 Create and activate the venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` should use compatible version ranges rather than unbounded dependencies. A suitable initial file is:

```text
PySide6>=6.7,<7
opencv-python-headless>=4.10,<5
numpy>=1.26,<3
PyYAML>=6,<7
pytest>=8,<9
```

If a full OpenCV wheel was previously installed, the README must explain how to clean the environment:

```bash
python -m pip uninstall -y opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless
python -m pip install -r requirements.txt
```

If ArUco/ChArUco support is added later, replace the OpenCV dependency with `opencv-contrib-python-headless`; never install both headless wheels together.

### 4.3 Check cameras and permissions

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

If the current user cannot access the device:

```bash
sudo usermod -aG video "$USER"
```

The user must log out and back in after the group change. The application should list probeable camera indices, but it must not open every camera continuously during discovery.

Useful capability check:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### 4.4 Run and test

```bash
source .venv/bin/activate
python app.py
```

```bash
source .venv/bin/activate
pytest -q
```

For a GUI smoke test on a machine without a display only:

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

Do not use offscreen mode for real camera calibration because the operator must inspect the images.

## 5. Project structure

```text
stereo_calibration_gui/
├── README.md
├── requirements.txt
├── pyproject.toml
├── app.py
├── configs/
│   └── default.yaml
├── calibration_gui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── camera_backend.py
│   ├── camera_worker.py
│   ├── board_config.py
│   ├── corner_detector.py
│   ├── mono_calibrator.py
│   ├── stereo_calibrator.py
│   ├── metrics.py
│   ├── models.py
│   ├── session.py
│   └── storage.py
├── tests/
│   ├── test_board_points.py
│   ├── test_metrics.py
│   ├── test_storage.py
│   └── test_synthetic_calibration.py
└── output/
```

Separate GUI, camera acquisition, board detection, calibration mathematics, metrics, session state, and serialization. Long operations must not block the Qt event loop.

## 6. Main window

### 6.1 Configuration toolbar

Provide:

- mode: `Monocular intrinsics` or `Stereo calibration`;
- monocular side: `Left` or `Right`;
- left and right camera device index;
- requested width, height, and FPS, defaulting to `1280 × 720`;
- board preset selector with the three named boards and `Custom`;
- internal-corner columns and rows;
- square size in mm;
- output directory;
- `Open camera(s)` and `Stop camera(s)`.

Changing mode must not mix samples from different sessions. Ask for confirmation if unsaved samples exist.

### 6.2 Live view

Monocular mode shows one large view. Stereo mode shows synchronized views side by side with clear `LEFT` and `RIGHT` labels.

Preview images may be scaled to fit the UI, but detection during accepted capture and all calibration calculations must use original frames. Never calibrate with display coordinates or permanently resized preview images.

Overlay:

- actual resolution and measured FPS;
- board-detection state and indexed corners;
- accepted sample count and target count;
- stereo timestamp difference;
- blur, small-board, boundary, duplicate-pose, or missing-pair warnings;
- estimated board distance only after usable intrinsics are available; label it as an estimate.

For two independent UVC cameras, timestamp frames with `time.monotonic_ns()`. Pair the nearest frames. Show the time difference and reject pairs above a configurable threshold, initially `20 ms`. Make clear that software pairing is not equivalent to hardware synchronization.

### 6.3 Controls below the view

Include:

- `Capture frame/pair`;
- `Auto capture`;
- `Delete last sample`;
- `Manage samples`;
- `Run calibration`;
- `Save result`;
- `Load calibration`;
- stereo-only `Preview rectification`;
- `Reset session`.

Shortcuts:

- Space: capture;
- C: calibrate;
- Delete or Backspace: remove last sample;
- R: toggle raw/corner overlay;
- Esc: stop auto-capture or rectification preview, not the whole program.

Disable actions whose prerequisites are not satisfied and explain why in a tooltip/status message.

## 7. Permanently visible error-evaluation panel

Place this panel below the operation buttons. Do not show the principal results only in a modal dialog.

### 7.1 Monocular panel

Display:

- success state, timestamp, and number of samples used;
- RMS returned by `cv2.calibrateCamera`, in pixels;
- independently computed all-corner reprojection RMSE;
- mean absolute, median, 95th-percentile, and maximum reprojection error;
- per-image reprojection RMSE table;
- worst five images, selectable for removal and recalibration;
- `fx`, `fy`, `cx`, and `cy` and the full camera matrix;
- distortion coefficients and camera model;
- parameter sanity warnings, including principal point outside the image or unusual `fx/fy` ratio;
- image-grid corner coverage, board-scale/distance distribution, and pose-diversity indicators.

For every view, project object points with its fitted `rvec` and `tvec` using `cv2.projectPoints`, then compute per-corner and per-view errors. Do not confuse OpenCV's calibration RMS with mean Euclidean error.

Use colour hints only as guidance:

- green: `<0.3 px`;
- yellow: `0.3–0.6 px`;
- orange: `0.6–1.0 px`;
- red: `>1.0 px`.

The UI and report must state that these are empirical guidance, not universal pass/fail thresholds.

### 7.2 Stereo panel

Display:

- success state, timestamp, and number of pairs used;
- RMS returned by `cv2.stereoCalibrate`;
- baseline `||T||` in mm;
- rotation matrix `R` and translation vector `T`;
- rectified common-corner vertical error `|y_left-y_right|`: mean, median, P95, and maximum;
- symmetric epipolar distance or Sampson distance: mean, median, P95, and maximum;
- per-pair vertical and epipolar errors;
- worst five pairs, selectable for removal and recalibration;
- valid rectification ROIs and their common-valid-area ratio;
- optional comparison with a mechanically measured baseline, clearly noting that housing distance is not necessarily optical-centre distance.

Compute rectified coordinates with:

```python
cv2.undistortPoints(points_left, K1, D1, R=R1, P=P1)
cv2.undistortPoints(points_right, K2, D2, R=R2, P=P2)
```

Then evaluate vertical residuals. Do not calculate the rectified vertical error from raw coordinates.

Use a mean vertical residual below `0.5 px` as an initial target, with `0.1–0.3 px` preferred. Always show P95 and maximum as well.

### 7.3 Depth-uncertainty estimator

In stereo mode, provide a small calculator using rectified focal length, baseline, target distance, and assumed disparity error:

\[
\sigma_Z \approx \frac{Z^2}{f_x B}\sigma_d
\]

Default to `Z = 100 mm` and show results for `sigma_d = 0.2, 0.5, 1.0 px`. All input values must be editable. This is a theoretical uncertainty estimate, not measured depth accuracy.

## 8. Sample acquisition and quality control

Use 20 valid samples as an initial minimum and target 30–50 diverse samples. More highly repetitive images should not be treated as better calibration data.

For the approximately `100 mm` working distance:

- guide the user to cover roughly `70–130 mm` when focus and field of view permit;
- include at least near, middle, and far board scales;
- cover the image centre, corners, and edges;
- include clear tilt about the board X and Y axes and some in-plane rotation;
- avoid a dataset containing only centred, fronto-parallel boards.

Before intrinsics exist, board distance cannot be determined reliably in mm. Use board image-area ratio as a scale proxy. After an initial calibration, use `solvePnP` to estimate distance and recommend missing ranges.

Before accepting a sample, verify:

- all internal corners are detected;
- stereo sides use the same board and corner count;
- blur score passes a configurable threshold;
- board area is not too small;
- corners are not too close to the boundary;
- centre, scale, and pose are sufficiently different from existing samples;
- stereo timestamp difference passes the threshold.

Use `cv2.findChessboardCornersSB()` by default. If falling back to `findChessboardCorners()`, refine using `cornerSubPix()`.

Fast downscaled detection is allowed for live feedback, but each accepted frame must be re-detected at original `1280 × 720` resolution. Visualize corner indices so the operator can detect correspondence-order problems. Reject suspected left/right 180-degree ordering inconsistencies.

Auto-capture requires a stable, valid detection over multiple frames, a diversity check, and a configurable cooldown of approximately `0.5–1.0 s`.

## 9. Calibration algorithms

### 9.1 Object points

```python
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
objp *= square_size_mm
```

Use millimetres consistently; stereo translation and baseline are consequently in mm.

### 9.2 Monocular calibration

Initially implement the OpenCV pinhole/Brown–Conrady model using `cv2.calibrateCamera`. Keep advanced flags optional and avoid over-parameterization by default. Add an explicitly separate fisheye mode only if needed; never mix pinhole and fisheye parameter formats.

Outlier handling must be transparent:

1. calibrate using all accepted samples;
2. display per-view results;
3. let the user remove selected outliers;
4. recalibrate;
5. preserve removal history.

Do not repeatedly auto-delete images merely to produce an attractive RMS.

### 9.3 Stereo calibration

Preferred sequence:

1. load left and right intrinsics captured with the same resolution, focus, optical window, and image settings;
2. validate image-size compatibility;
3. run `cv2.stereoCalibrate(..., flags=cv2.CALIB_FIX_INTRINSIC)`;
4. run `cv2.stereoRectify`;
5. build remap matrices;
6. provide live rectified preview with horizontal guide lines;
7. compute stereo metrics.

If intrinsics are unavailable, the stereo workflow may estimate separate left and right intrinsics from the paired dataset, show those results, and then rerun stereo calibration with fixed intrinsics. Do not silently finish after a single unconstrained joint optimization.

## 10. Image-setting compatibility

Store and validate:

- image width and height;
- camera index and serial number when available;
- requested and measured FPS;
- obtainable focus, zoom, crop, and binning settings;
- board configuration;
- calibration model.

Reject calibration files with mismatched image dimensions by default. Do not silently scale intrinsics. Permit explicit proportional scaling only when the user confirms that no crop or aspect-ratio change occurred, and show a warning.

## 11. Storage

Create one timestamped session directory:

```text
output/session_YYYYMMDD_HHMMSS/
├── session.yaml
├── accepted/left/
├── accepted/right/
├── rejected/
├── overlays/
├── mono_left.yaml
├── mono_right.yaml
├── stereo.yaml
├── metrics.json
├── per_view_errors.csv
└── calibration_report.md
```

Save original images without overlays. Store overlays separately. Record timestamps, detected corners, blur and coverage scores, acceptance/rejection reason, and stereo time difference.

YAML must contain ordinary lists, not NumPy/Python-specific tags, and include a schema version. At minimum save image dimensions, board settings, camera matrices, distortion coefficients, `R`, `T`, baseline, rectification matrices, projection matrices, `Q`, calibration RMS, and vertical-error summary. Use temporary-file plus atomic replacement for result files.

The generated Markdown report should summarize settings, samples, parameters, metrics, worst views/pairs, removals, and warnings.

## 12. Concurrency and robustness

- Capture frames in worker threads and communicate through Qt signals/slots.
- Run calibration in a worker/thread pool.
- Release cameras and stop workers on shutdown.
- Prevent identical left/right indices unless a backend explicitly supports a dual-stream device.
- Report unsupported resolution, permission error, missing frame, disconnection, or unexpected actual dimensions clearly.
- Freeze the sample set during calibration.
- Never use preview-display coordinates for calibration.
- Catch OpenCV exceptions and provide actionable error messages.
- Avoid unbounded frame queues; retain only recent frames required for display/pairing.

## 13. Tests

Implement at least:

1. correct object-point count, order, and millimetre scaling;
2. correct reprojection-error aggregation and units;
3. correct rectified vertical-error calculation;
4. Sampson/epipolar error near zero for perfect synthetic correspondences;
5. YAML round trip preserves values and shapes;
6. mismatched image dimensions are rejected;
7. stereo sample pairing/deletion preserves correspondence;
8. synthetic noisy projections recover known camera parameters within stated tolerances;
9. basic Qt widget construction works with `QT_QPA_PLATFORM=offscreen`.

Do not claim real-camera accuracy from synthetic tests. Document a separate manual camera validation procedure.

## 14. Manual acceptance checklist

- [ ] One documented command starts the application.
- [ ] Monocular and stereo modes can be selected.
- [ ] All three board presets can be selected and edited.
- [ ] Both cameras and `1280 × 720` can be configured.
- [ ] Monocular live view and corner/quality overlay work.
- [ ] Stereo views update together and samples remain paired.
- [ ] The corresponding error panel remains visible below the buttons.
- [ ] Monocular overall and per-view reprojection errors are shown.
- [ ] Stereo RMS, baseline, vertical residual, and epipolar error are shown.
- [ ] Worst samples can be removed and calibration rerun.
- [ ] Live rectification preview includes horizontal guide lines.
- [ ] YAML, JSON, CSV, and Markdown report outputs are created.
- [ ] Saved calibration can be loaded after restart.
- [ ] Camera errors do not crash the application.
- [ ] All pytest tests pass.
- [ ] Neither `opencv-python` nor OpenCV HighGUI calls are used.

## 15. Codex implementation sequence

Complete and test each stage before continuing:

1. Create structure, dependencies, models, configuration, and Ubuntu README.
2. Implement editable board presets and YAML persistence.
3. Implement camera abstraction, UVC backend, threaded capture, and PySide6 preview.
4. Implement original-resolution board detection, overlays, and quality checks.
5. Implement sessions, paired sample management, deletion, and storage.
6. Implement monocular calibration and its complete error panel.
7. Implement stereo calibration, rectification, and its complete error panel.
8. Implement rectified live preview and depth-uncertainty estimator.
9. Complete reports, failure handling, and automated tests.
10. Run `pytest`, run an application smoke test, and fix all failures.
11. Report the launch command, completion checklist, and remaining user inputs.

Do not leave pseudocode in core features. The only acceptable unresolved inputs are the three boards' true internal-corner dimensions and measured square sizes, plus any proprietary camera-SDK details.

## 16. Initial configuration

```yaml
camera:
  left_index: 0
  right_index: 1
  width: 1280
  height: 720
  fps: 30
  max_pair_delta_ms: 20

capture:
  minimum_samples: 20
  target_samples: 35
  auto_capture_cooldown_s: 0.8
  work_distance_hint_mm: 100
  preferred_distance_range_mm: [70, 130]

boards:
  - preset_name: "12-150-10"
    inner_corners_cols: null
    inner_corners_rows: null
    square_size_mm: null
  - preset_name: "12-100-6"
    inner_corners_cols: null
    inner_corners_rows: null
    square_size_mm: null
  - preset_name: "12-70-5"
    inner_corners_cols: null
    inner_corners_rows: null
    square_size_mm: null

calibration:
  model: pinhole
  detector: findChessboardCornersSB
  fix_intrinsic_for_stereo: true
  rectify_alpha: 0.0

evaluation:
  mono_good_rms_px: 0.3
  mono_warning_rms_px: 0.6
  mono_poor_rms_px: 1.0
  stereo_target_mean_vertical_px: 0.5
  depth_evaluation_distance_mm: 100
  disparity_noise_px: [0.2, 0.5, 1.0]
```

All thresholds are initial empirical settings. Keep them editable and label them as guidance in the generated report.
