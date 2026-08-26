from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PySide6.QtCore import Qt, QThread, QTimer, Slot
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from .board_config import board_from_dict, board_to_dict
from .camera_backend import CameraFrame, probe_camera_indices
from .camera_worker import CameraWorker
from .corner_detector import detect_chessboard, draw_overlay
from .metrics import depth_uncertainty, ros_is_good_sample, ros_sample_parameters, ros_sample_progress
from .models import BoardConfig, DetectionResult, MonoCalibrationResult, Sample, StereoSample
from .mono_calibrator import calibrate_mono
from .session import CalibrationSession
from .stereo_calibrator import calibrate_stereo
from .storage import (
    atomic_write_json,
    atomic_write_yaml,
    load_yaml,
    mono_to_dict,
    save_image,
    stereo_to_dict,
    validate_image_size,
    write_per_view_csv,
    write_report,
)


class MainWindow(QMainWindow):
    def __init__(self, config_path: str | Path = "configs/default.yaml"):
        super().__init__()
        self.setWindowTitle("Stereo Camera Calibration")
        self.resize(1300, 900)
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) if self.config_path.exists() else {}
        self.boards = [board_from_dict(item) for item in self.config.get("boards", [])]
        self.boards.append(BoardConfig("Custom"))
        self.session: CalibrationSession | None = None
        self.left_frame: CameraFrame | None = None
        self.right_frame: CameraFrame | None = None
        self.left_detection = None
        self.right_detection = None
        self.last_detection_time_s = {"left": 0.0, "right": 0.0}
        self.mono_result: MonoCalibrationResult | None = None
        self.mono_left_result: MonoCalibrationResult | None = None
        self.mono_right_result: MonoCalibrationResult | None = None
        self.stereo_result = None
        self.workers: list[tuple[QThread, CameraWorker]] = []
        self.auto_capture = False
        self.last_auto_capture_s = 0.0
        self.rectification_preview = False
        capture_config = self.config.get("capture", {})
        self.live_detection_interval_s = capture_config.get("live_detection_interval_s", 0.25)
        self.max_preview_lag_ms = capture_config.get("max_preview_lag_ms", 250.0)
        self._build_ui()
        self._wire_shortcuts()
        self._apply_config_defaults()
        self._sync_board_fields()
        self._update_buttons()

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        self.setCentralWidget(root)

        toolbar = QGroupBox("Configuration")
        form = QGridLayout(toolbar)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setColumnMinimumWidth(0, 170)
        form.setColumnMinimumWidth(1, 240)
        form.setColumnMinimumWidth(2, 170)
        form.setColumnMinimumWidth(3, 240)
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(2, 0)
        form.setColumnStretch(3, 1)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Monocular intrinsics", "Stereo calibration"])
        self.side_combo = QComboBox()
        self.side_combo.addItems(["Left", "Right"])
        self.left_index = QSpinBox()
        self.left_index.setRange(0, 32)
        self.right_index = QSpinBox()
        self.right_index.setRange(0, 32)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(160, 4096)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(120, 2160)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.board_combo = QComboBox()
        self.board_combo.addItems([board.preset_name for board in self.boards])
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(0, 100)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(0, 100)
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setDecimals(3)
        self.square_spin.setSingleStep(0.001)
        self.square_spin.setRange(0, 1000)
        self.output_dir = QLineEdit("output")
        self.browse_output = QPushButton("Browse")
        self.create_session_button = QPushButton("Create save dir")
        self.open_button = QPushButton("Open camera(s)")
        self.stop_button = QPushButton("Stop camera(s)")
        self.probe_button = QPushButton("Probe indices")

        fields = [
            ("Mode", self.mode_combo),
            ("Monocular side", self.side_combo),
            ("Left camera index", self.left_index),
            ("Right camera index", self.right_index),
            ("Width", self.width_spin),
            ("Height", self.height_spin),
            ("FPS", self.fps_spin),
            ("Board preset", self.board_combo),
            ("Internal corner columns", self.cols_spin),
            ("Internal corner rows", self.rows_spin),
            ("Square size mm", self.square_spin),
        ]
        for row, (label, widget) in enumerate(fields):
            label_widget = QLabel(label)
            label_widget.setMinimumWidth(170)
            widget.setMinimumWidth(240)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            form.addWidget(label_widget, row // 2, (row % 2) * 2)
            form.addWidget(widget, row // 2, (row % 2) * 2 + 1)
        output_label = QLabel("Output directory")
        output_label.setMinimumWidth(170)
        self.output_dir.setMinimumWidth(240)
        self.output_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for button in [self.browse_output, self.create_session_button, self.probe_button, self.open_button, self.stop_button]:
            button.setMinimumWidth(150)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addWidget(output_label, 6, 0)
        form.addWidget(self.output_dir, 6, 1)
        form.addWidget(self.browse_output, 6, 2)
        form.addWidget(self.create_session_button, 6, 3)
        form.addWidget(self.probe_button, 7, 0)
        form.addWidget(self.open_button, 7, 1)
        form.addWidget(self.stop_button, 7, 2)
        outer.addWidget(toolbar)

        self.view_area = QHBoxLayout()
        self.left_view = QLabel("LEFT")
        self.right_view = QLabel("RIGHT")
        for view in (self.left_view, self.right_view):
            view.setAlignment(Qt.AlignCenter)
            view.setMinimumSize(420, 260)
            view.setStyleSheet("background: #14171a; color: #e8ecef; border: 1px solid #3b444b;")
            self.view_area.addWidget(view, 1)
        outer.addLayout(self.view_area, 4)

        controls = QHBoxLayout()
        self.capture_button = QPushButton("Capture frame/pair")
        self.auto_checkbox = QCheckBox("Auto capture")
        self.delete_button = QPushButton("Delete last sample")
        self.manage_button = QPushButton("Manage samples")
        self.calibrate_button = QPushButton("Run calibration")
        self.save_button = QPushButton("Save result")
        self.load_button = QPushButton("Load calibration")
        self.rectify_button = QPushButton("Preview rectification")
        self.reset_button = QPushButton("Reset session")
        for button in [
            self.capture_button,
            self.delete_button,
            self.manage_button,
            self.calibrate_button,
            self.save_button,
            self.load_button,
            self.rectify_button,
            self.reset_button,
        ]:
            button.setMinimumWidth(150)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            controls.addWidget(button)
        self.auto_checkbox.setMinimumWidth(150)
        self.auto_checkbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        controls.addWidget(self.auto_checkbox)
        outer.addLayout(controls)

        self.error_tabs = QTabWidget()
        self.mono_tab = QWidget()
        mono_layout = QVBoxLayout(self.mono_tab)
        axis_box = QGroupBox("Online monocular calibration indicators")
        axis_layout = QGridLayout(axis_box)
        self.online_state_label = QLabel("ROS-style sample coverage: collect diverse X, Y, scale, and skew.")
        axis_layout.addWidget(self.online_state_label, 0, 0, 1, 2)
        self.online_bars: dict[str, QProgressBar] = {}
        self.online_labels: dict[str, QLabel] = {}
        for row, col, key, text in [
            (1, 0, "x", "X"),
            (1, 1, "y", "Y"),
            (2, 0, "scale", "Scale"),
            (2, 1, "skew", "Skew"),
        ]:
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            label = QLabel(f"{text}: --")
            label.setMinimumWidth(155)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%v / 100")
            bar.setFixedHeight(18)
            bar.setMinimumWidth(220)
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cell_layout.addWidget(label)
            cell_layout.addWidget(bar, 1)
            axis_layout.addWidget(cell, row, col)
            self.online_labels[key] = label
            self.online_bars[key] = bar
        axis_layout.setColumnStretch(0, 1)
        axis_layout.setColumnStretch(1, 1)
        axis_layout.setColumnMinimumWidth(0, 420)
        axis_layout.setColumnMinimumWidth(1, 420)
        mono_layout.addWidget(axis_box)
        self.mono_panel = QPlainTextEdit()
        self.mono_panel.setReadOnly(True)
        mono_layout.addWidget(self.mono_panel)
        self.stereo_panel = QPlainTextEdit()
        self.stereo_panel.setReadOnly(True)
        self.per_view_table = QTableWidget(0, 6)
        self.per_view_table.setHorizontalHeaderLabels(["index", "rmse", "mean", "median", "p95", "max"])
        self.depth_panel = QWidget()
        depth_form = QFormLayout(self.depth_panel)
        self.depth_z = QDoubleSpinBox()
        self.depth_z.setRange(1, 5000)
        self.depth_z.setValue(100)
        self.depth_fx = QDoubleSpinBox()
        self.depth_fx.setRange(1, 10000)
        self.depth_baseline = QDoubleSpinBox()
        self.depth_baseline.setRange(0.001, 1000)
        self.depth_noise = QLineEdit("0.2, 0.5, 1.0")
        self.depth_output = QLabel("")
        for label, widget in [
            ("Target distance Z mm", self.depth_z),
            ("Rectified fx px", self.depth_fx),
            ("Baseline mm", self.depth_baseline),
            ("Disparity error px", self.depth_noise),
            ("Uncertainty", self.depth_output),
        ]:
            depth_form.addRow(label, widget)
        self.error_tabs.addTab(self.mono_tab, "Monocular errors")
        self.error_tabs.addTab(self.stereo_panel, "Stereo errors")
        self.error_tabs.addTab(self.per_view_table, "Per-view/pair")
        self.error_tabs.addTab(self.depth_panel, "Depth uncertainty")
        outer.addWidget(self.error_tabs, 2)

        self.status = QLabel("Enter board values before capture. Internal corners, not the number of black/white squares.")
        outer.addWidget(self.status)

        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        self.side_combo.currentTextChanged.connect(self._mode_changed)
        self.board_combo.currentIndexChanged.connect(self._sync_board_fields)
        self.cols_spin.valueChanged.connect(self._board_fields_changed)
        self.rows_spin.valueChanged.connect(self._board_fields_changed)
        self.square_spin.valueChanged.connect(self._board_fields_changed)
        self.browse_output.clicked.connect(self._browse_output)
        self.create_session_button.clicked.connect(self.create_session)
        self.probe_button.clicked.connect(self._probe)
        self.open_button.clicked.connect(self.open_cameras)
        self.stop_button.clicked.connect(self.stop_cameras)
        self.capture_button.clicked.connect(self.capture)
        self.auto_checkbox.stateChanged.connect(self._toggle_auto)
        self.delete_button.clicked.connect(self.delete_last_sample)
        self.manage_button.clicked.connect(self.manage_samples)
        self.calibrate_button.clicked.connect(self.run_calibration)
        self.save_button.clicked.connect(self.save_result)
        self.load_button.clicked.connect(self.load_calibration)
        self.rectify_button.clicked.connect(self._toggle_rectification)
        self.reset_button.clicked.connect(self.reset_session)
        for widget in (self.depth_z, self.depth_fx, self.depth_baseline, self.depth_noise):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._update_depth)
            else:
                widget.textChanged.connect(self._update_depth)

    def _wire_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.capture)
        QShortcut(QKeySequence("C"), self, activated=self.run_calibration)
        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self.delete_last_sample)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, activated=self.delete_last_sample)
        QShortcut(QKeySequence("R"), self, activated=self._toggle_rectification)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._escape)

    def _apply_config_defaults(self) -> None:
        camera = self.config.get("camera", {})
        self.left_index.setValue(camera.get("left_index", 0))
        self.right_index.setValue(camera.get("right_index", 1))
        self.width_spin.setValue(camera.get("width", 1280))
        self.height_spin.setValue(camera.get("height", 720))
        self.fps_spin.setValue(camera.get("fps", 30))
        evaluation = self.config.get("evaluation", {})
        self.depth_z.setValue(evaluation.get("depth_evaluation_distance_mm", 100))
        capture_config = self.config.get("capture", {})
        self.auto_checkbox.setChecked(capture_config.get("auto_capture_default", False))
        self.auto_capture = self.auto_checkbox.isChecked()

    def _current_board(self) -> BoardConfig:
        return self.boards[self.board_combo.currentIndex()]

    def _sync_board_fields(self) -> None:
        board = self._current_board()
        self.cols_spin.blockSignals(True)
        self.rows_spin.blockSignals(True)
        self.square_spin.blockSignals(True)
        self.cols_spin.setValue(board.inner_corners_cols or 0)
        self.rows_spin.setValue(board.inner_corners_rows or 0)
        self.square_spin.setValue(board.square_size_mm or 0.0)
        self.cols_spin.blockSignals(False)
        self.rows_spin.blockSignals(False)
        self.square_spin.blockSignals(False)
        self._update_buttons()

    def _board_fields_changed(self) -> None:
        board = self._current_board()
        board.inner_corners_cols = self.cols_spin.value() or None
        board.inner_corners_rows = self.rows_spin.value() or None
        board.square_size_mm = self.square_spin.value() or None
        if board.complete and self.session is None:
            self.status.setText("Board is valid. Press Create save dir before capturing.")
        self._update_buttons()

    def _mode(self) -> str:
        return "stereo" if self.mode_combo.currentText().startswith("Stereo") else "mono"

    def _side(self) -> str:
        return self.side_combo.currentText().lower()

    def _mode_changed(self) -> None:
        if self.session and self.session.sample_count:
            choice = QMessageBox.question(self, "Reset session?", "Changing mode or side starts a new session. Reset unsaved samples?")
            if choice != QMessageBox.Yes:
                return
        self.reset_session()

    def create_session(self) -> None:
        if self.session and self.session.sample_count:
            choice = QMessageBox.question(self, "Create new save dir?", "Create a new save directory and stop using the current session?")
            if choice != QMessageBox.Yes:
                return
        self.session = CalibrationSession.create(self.output_dir.text(), self._mode(), self._side())
        atomic_write_yaml(self.session.root / "session.yaml", self.session.to_dict())
        auto_text = " Auto capture is on." if self.auto_capture else ""
        self.status.setText(f"Save directory created: {self.session.root}.{auto_text}")
        self._reset_online_bars()
        self._update_buttons()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output directory", self.output_dir.text())
        if path:
            self.output_dir.setText(path)
            self.session = None
            self.status.setText("Output base directory selected. Press Create save dir before capturing.")
            self._update_buttons()

    def _probe(self) -> None:
        indices = probe_camera_indices()
        self.status.setText("Probeable camera indices: " + (", ".join(map(str, indices)) if indices else "none found"))

    def open_cameras(self) -> None:
        if self._mode() == "stereo" and self.left_index.value() == self.right_index.value():
            QMessageBox.warning(self, "Invalid camera indices", "Stereo mode requires different left/right indices for ordinary UVC cameras.")
            return
        self.stop_cameras()
        specs = [(self.left_index.value(), "left")]
        if self._mode() == "stereo":
            specs.append((self.right_index.value(), "right"))
        for index, side in specs:
            thread = QThread(self)
            worker = CameraWorker(index, self.width_spin.value(), self.height_spin.value(), self.fps_spin.value())
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            if side == "left":
                worker.frame_ready.connect(self._left_frame_ready, Qt.QueuedConnection)
            else:
                worker.frame_ready.connect(self._right_frame_ready, Qt.QueuedConnection)
            worker.error.connect(self._camera_error, Qt.QueuedConnection)
            worker.stopped.connect(thread.quit)
            worker.stopped.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            self.workers.append((thread, worker))
            thread.start()
        self.status.setText("Camera stream started.")
        self._update_buttons()

    def stop_cameras(self) -> None:
        for thread, worker in self.workers:
            worker.stop()
            thread.quit()
            thread.wait(1000)
        self.workers.clear()
        self._update_buttons()

    def _camera_error(self, message: str) -> None:
        self.status.setText(message)
        QMessageBox.warning(self, "Camera error", message)

    @Slot(object)
    def _left_frame_ready(self, frame: CameraFrame) -> None:
        self._frame_ready("left", frame)

    @Slot(object)
    def _right_frame_ready(self, frame: CameraFrame) -> None:
        self._frame_ready("right", frame)

    def _frame_ready(self, side: str, frame: CameraFrame) -> None:
        now_s = time.monotonic()
        lag_ms = (time.monotonic_ns() - frame.timestamp_ns) / 1e6
        if lag_ms > self.max_preview_lag_ms:
            if side == "left":
                self.left_frame = frame
            else:
                self.right_frame = frame
            return
        detection = self.left_detection if side == "left" else self.right_detection
        if detection is None:
            detection = DetectionResult(False, None, 0.0, 0.0, 0.0, ["Waiting for live detection."])
        if now_s - self.last_detection_time_s[side] >= self.live_detection_interval_s:
            detection = detect_chessboard(frame.frame_bgr, self._current_board())
            self.last_detection_time_s[side] = now_s
        if side == "left":
            self.left_frame = frame
            self.left_detection = detection
        else:
            self.right_frame = frame
            self.right_detection = detection
        self._show_frame(side, frame, detection)
        if self.auto_capture:
            now = time.monotonic()
            cooldown = self.config.get("capture", {}).get("auto_capture_cooldown_s", 0.8)
            if now - self.last_auto_capture_s >= cooldown:
                if self._capture_ready()[0]:
                    QTimer.singleShot(0, self.capture)
                    self.last_auto_capture_s = now

    def _show_frame(self, side: str, frame: CameraFrame, detection) -> None:
        board = self._current_board()
        image = draw_overlay(frame.frame_bgr, board, detection, side.upper())
        if self.rectification_preview and self.stereo_result is not None:
            image = self._rectify_frame(side, image)
        text = f"{frame.actual_size[0]}x{frame.actual_size[1]} {frame.measured_fps:.1f} fps"
        cv2.putText(image, text, (12, image.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        label = self.left_view if side == "left" else self.right_view
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _rectify_frame(self, side: str, frame_bgr: np.ndarray) -> np.ndarray:
        result = self.stereo_result
        K = result.camera_matrix_left if side == "left" else result.camera_matrix_right
        D = result.dist_coeffs_left if side == "left" else result.dist_coeffs_right
        R = result.R1 if side == "left" else result.R2
        P = result.P1 if side == "left" else result.P2
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, result.image_size, cv2.CV_16SC2)
        rectified = cv2.remap(frame_bgr, map1, map2, cv2.INTER_LINEAR)
        for y in range(40, rectified.shape[0], 80):
            cv2.line(rectified, (0, y), (rectified.shape[1], y), (0, 255, 255), 1)
        return rectified

    def _capture_ready(self) -> tuple[bool, str]:
        board = self._current_board()
        if self.session is None:
            return False, "Press Create save dir before capturing."
        if not board.complete:
            return False, board.missing_reason()
        if not self.workers:
            return False, "Open camera(s) first."
        if self._mode() == "mono":
            detection = self.left_detection
            frame = self.left_frame
            if self._side() == "right":
                detection = self.right_detection or self.left_detection
                frame = self.right_frame or self.left_frame
            if frame is None:
                return False, "A camera frame is required before capture."
            if detection is None or not detection.found:
                return False, "A detected board is required before capture."
            return True, ""
        if self.left_frame is None or self.right_frame is None or self.left_detection is None or self.right_detection is None:
            return False, "Both stereo frames are required."
        if not (self.left_detection.found and self.right_detection.found):
            return False, "Both cameras must detect the same board."
        delta_ms = abs(self.left_frame.timestamp_ns - self.right_frame.timestamp_ns) / 1e6
        max_delta = self.config.get("camera", {}).get("max_pair_delta_ms", 20)
        if delta_ms > max_delta:
            return False, f"Stereo timestamp delta {delta_ms:.1f} ms exceeds {max_delta} ms. Software pairing is not hardware sync."
        return True, ""

    def capture(self) -> None:
        ready, reason = self._capture_ready()
        if not ready:
            self.status.setText(reason)
            QToolTip.showText(self.capture_button.mapToGlobal(self.capture_button.rect().center()), reason)
            return
        assert self.session is not None
        sample_id = f"{self.session.sample_count + 1:04d}"
        if self._mode() == "mono":
            side = self._side()
            frame = self.right_frame if side == "right" and self.right_frame is not None else self.left_frame
            detection = detect_chessboard(frame.frame_bgr, self._current_board())
            if not detection.found:
                self.status.setText("Capture rejected: board was not detected in the original frame.")
                return
            params = self._sample_params(detection.corners)
            if not ros_is_good_sample(params, self._existing_ros_params("mono")):
                self.status.setText("Capture rejected: frame is too similar to an accepted sample.")
                return
            path = self.session.root / "accepted" / side / f"{sample_id}.png"
            save_image(path, frame.frame_bgr)
            sample = Sample(
                sample_id,
                side,
                path,
                frame.timestamp_ns,
                detection.corners.copy(),
                detection.blur_score,
                detection.board_area_ratio,
                params,
            )
            self.session.add_mono(sample)
        else:
            left_detection = detect_chessboard(self.left_frame.frame_bgr, self._current_board())
            right_detection = detect_chessboard(self.right_frame.frame_bgr, self._current_board())
            if not (left_detection.found and right_detection.found):
                self.status.setText("Capture rejected: board was not detected in both original stereo frames.")
                return
            params = self._sample_params(left_detection.corners)
            if not ros_is_good_sample(params, self._existing_ros_params("stereo")):
                self.status.setText("Capture rejected: pair is too similar to an accepted pair.")
                return
            left_path = self.session.root / "accepted" / "left" / f"{sample_id}.png"
            right_path = self.session.root / "accepted" / "right" / f"{sample_id}.png"
            save_image(left_path, self.left_frame.frame_bgr)
            save_image(right_path, self.right_frame.frame_bgr)
            left = Sample(sample_id, "left", left_path, self.left_frame.timestamp_ns, left_detection.corners.copy(), left_detection.blur_score, left_detection.board_area_ratio, params)
            right = Sample(sample_id, "right", right_path, self.right_frame.timestamp_ns, right_detection.corners.copy(), right_detection.blur_score, right_detection.board_area_ratio, params)
            delta_ms = abs(left.timestamp_ns - right.timestamp_ns) / 1e6
            self.session.add_pair(StereoSample(sample_id, left, right, delta_ms))
        atomic_write_yaml(self.session.root / "session.yaml", self.session.to_dict())
        self.status.setText(f"Accepted sample {sample_id}. Count: {self.session.sample_count}.")
        self._update_ros_coverage_bars()
        self._update_buttons()

    def _sample_params(self, corners: np.ndarray) -> list[float]:
        return ros_sample_parameters(corners, (self.width_spin.value(), self.height_spin.value()), self._current_board().pattern_size[0])

    def _existing_ros_params(self, mode: str) -> list[list[float]]:
        if not self.session:
            return []
        if mode == "stereo":
            return [pair.left.ros_params for pair in self.session.stereo_pairs if pair.left.ros_params]
        return [sample.ros_params for sample in self.session.mono_samples if sample.ros_params]

    def _toggle_auto(self, *_args) -> None:
        self.auto_capture = self.auto_checkbox.isChecked()
        self.status.setText("Auto capture enabled." if self.auto_capture else "Auto capture stopped.")
        self._update_buttons()

    def delete_last_sample(self) -> None:
        if not self.session:
            return
        removed = self.session.delete_last()
        self.status.setText(f"Removed {removed}." if removed else "No samples to remove.")
        atomic_write_yaml(self.session.root / "session.yaml", self.session.to_dict())
        self._update_ros_coverage_bars()
        self._update_buttons()

    def manage_samples(self) -> None:
        count = self.session.sample_count if self.session else 0
        removed = ", ".join(self.session.removed_ids) if self.session and self.session.removed_ids else "none"
        QMessageBox.information(self, "Samples", f"Accepted samples: {count}\nRemoved samples: {removed}\nWorst views/pairs appear in the error panel after calibration.")

    def run_calibration(self) -> None:
        if self.session is None:
            self.status.setText("Press Create save dir before calibration.")
            return
        image_size = (self.width_spin.value(), self.height_spin.value())
        try:
            if self._mode() == "mono":
                result = calibrate_mono(list(self.session.mono_samples), self._current_board(), image_size, self._side())
                self.mono_result = result
                if result.side == "left":
                    self.mono_left_result = result
                else:
                    self.mono_right_result = result
                self._show_mono_result(result)
            else:
                result = calibrate_stereo(
                    list(self.session.stereo_pairs),
                    self._current_board(),
                    image_size,
                    self.mono_left_result,
                    self.mono_right_result,
                    self.config.get("calibration", {}).get("fix_intrinsic_for_stereo", True),
                    self.config.get("calibration", {}).get("rectify_alpha", 0.0),
                )
                self.stereo_result = result
                self._show_stereo_result(result)
        except Exception as exc:
            self.status.setText(str(exc))
            QMessageBox.warning(self, "Calibration failed", str(exc))
        self._update_buttons()

    def _show_mono_result(self, result: MonoCalibrationResult) -> None:
        metrics = result.metrics["all_corner"]
        self._update_ros_coverage_bars()
        warnings = "\n".join(result.metrics.get("sanity_warnings", [])) or "None"
        self.mono_panel.setPlainText(
            "\n".join(
                [
                    f"Success: yes",
                    f"Timestamp: {result.timestamp}",
                    f"Samples used: {result.samples_used}",
                    f"OpenCV RMS: {result.rms_px:.4f} px",
                    f"Independent all-corner RMSE: {metrics['rmse']:.4f} px",
                    f"Mean/median/P95/max: {metrics['mean']:.4f} / {metrics['median']:.4f} / {metrics['p95']:.4f} / {metrics['max']:.4f} px",
                    "Guidance colours: green <0.3 px, yellow 0.3-0.6 px, orange 0.6-1.0 px, red >1.0 px; empirical guidance only.",
                    f"fx fy cx cy: {result.camera_matrix[0,0]:.3f}, {result.camera_matrix[1,1]:.3f}, {result.camera_matrix[0,2]:.3f}, {result.camera_matrix[1,2]:.3f}",
                    f"Camera matrix:\n{result.camera_matrix}",
                    f"Distortion coefficients:\n{result.dist_coeffs.reshape(-1)}",
                    f"Sanity warnings:\n{warnings}",
                    "Coverage/diversity: inspect accepted images; include centre, edges, corners, tilt, rotation, and near/mid/far board scales.",
                ]
            )
        )
        self._fill_table(result.metrics["per_view"])
        self.error_tabs.setCurrentWidget(self.mono_tab)
        self.status.setText("Monocular calibration complete.")

    def _reset_online_bars(self) -> None:
        self.online_state_label.setText("ROS-style sample coverage: collect diverse X, Y, scale, and skew.")
        defaults = {
            "x": "X: --",
            "y": "Y: --",
            "scale": "Scale: --",
            "skew": "Skew: --",
        }
        for key, bar in self.online_bars.items():
            bar.setValue(0)
            bar.setFormat("--")
            bar.setStyleSheet("")
            self.online_labels[key].setText(defaults[key])

    def _update_ros_coverage_bars(self) -> None:
        mode = "stereo" if self._mode() == "stereo" else "mono"
        params = self._existing_ros_params(mode)
        progress = ros_sample_progress(params)
        count = len(params)
        ready = "ready" if progress["goodenough"] else "not ready"
        self.online_state_label.setText(f"ROS-style sample coverage: {count} samples, {ready}.")
        key_map = {"X": "x", "Y": "y", "Scale": "scale", "Skew": "skew"}
        for item in progress["items"]:
            self._set_online_bar(key_map[item["name"]], item["name"], item["progress"], item["min"], item["max"])

    def _set_online_bar(self, key: str, title: str, progress: float, min_value: float, max_value: float) -> None:
        bar = self.online_bars[key]
        label = self.online_labels[key]
        value = min(100, max(0, int(round(progress * 100))))
        text = f"{value}% ({min_value:.2f}-{max_value:.2f})"
        bar.setValue(value)
        bar.setFormat(text)
        label.setText(f"{title}: {text}")
        bar.setStyleSheet(
            "QProgressBar { border: 1px solid #9aa4aa; text-align: center; }"
            f"QProgressBar::chunk {{ background-color: {self._progress_colour(progress)}; }}"
        )

    def _progress_colour(self, progress: float) -> str:
        if progress >= 1.0:
            return "#93e2a2"
        if progress >= 0.75:
            return "#f1dc72"
        if progress >= 0.4:
            return "#f2ad6f"
        return "#ee7d7d"

    def _error_colour(self, value: float) -> str:
        if value < 0.3:
            return "#93e2a2"
        if value < 0.6:
            return "#f1dc72"
        if value < 1.0:
            return "#f2ad6f"
        return "#ee7d7d"

    def _show_stereo_result(self, result) -> None:
        vertical = result.metrics["rectified_vertical_px"]["summary"]
        sampson = result.metrics["sampson_px"]["summary"]
        self.depth_fx.setValue(float(result.P1[0, 0]))
        self.depth_baseline.setValue(result.baseline_mm)
        self.stereo_panel.setPlainText(
            "\n".join(
                [
                    "Success: yes",
                    f"Timestamp: {result.timestamp}",
                    f"Pairs used: {result.pairs_used}",
                    f"OpenCV stereo RMS: {result.rms_px:.4f} px",
                    f"Baseline ||T||: {result.baseline_mm:.4f} mm",
                    f"R:\n{result.R}",
                    f"T:\n{result.T.reshape(-1)}",
                    f"Rectified |y_left-y_right| mean/median/P95/max: {vertical['mean']:.4f} / {vertical['median']:.4f} / {vertical['p95']:.4f} / {vertical['max']:.4f} px",
                    f"Sampson error mean/median/P95/max: {sampson['mean']:.4f} / {sampson['median']:.4f} / {sampson['p95']:.4f} / {sampson['max']:.4f} px",
                    f"Rectification ROIs: left {result.roi_left}, right {result.roi_right}",
                    f"Common valid area ratio: {result.metrics['common_valid_area_ratio']:.3f}",
                    "Mean vertical residual below 0.5 px is an initial target; 0.1-0.3 px is preferred.",
                    "Software pairing with monotonic timestamps is not equivalent to hardware synchronization.",
                ]
            )
        )
        rows = result.metrics["rectified_vertical_px"]["per_pair"]
        self._fill_table(rows)
        self._update_depth()
        self.error_tabs.setCurrentWidget(self.stereo_panel)
        self.status.setText("Stereo calibration complete.")

    def _fill_table(self, rows: list[dict]) -> None:
        self.per_view_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(["index", "rmse", "mean", "median", "p95", "max"]):
                value = row.get(key)
                self.per_view_table.setItem(r, c, QTableWidgetItem("" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)))

    def save_result(self) -> None:
        if self.session is None:
            self.status.setText("Press Create save dir before saving.")
            return
        if self._mode() == "mono" and self.mono_result:
            data = mono_to_dict(self.mono_result)
            path = self.session.root / f"mono_{self.mono_result.side}.yaml"
            atomic_write_yaml(path, data)
            atomic_write_json(self.session.root / "metrics.json", {"mono": data["metrics"]})
            write_per_view_csv(self.session.root / "per_view_errors.csv", data["metrics"]["per_view"])
            write_report(self.session.root / "calibration_report.md", "Monocular Calibration Report", data)
            self.status.setText(f"Saved {path}.")
        elif self._mode() == "stereo" and self.stereo_result:
            data = stereo_to_dict(self.stereo_result)
            path = self.session.root / "stereo.yaml"
            atomic_write_yaml(path, data)
            atomic_write_json(self.session.root / "metrics.json", {"stereo": data["metrics"]})
            write_per_view_csv(self.session.root / "per_view_errors.csv", data["metrics"]["rectified_vertical_px"]["per_pair"])
            write_report(self.session.root / "calibration_report.md", "Stereo Calibration Report", data)
            self.status.setText(f"Saved {path}.")
        else:
            self.status.setText("Run calibration before saving.")

    def load_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration YAML", self.output_dir.text(), "YAML files (*.yaml *.yml)")
        if not path:
            return
        try:
            data = load_yaml(Path(path))
            validate_image_size(data, (self.width_spin.value(), self.height_spin.value()))
            kind = data.get("kind")
            if kind == "mono":
                side = data.get("side", "left")
                board = board_from_dict(data.get("board", board_to_dict(self._current_board())))
                result = MonoCalibrationResult(
                    side=side,
                    image_size=tuple(data["image_size"]),
                    rms_px=float(data["rms_px"]),
                    camera_matrix=np.array(data["camera_matrix"], dtype=float),
                    dist_coeffs=np.array(data["distortion_coefficients"], dtype=float),
                    rvecs=[],
                    tvecs=[],
                    metrics=data.get("metrics", {}),
                    timestamp=data.get("timestamp", ""),
                    samples_used=int(data.get("samples_used", 0)),
                    board=board,
                )
                if side == "left":
                    self.mono_left_result = result
                else:
                    self.mono_right_result = result
                self.mono_result = result
                self._show_mono_result(result)
            else:
                self.status.setText("Loaded stereo YAML metadata; full stereo preview reload is available after rerunning calibration in this session.")
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", str(exc))

    def _toggle_rectification(self) -> None:
        if self.auto_capture:
            self.auto_checkbox.setChecked(False)
            self.auto_capture = False
        if self.stereo_result is None:
            self.status.setText("Run stereo calibration before rectification preview.")
            return
        self.rectification_preview = not self.rectification_preview
        self.rectify_button.setText("Stop rectification" if self.rectification_preview else "Preview rectification")

    def _escape(self) -> None:
        self.auto_capture = False
        self.rectification_preview = False
        self.auto_checkbox.setChecked(False)
        self.rectify_button.setText("Preview rectification")
        self.status.setText("Stopped auto-capture/rectification preview.")

    def reset_session(self) -> None:
        self.session = None
        self.mono_panel.setPlainText("")
        self.stereo_panel.setPlainText("")
        self.per_view_table.setRowCount(0)
        self._reset_online_bars()
        self.status.setText("Session cleared. Press Create save dir before capturing.")
        self._update_buttons()

    def _update_depth(self) -> None:
        try:
            noises = [float(item.strip()) for item in self.depth_noise.text().split(",") if item.strip()]
            parts = []
            for sigma_d in noises:
                sigma_z = depth_uncertainty(self.depth_z.value(), self.depth_fx.value(), self.depth_baseline.value(), sigma_d)
                parts.append(f"sigma_d={sigma_d:g}px -> sigma_Z={sigma_z:.3f} mm")
            self.depth_output.setText("; ".join(parts) + ". Theoretical estimate, not measured depth accuracy.")
        except ValueError:
            self.depth_output.setText("Enter comma-separated numeric disparity errors.")

    def _update_buttons(self) -> None:
        board_ready = self._current_board().complete
        session_ready = self.session is not None
        camera_ready = bool(self.workers)
        reason = ""
        if not board_ready:
            reason = self._current_board().missing_reason()
        elif not session_ready:
            reason = "Next step: press Create save dir. Capture stays disabled until the save directory exists."
        elif not camera_ready:
            reason = "Next step: press Open camera(s)."
        capture_enabled = board_ready and session_ready
        self.capture_button.setEnabled(capture_enabled)
        self.capture_button.setToolTip(reason)
        self.auto_checkbox.setEnabled(capture_enabled)
        self.auto_checkbox.setToolTip(reason)
        has_samples = bool(self.session and self.session.sample_count)
        self.delete_button.setEnabled(has_samples)
        self.calibrate_button.setEnabled(has_samples and board_ready)
        self.save_button.setEnabled(bool(self.mono_result or self.stereo_result))
        self.rectify_button.setEnabled(self._mode() == "stereo")
        self.right_view.setVisible(self._mode() == "stereo")
        self.side_combo.setEnabled(self._mode() == "mono")
        self.create_session_button.setEnabled(board_ready)
        self.create_session_button.setToolTip("" if board_ready else self._current_board().missing_reason())
        if self.auto_capture:
            self.auto_checkbox.setStyleSheet("QCheckBox { background: #c8f7d0; padding: 4px; border-radius: 4px; }")
        else:
            self.auto_checkbox.setStyleSheet("")
        if capture_enabled and self.workers:
            self.capture_button.setStyleSheet("QPushButton { background: #c8f7d0; }")
        elif capture_enabled:
            self.capture_button.setStyleSheet("QPushButton { background: #e8f7eb; }")
        else:
            self.capture_button.setStyleSheet("")

    def closeEvent(self, event) -> None:
        self.stop_cameras()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
