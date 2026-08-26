from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BoardConfig:
    preset_name: str
    inner_corners_cols: int | None = None
    inner_corners_rows: int | None = None
    square_size_mm: float | None = None
    notes: str = ""
    manufacturer: str = ""
    measured_total_length_mm: float | None = None

    @property
    def complete(self) -> bool:
        return (
            self.inner_corners_cols is not None
            and self.inner_corners_rows is not None
            and self.square_size_mm is not None
            and self.inner_corners_cols > 1
            and self.inner_corners_rows > 1
            and self.square_size_mm > 0
        )

    @property
    def pattern_size(self) -> tuple[int, int]:
        if not self.complete:
            raise ValueError("Board requires internal corner columns, rows, and square size.")
        return int(self.inner_corners_cols), int(self.inner_corners_rows)

    def object_points(self) -> np.ndarray:
        cols, rows = self.pattern_size
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= float(self.square_size_mm)
        return objp

    def missing_reason(self) -> str:
        missing = []
        if self.inner_corners_cols is None or self.inner_corners_cols <= 1:
            missing.append("internal corner columns")
        if self.inner_corners_rows is None or self.inner_corners_rows <= 1:
            missing.append("internal corner rows")
        if self.square_size_mm is None or self.square_size_mm <= 0:
            missing.append("square size in mm")
        if missing:
            return "Enter " + ", ".join(missing) + " (internal corners, not the number of black/white squares)."
        return ""


@dataclass
class DetectionResult:
    found: bool
    corners: np.ndarray | None
    blur_score: float
    board_area_ratio: float
    boundary_margin_px: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class Sample:
    sample_id: str
    side: str
    image_path: Path
    timestamp_ns: int
    corners: np.ndarray
    blur_score: float
    board_area_ratio: float
    ros_params: list[float] = field(default_factory=list)
    accepted: bool = True
    rejection_reason: str = ""


@dataclass
class StereoSample:
    sample_id: str
    left: Sample
    right: Sample
    timestamp_delta_ms: float


@dataclass
class MonoCalibrationResult:
    side: str
    image_size: tuple[int, int]
    rms_px: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    metrics: dict[str, Any]
    timestamp: str
    samples_used: int
    board: BoardConfig
    model: str = "pinhole"


@dataclass
class StereoCalibrationResult:
    image_size: tuple[int, int]
    rms_px: float
    camera_matrix_left: np.ndarray
    dist_coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi_left: tuple[int, int, int, int]
    roi_right: tuple[int, int, int, int]
    metrics: dict[str, Any]
    timestamp: str
    pairs_used: int
    board: BoardConfig
    baseline_mm: float
