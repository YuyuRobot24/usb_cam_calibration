from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np

from .metrics import reprojection_metrics, sanity_warnings
from .models import BoardConfig, MonoCalibrationResult, Sample


def calibrate_mono(samples: list[Sample], board: BoardConfig, image_size: tuple[int, int], side: str) -> MonoCalibrationResult:
    if not board.complete:
        raise ValueError(board.missing_reason())
    if len(samples) < 3:
        raise ValueError("At least 3 valid samples are required for calibration; 20+ diverse samples are recommended.")
    objp = board.object_points()
    object_points = [objp.copy() for _ in samples]
    image_points = [sample.corners.astype(np.float32) for sample in samples]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    metrics = reprojection_metrics(object_points, image_points, rvecs, tvecs, K, D)
    metrics["sanity_warnings"] = sanity_warnings(K, image_size)
    return MonoCalibrationResult(
        side=side,
        image_size=image_size,
        rms_px=float(rms),
        camera_matrix=K,
        dist_coeffs=D,
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        metrics=metrics,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        samples_used=len(samples),
        board=board,
    )

