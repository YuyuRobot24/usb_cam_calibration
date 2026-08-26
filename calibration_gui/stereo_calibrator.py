from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np

from .metrics import common_roi_ratio, rectified_vertical_errors, sampson_errors
from .models import BoardConfig, MonoCalibrationResult, StereoCalibrationResult, StereoSample


def calibrate_stereo(
    pairs: list[StereoSample],
    board: BoardConfig,
    image_size: tuple[int, int],
    left_intrinsics: MonoCalibrationResult | None = None,
    right_intrinsics: MonoCalibrationResult | None = None,
    fix_intrinsic: bool = True,
    rectify_alpha: float = 0.0,
) -> StereoCalibrationResult:
    if not board.complete:
        raise ValueError(board.missing_reason())
    if len(pairs) < 3:
        raise ValueError("At least 3 valid stereo pairs are required; 20+ diverse pairs are recommended.")
    objp = board.object_points()
    object_points = [objp.copy() for _ in pairs]
    left_points = [pair.left.corners.astype(np.float32) for pair in pairs]
    right_points = [pair.right.corners.astype(np.float32) for pair in pairs]
    if left_intrinsics and right_intrinsics:
        K1 = left_intrinsics.camera_matrix.copy()
        D1 = left_intrinsics.dist_coeffs.copy()
        K2 = right_intrinsics.camera_matrix.copy()
        D2 = right_intrinsics.dist_coeffs.copy()
        flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsic else 0
    else:
        K1 = D1 = K2 = D2 = None
        flags = 0
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        K1,
        D1,
        K2,
        D2,
        image_size,
        flags=flags,
    )
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=rectify_alpha)
    vertical = rectified_vertical_errors(left_points, right_points, K1, D1, K2, D2, R1, R2, P1, P2)
    epipolar = sampson_errors(left_points, right_points, F)
    baseline = float(np.linalg.norm(T))
    metrics = {
        "rectified_vertical_px": vertical,
        "sampson_px": epipolar,
        "common_valid_area_ratio": common_roi_ratio(roi_left, roi_right, image_size),
        "pair_timestamp_delta_ms": [pair.timestamp_delta_ms for pair in pairs],
    }
    return StereoCalibrationResult(
        image_size=image_size,
        rms_px=float(rms),
        camera_matrix_left=K1,
        dist_coeffs_left=D1,
        camera_matrix_right=K2,
        dist_coeffs_right=D2,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi_left=tuple(int(v) for v in roi_left),
        roi_right=tuple(int(v) for v in roi_right),
        metrics=metrics,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        pairs_used=len(pairs),
        board=board,
        baseline_mm=baseline,
    )

