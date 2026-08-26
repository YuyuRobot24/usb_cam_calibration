from __future__ import annotations

import cv2
import numpy as np

from .models import BoardConfig, DetectionResult


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_chessboard(frame_bgr: np.ndarray, board: BoardConfig) -> DetectionResult:
    if not board.complete:
        return DetectionResult(False, None, 0.0, 0.0, 0.0, [board.missing_reason()])
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    score = blur_score(gray)
    pattern = board.pattern_size
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    if not found or corners is None:
        return DetectionResult(False, None, score, 0.0, 0.0, ["Board not detected."])
    pts = corners.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
    area_ratio = float((w * h) / (frame_bgr.shape[1] * frame_bgr.shape[0]))
    margin = min(float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(frame_bgr.shape[1] - np.max(pts[:, 0])), float(frame_bgr.shape[0] - np.max(pts[:, 1])))
    warnings = []
    if score < 80:
        warnings.append("Blur warning.")
    if area_ratio < 0.01:
        warnings.append("Board appears small.")
    if margin < 8:
        warnings.append("Corners are close to image boundary.")
    return DetectionResult(True, corners.astype(np.float32), score, area_ratio, margin, warnings)


def draw_overlay(frame_bgr: np.ndarray, board: BoardConfig, detection: DetectionResult, label: str) -> np.ndarray:
    overlay = frame_bgr.copy()
    if detection.found and detection.corners is not None:
        cv2.drawChessboardCorners(overlay, board.pattern_size, detection.corners, True)
        for idx, point in enumerate(detection.corners.reshape(-1, 2)):
            if idx in (0, len(detection.corners) - 1) or idx % max(1, len(detection.corners) // 8) == 0:
                cv2.putText(overlay, str(idx), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    state = "FOUND" if detection.found else "MISSING"
    cv2.putText(overlay, f"{label} {state} blur {detection.blur_score:.0f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40) if detection.found else (40, 40, 255), 2, cv2.LINE_AA)
    return overlay

