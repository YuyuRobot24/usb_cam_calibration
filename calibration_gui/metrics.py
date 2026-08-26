from __future__ import annotations

import numpy as np
import cv2


ROS_PARAM_RANGES = [0.7, 0.7, 0.4, 0.5]


def ros_outside_corners(corners: np.ndarray, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = corners.reshape(-1, 2)
    return pts[0], pts[cols - 1], pts[-1], pts[-cols]


def ros_quadrilateral_area(outside_corners: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
    up_left, up_right, down_right, down_left = outside_corners
    a = up_right - up_left
    b = down_right - up_right
    c = down_left - down_right
    p = b + c
    q = a + b
    return float(abs(p[0] * q[1] - p[1] * q[0]) / 2.0)


def ros_skew(outside_corners: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
    up_left, up_right, down_right, _ = outside_corners
    ab = up_left - up_right
    cb = down_right - up_right
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom <= 0:
        return 1.0
    cos_angle = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    return float(min(1.0, 2.0 * abs((np.pi / 2.0) - angle)))


def ros_sample_parameters(corners: np.ndarray, image_size: tuple[int, int], cols: int) -> list[float]:
    width, height = image_size
    pts = corners.reshape(-1, 2)
    outside = ros_outside_corners(corners, cols)
    area = ros_quadrilateral_area(outside)
    skew = ros_skew(outside)
    border = float(np.sqrt(area))
    usable_width = max(1.0, width - border)
    usable_height = max(1.0, height - border)
    p_x = min(1.0, max(0.0, (float(np.mean(pts[:, 0])) - border / 2.0) / usable_width))
    p_y = min(1.0, max(0.0, (float(np.mean(pts[:, 1])) - border / 2.0) / usable_height))
    p_size = float(np.sqrt(max(0.0, area) / (width * height)))
    return [p_x, p_y, p_size, skew]


def ros_param_distance(params_a: list[float], params_b: list[float]) -> float:
    return float(sum(abs(a - b) for a, b in zip(params_a, params_b)))


def ros_is_good_sample(params: list[float], existing_params: list[list[float]], threshold: float = 0.2) -> bool:
    if not existing_params:
        return True
    return min(ros_param_distance(params, old) for old in existing_params) > threshold


def ros_sample_progress(existing_params: list[list[float]]) -> dict:
    names = ["X", "Y", "Scale", "Skew"]
    if not existing_params:
        return {
            "goodenough": False,
            "items": [
                {"name": name, "min": 0.0, "max": 0.0, "progress": 0.0}
                for name in names
            ],
        }
    values = np.asarray(existing_params, dtype=float)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    mins[2] = 0.0
    mins[3] = 0.0
    progress = np.minimum((maxs - mins) / np.asarray(ROS_PARAM_RANGES), 1.0)
    return {
        "goodenough": bool(len(existing_params) >= 40 or np.all(progress == 1.0)),
        "items": [
            {"name": name, "min": float(lo), "max": float(hi), "progress": float(p)}
            for name, lo, hi, p in zip(names, mins, maxs, progress)
        ],
    }


def describe_errors(errors: np.ndarray) -> dict:
    values = np.asarray(errors, dtype=float).reshape(-1)
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None, "rmse": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }


def reprojection_metrics(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict:
    per_view = []
    all_errors = []
    all_axis_errors = []
    for idx, (objp, imgp, rvec, tvec) in enumerate(zip(object_points, image_points, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        measured = imgp.reshape(-1, 2)
        axis_errors = np.abs(projected - measured)
        errors = np.linalg.norm(projected - measured, axis=1)
        all_errors.append(errors)
        all_axis_errors.append(axis_errors)
        summary = describe_errors(errors)
        summary["x_rmse"] = float(np.sqrt(np.mean(axis_errors[:, 0] ** 2)))
        summary["y_rmse"] = float(np.sqrt(np.mean(axis_errors[:, 1] ** 2)))
        summary["index"] = idx
        summary["corner_count"] = int(errors.size)
        per_view.append(summary)
    all_errors_array = np.concatenate(all_errors) if all_errors else np.array([], dtype=float)
    all_axis_array = np.vstack(all_axis_errors) if all_axis_errors else np.empty((0, 2), dtype=float)
    axis_summary = {
        "x": describe_errors(all_axis_array[:, 0] if all_axis_array.size else np.array([], dtype=float)),
        "y": describe_errors(all_axis_array[:, 1] if all_axis_array.size else np.array([], dtype=float)),
    }
    return {
        "all_corner": describe_errors(all_errors_array),
        "axis": axis_summary,
        "per_view": per_view,
        "worst_views": sorted(per_view, key=lambda row: row["rmse"] or 0, reverse=True)[:5],
    }


def rectified_vertical_errors(
    points_left: list[np.ndarray],
    points_right: list[np.ndarray],
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> dict:
    per_pair = []
    all_errors = []
    for idx, (left, right) in enumerate(zip(points_left, points_right)):
        ul = cv2.undistortPoints(left.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
        ur = cv2.undistortPoints(right.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)
        errors = np.abs(ul[:, 1] - ur[:, 1])
        all_errors.append(errors)
        row = describe_errors(errors)
        row["index"] = idx
        per_pair.append(row)
    values = np.concatenate(all_errors) if all_errors else np.array([], dtype=float)
    return {
        "summary": describe_errors(values),
        "per_pair": per_pair,
        "worst_pairs": sorted(per_pair, key=lambda row: row["mean"] or 0, reverse=True)[:5],
    }


def sampson_errors(points_left: list[np.ndarray], points_right: list[np.ndarray], F: np.ndarray) -> dict:
    per_pair = []
    all_errors = []
    for idx, (left, right) in enumerate(zip(points_left, points_right)):
        pts1 = left.reshape(-1, 2).astype(float)
        pts2 = right.reshape(-1, 2).astype(float)
        ones = np.ones((pts1.shape[0], 1))
        x1 = np.hstack([pts1, ones])
        x2 = np.hstack([pts2, ones])
        Fx1 = (F @ x1.T).T
        Ftx2 = (F.T @ x2.T).T
        x2tFx1 = np.sum(x2 * Fx1, axis=1)
        denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
        errors = np.sqrt(np.divide(x2tFx1**2, denom, out=np.zeros_like(x2tFx1), where=denom > 0))
        all_errors.append(errors)
        row = describe_errors(errors)
        row["index"] = idx
        per_pair.append(row)
    values = np.concatenate(all_errors) if all_errors else np.array([], dtype=float)
    return {"summary": describe_errors(values), "per_pair": per_pair}


def sanity_warnings(camera_matrix: np.ndarray, image_size: tuple[int, int]) -> list[str]:
    width, height = image_size
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    warnings = []
    if not (0 <= cx <= width and 0 <= cy <= height):
        warnings.append("Principal point is outside the image.")
    ratio = fx / fy if fy else np.inf
    if ratio < 0.8 or ratio > 1.25:
        warnings.append(f"Unusual fx/fy ratio: {ratio:.3f}.")
    return warnings


def common_roi_ratio(roi_left: tuple[int, int, int, int], roi_right: tuple[int, int, int, int], image_size: tuple[int, int]) -> float:
    lx, ly, lw, lh = roi_left
    rx, ry, rw, rh = roi_right
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return float(area / (image_size[0] * image_size[1]))


def depth_uncertainty(z_mm: float, fx_px: float, baseline_mm: float, disparity_error_px: float) -> float:
    if fx_px <= 0 or baseline_mm <= 0:
        return float("nan")
    return float((z_mm**2 / (fx_px * baseline_mm)) * disparity_error_px)
