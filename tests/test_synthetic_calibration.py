import cv2
import numpy as np

from calibration_gui.metrics import reprojection_metrics
from calibration_gui.models import BoardConfig, Sample
from calibration_gui.mono_calibrator import calibrate_mono


def _synthetic_samples(noise_px=0.05):
    rng = np.random.default_rng(4)
    board = BoardConfig("synthetic", 8, 6, 5.0)
    objp = board.object_points()
    K = np.array([[700.0, 0, 640.0], [0, 705.0, 360.0], [0, 0, 1]], dtype=float)
    D = np.zeros((5, 1), dtype=float)
    samples = []
    for idx in range(12):
        rvec = np.array([0.05 * (idx % 4), -0.04 * (idx % 3), 0.03 * idx], dtype=float)
        tvec = np.array([-40 + 8 * idx, -20 + 4 * (idx % 5), 300 + 12 * idx], dtype=float)
        points, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        points = points.astype(np.float32)
        points += rng.normal(0, noise_px, points.shape).astype(np.float32)
        samples.append(Sample(f"{idx:04d}", "left", f"{idx}.png", idx, points, 100.0, 0.2))
    return board, samples, K


def test_synthetic_noisy_projections_recover_camera_parameters():
    board, samples, expected_K = _synthetic_samples()
    result = calibrate_mono(samples, board, (1280, 720), "left")
    assert abs(result.camera_matrix[0, 0] - expected_K[0, 0]) < 45
    assert abs(result.camera_matrix[1, 1] - expected_K[1, 1]) < 45
    assert result.metrics["all_corner"]["rmse"] < 0.2


def test_reprojection_error_aggregation_units():
    board, samples, _ = _synthetic_samples(noise_px=0.0)
    result = calibrate_mono(samples, board, (1280, 720), "left")
    metrics = reprojection_metrics(
        [board.object_points() for _ in samples],
        [sample.corners for sample in samples],
        result.rvecs,
        result.tvecs,
        result.camera_matrix,
        result.dist_coeffs,
    )
    assert metrics["all_corner"]["rmse"] < 1e-3
