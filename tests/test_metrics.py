import numpy as np

from calibration_gui.metrics import (
    common_roi_ratio,
    depth_uncertainty,
    rectified_vertical_errors,
    ros_is_good_sample,
    ros_sample_parameters,
    ros_sample_progress,
    sampson_errors,
)


def test_rectified_vertical_error_uses_undistort_points_identity():
    left = [np.array([[[10.0, 20.0]], [[30.0, 40.0]]], dtype=np.float32)]
    right = [np.array([[[12.0, 21.5]], [[31.0, 42.0]]], dtype=np.float32)]
    K = np.eye(3, dtype=float)
    D = np.zeros((5, 1), dtype=float)
    result = rectified_vertical_errors(left, right, K, D, K, D, K, K, K, K)
    assert result["summary"]["mean"] == 1.75
    assert result["summary"]["max"] == 2.0


def test_sampson_error_near_zero_for_perfect_correspondences():
    F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    left = [np.array([[[10.0, 4.0]], [[20.0, 8.0]]], dtype=np.float32)]
    right = [np.array([[[11.0, 4.0]], [[21.0, 8.0]]], dtype=np.float32)]
    result = sampson_errors(left, right, F)
    assert result["summary"]["max"] < 1e-9


def test_common_roi_ratio_and_depth_uncertainty():
    assert common_roi_ratio((0, 0, 100, 100), (50, 50, 100, 100), (200, 200)) == 0.0625
    assert depth_uncertainty(100, 500, 20, 0.5) == 0.5


def test_ros_sample_parameters_and_selection_progress():
    corners = np.array(
        [
            [[10.0, 10.0]],
            [[30.0, 10.0]],
            [[50.0, 10.0]],
            [[10.0, 30.0]],
            [[30.0, 30.0]],
            [[50.0, 30.0]],
        ],
        dtype=np.float32,
    )
    params = ros_sample_parameters(corners, (100, 80), cols=3)
    assert len(params) == 4
    assert all(0.0 <= value <= 1.0 for value in params)
    assert ros_is_good_sample(params, [])
    assert not ros_is_good_sample(params, [params])
    progress = ros_sample_progress([params, [0.9, 0.9, 0.5, 0.5]])
    assert [item["name"] for item in progress["items"]] == ["X", "Y", "Scale", "Skew"]
    assert progress["items"][0]["progress"] > 0
