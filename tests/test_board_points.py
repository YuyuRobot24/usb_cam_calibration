import numpy as np

from calibration_gui.models import BoardConfig


def test_board_points_count_order_and_scaling():
    board = BoardConfig("test", inner_corners_cols=3, inner_corners_rows=2, square_size_mm=2.5)
    points = board.object_points()
    assert points.shape == (6, 3)
    np.testing.assert_allclose(points[0], [0, 0, 0])
    np.testing.assert_allclose(points[1], [2.5, 0, 0])
    np.testing.assert_allclose(points[2], [5.0, 0, 0])
    np.testing.assert_allclose(points[3], [0, 2.5, 0])


def test_incomplete_board_reason_mentions_internal_corners():
    board = BoardConfig("blank")
    assert not board.complete
    assert "internal corners" in board.missing_reason()
