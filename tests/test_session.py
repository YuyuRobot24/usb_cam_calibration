from calibration_gui.models import Sample, StereoSample
from calibration_gui.session import CalibrationSession


def test_stereo_pair_deletion_preserves_correspondence(tmp_path):
    session = CalibrationSession.create(tmp_path, "stereo")
    left = Sample("0001", "left", tmp_path / "l.png", 1, corners=[], blur_score=1, board_area_ratio=1)
    right = Sample("0001", "right", tmp_path / "r.png", 2, corners=[], blur_score=1, board_area_ratio=1)
    session.add_pair(StereoSample("0001", left, right, 0.001))
    removed = session.delete_last()
    assert removed == "0001"
    assert session.stereo_pairs == []
    assert session.removed_ids == ["0001"]
