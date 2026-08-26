import numpy as np
import pytest

from calibration_gui.storage import atomic_write_yaml, load_yaml, validate_image_size


def test_yaml_round_trip_preserves_plain_values(tmp_path):
    path = tmp_path / "calibration.yaml"
    data = {"schema_version": 1, "matrix": np.eye(3), "shape": (1280, 720)}
    atomic_write_yaml(path, data)
    loaded = load_yaml(path)
    assert loaded["matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert loaded["shape"] == [1280, 720]


def test_mismatched_image_dimensions_are_rejected():
    with pytest.raises(ValueError):
        validate_image_size({"image_size": [640, 480]}, (1280, 720))
