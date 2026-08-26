from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np
import yaml

from .board_config import board_to_dict
from .models import MonoCalibrationResult, StereoCalibrationResult


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def atomic_write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        yaml.safe_dump(plain(data), tmp, sort_keys=False)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(plain(data), tmp, indent=2)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def save_image(path: Path, frame_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame_bgr):
        raise OSError(f"Could not save image: {path}")


def mono_to_dict(result: MonoCalibrationResult) -> dict:
    return {
        "schema_version": 1,
        "kind": "mono",
        "side": result.side,
        "timestamp": result.timestamp,
        "samples_used": result.samples_used,
        "image_size": list(result.image_size),
        "board": board_to_dict(result.board),
        "model": result.model,
        "rms_px": result.rms_px,
        "camera_matrix": result.camera_matrix,
        "distortion_coefficients": result.dist_coeffs,
        "metrics": result.metrics,
    }


def stereo_to_dict(result: StereoCalibrationResult) -> dict:
    return {
        "schema_version": 1,
        "kind": "stereo",
        "timestamp": result.timestamp,
        "pairs_used": result.pairs_used,
        "image_size": list(result.image_size),
        "board": board_to_dict(result.board),
        "rms_px": result.rms_px,
        "baseline_mm": result.baseline_mm,
        "camera_matrix_left": result.camera_matrix_left,
        "distortion_coefficients_left": result.dist_coeffs_left,
        "camera_matrix_right": result.camera_matrix_right,
        "distortion_coefficients_right": result.dist_coeffs_right,
        "R": result.R,
        "T": result.T,
        "E": result.E,
        "F": result.F,
        "R1": result.R1,
        "R2": result.R2,
        "P1": result.P1,
        "P2": result.P2,
        "Q": result.Q,
        "roi_left": result.roi_left,
        "roi_right": result.roi_right,
        "metrics": result.metrics,
    }


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def validate_image_size(calibration: dict, image_size: tuple[int, int]) -> None:
    saved = tuple(calibration.get("image_size", []))
    if saved != tuple(image_size):
        raise ValueError(f"Calibration image size {saved} does not match current image size {image_size}.")


def write_per_view_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "rmse", "mean", "median", "p95", "max", "corner_count"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_report(path: Path, title: str, data: dict) -> None:
    lines = [f"# {title}", "", "These thresholds are empirical guidance, not universal pass/fail limits.", ""]
    for key, value in plain(data).items():
        lines.append(f"## {key}")
        lines.append("")
        if isinstance(value, (dict, list)):
            lines.append("```yaml")
            lines.append(yaml.safe_dump(value, sort_keys=False).strip())
            lines.append("```")
        else:
            lines.append(str(value))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

