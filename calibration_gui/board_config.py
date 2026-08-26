from __future__ import annotations

from pathlib import Path

import yaml

from .models import BoardConfig


def load_board_presets(path: str | Path) -> list[BoardConfig]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [BoardConfig(**item) for item in data.get("boards", [])]


def board_from_dict(data: dict) -> BoardConfig:
    return BoardConfig(**data)


def board_to_dict(board: BoardConfig) -> dict:
    return {
        "preset_name": board.preset_name,
        "inner_corners_cols": board.inner_corners_cols,
        "inner_corners_rows": board.inner_corners_rows,
        "square_size_mm": board.square_size_mm,
        "notes": board.notes,
        "manufacturer": board.manufacturer,
        "measured_total_length_mm": board.measured_total_length_mm,
    }

