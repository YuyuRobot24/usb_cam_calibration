from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .models import Sample, StereoSample


@dataclass
class CalibrationSession:
    root: Path
    mode: str
    mono_side: str = "left"
    mono_samples: list[Sample] = field(default_factory=list)
    stereo_pairs: list[StereoSample] = field(default_factory=list)
    removed_ids: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, output_root: str | Path, mode: str, mono_side: str = "left") -> "CalibrationSession":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(output_root) / f"session_{stamp}"
        for sub in ["accepted/left", "accepted/right", "rejected", "overlays"]:
            (root / sub).mkdir(parents=True, exist_ok=True)
        return cls(root=root, mode=mode, mono_side=mono_side)

    @property
    def sample_count(self) -> int:
        return len(self.stereo_pairs) if self.mode == "stereo" else len(self.mono_samples)

    def add_mono(self, sample: Sample) -> None:
        self.mono_samples.append(sample)

    def add_pair(self, pair: StereoSample) -> None:
        self.stereo_pairs.append(pair)

    def delete_last(self) -> str | None:
        if self.mode == "stereo" and self.stereo_pairs:
            pair = self.stereo_pairs.pop()
            self.removed_ids.append(pair.sample_id)
            return pair.sample_id
        if self.mode != "stereo" and self.mono_samples:
            sample = self.mono_samples.pop()
            self.removed_ids.append(sample.sample_id)
            return sample.sample_id
        return None

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "mono_side": self.mono_side,
            "root": str(self.root),
            "sample_count": self.sample_count,
            "removed_ids": self.removed_ids,
        }

