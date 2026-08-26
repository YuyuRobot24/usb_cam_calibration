from __future__ import annotations

import time
from dataclasses import dataclass

import cv2


@dataclass
class CameraFrame:
    index: int
    frame_bgr: object
    timestamp_ns: int
    measured_fps: float
    actual_size: tuple[int, int]


class UVCCamera:
    def __init__(self, index: int, width: int, height: int, fps: int):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: cv2.VideoCapture | None = None
        self._last_ts: int | None = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            raise OSError(f"Could not open /dev/video{self.index}. Check index and permissions.")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> CameraFrame:
        if self.capture is None:
            raise RuntimeError("Camera is not open.")
        ok, frame = self.capture.read()
        ts = time.monotonic_ns()
        if not ok or frame is None:
            raise OSError(f"Camera {self.index} did not return a frame.")
        dt = (ts - self._last_ts) / 1e9 if self._last_ts else 0
        self._last_ts = ts
        measured_fps = 1.0 / dt if dt > 0 else 0.0
        return CameraFrame(self.index, frame, ts, measured_fps, (frame.shape[1], frame.shape[0]))

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def probe_camera_indices(max_index: int = 8) -> list[int]:
    found: list[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                found.append(index)
        finally:
            cap.release()
    return found
