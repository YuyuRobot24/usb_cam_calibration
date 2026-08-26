from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .camera_backend import CameraFrame, UVCCamera


class CameraWorker(QObject):
    frame_ready = Signal(object)
    error = Signal(str)
    stopped = Signal()

    def __init__(self, index: int, width: int, height: int, fps: int):
        super().__init__()
        self.camera = UVCCamera(index, width, height, fps)
        self._running = False

    @Slot()
    def run(self) -> None:
        try:
            self.camera.open()
            self._running = True
            sleep_ms = max(1, int(1000 / max(1, self.camera.fps)))
            while self._running:
                frame: CameraFrame = self.camera.read()
                self.frame_ready.emit(frame)
                QThread.msleep(sleep_ms)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.camera.release()
            self.stopped.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False
