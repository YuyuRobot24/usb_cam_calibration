import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from calibration_gui.main_window import MainWindow


def test_qt_widget_construction():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "Stereo Camera Calibration"
    assert window.session is None
    assert window.capture_button.isEnabled() is False
    assert set(window.online_bars) == {"x", "y", "scale", "skew"}
    assert window.auto_checkbox.isChecked() is True


def test_capture_requires_user_created_save_dir():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.cols_spin.setValue(8)
    window.rows_spin.setValue(6)
    window.square_spin.setValue(1.0)
    assert window._current_board().complete
    assert window.capture_button.isEnabled() is False
