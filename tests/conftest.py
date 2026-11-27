from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def _reset_qt_keyboard_modifiers():
    yield
    app = QApplication.instance()
    if app is None:
        return
    mods = QApplication.keyboardModifiers()
    if mods == Qt.KeyboardModifier.NoModifier:
        return
    target = QApplication.focusWidget() or QApplication.activeWindow()
    if target is None:
        widgets = QApplication.topLevelWidgets()
        target = widgets[0] if widgets else None
    if target is None:
        return
    for flag, key in (
        (Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Control),
        (Qt.KeyboardModifier.ShiftModifier, Qt.Key.Key_Shift),
        (Qt.KeyboardModifier.AltModifier, Qt.Key.Key_Alt),
    ):
        if mods & flag:
            QApplication.sendEvent(
                target,
                QKeyEvent(
                    QKeyEvent.Type.KeyRelease,
                    key,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
