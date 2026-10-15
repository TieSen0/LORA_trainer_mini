"""Shared desktop styling and presentation-only widgets."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSizePolicy, QVBoxLayout


def apply_theme():
    app = QApplication.instance()
    if app is None or app.property("miniLoraTheme"):
        return
    app.setProperty("miniLoraTheme", True)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    palette = QPalette()
    for role, color in (
        (QPalette.Window, "#11151d"), (QPalette.WindowText, "#e5ebf5"),
        (QPalette.Base, "#151c27"), (QPalette.AlternateBase, "#202a3a"),
        (QPalette.Text, "#e5ebf5"), (QPalette.Button, "#222d3e"),
        (QPalette.ButtonText, "#e5ebf5"), (QPalette.Highlight, "#365fd1"),
        (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#273754"),
        (QPalette.ToolTipText, "#e5ebf5"), (QPalette.PlaceholderText, "#91a0b6"),
        (QPalette.Light, "#46556e"), (QPalette.Midlight, "#364359"),
        (QPalette.Mid, "#273347"), (QPalette.Dark, "#0d1118"),
        (QPalette.Shadow, "#090c12"), (QPalette.Link, "#8eb1ff"),
        (QPalette.LinkVisited, "#bca7ef"),
    ):
        palette.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText, QPalette.PlaceholderText):
        palette.setColor(QPalette.Disabled, role, QColor("#78859a"))
    app.setPalette(palette)
    app.setStyleSheet("""
        QWidget { color: #e5ebf5; }
        QWidget:disabled { color: #78859a; }
        QLabel[role="title"] { font-size: 24px; font-weight: 600; }
        QLabel[role="muted"] { color: #a3b0c5; }
        QLabel[role="section"] { font-size: 14px; font-weight: 600; }
        QLabel#imagePreview { background: #0b0f16; color: #a3b0c5; border-radius: 12px; padding: 12px; }
        QPushButton { background: #222d3e; border: 1px solid #364359; border-radius: 6px; padding: 7px 12px; min-height: 18px; }
        QPushButton:hover { background: #273754; border-color: #709bff; }
        QPushButton:pressed { background: #30476b; }
        QPushButton:focus { border: 2px solid #709bff; padding: 6px 11px; }
        QPushButton[role="primary"] { background: #365fd1; color: white; border-color: #709bff; font-weight: 600; }
        QPushButton[role="primary"]:hover { background: #416dde; }
        QPushButton[role="danger"] { color: #ffacb3; border-color: #885361; }
        QPushButton:disabled, QPushButton[role="primary"]:disabled, QPushButton[role="danger"]:disabled {
            color: #78859a; background: #1a2230; border-color: #2b3547;
        }
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #151c27; border: 1px solid #364359; border-radius: 6px; padding: 6px;
            selection-background-color: #365fd1; selection-color: white;
            placeholder-text-color: #91a0b6;
        }
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus,
        QDoubleSpinBox:focus, QComboBox:focus { border-color: #709bff; }
        QLineEdit:read-only { background: #1b2230; color: #a3b0c5; }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
        QTextEdit:disabled, QPlainTextEdit:disabled { color: #78859a; background: #1a2230; }
        QComboBox { padding-right: 24px; }
        QComboBox QAbstractItemView { background: #1b2230; color: #e5ebf5; selection-background-color: #365fd1; selection-color: white; }
        QSpinBox, QDoubleSpinBox { padding-right: 26px; }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border; subcontrol-position: top right;
            width: 22px; background: #222d3e; border-left: 1px solid #364359;
            border-top-right-radius: 6px;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border; subcontrol-position: bottom right;
            width: 22px; background: #222d3e; border-left: 1px solid #364359;
            border-bottom-right-radius: 6px;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #30476b; }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: url("$ICON_DIR/chevron-up.svg"); width: 10px; height: 8px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: url("$ICON_DIR/chevron-down.svg"); width: 10px; height: 8px;
        }
        QGroupBox { background: #1b2230; border: 1px solid #364359; border-radius: 9px; margin-top: 15px; padding: 16px 12px 12px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
        QGroupBox QLabel, QGroupBox QCheckBox { font-weight: normal; }
        QTabWidget::pane { border: 1px solid #364359; border-radius: 8px; background: #1b2230; }
        QTabBar::tab { background: transparent; color: #a3b0c5; padding: 10px 15px; border-bottom: 2px solid transparent; }
        QTabBar::tab:selected { color: #8eb1ff; border-bottom: 2px solid #709bff; font-weight: 600; }
        QTabBar::tab:hover { background: #273754; }
        QScrollArea { border: none; background: transparent; }
        QCheckBox { spacing: 8px; padding: 3px 0; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #617087; border-radius: 3px; background: #151c27; }
        QCheckBox::indicator:checked { background: #365fd1; border-color: #709bff; image: url("$ICON_DIR/check.svg"); }
        QCheckBox::indicator:hover, QCheckBox::indicator:focus { border-color: #8eb1ff; }
        QCheckBox::indicator:disabled { background: #273347; border-color: #46556e; }
        QProgressBar { border: 1px solid #364359; border-radius: 6px; background: #151c27; min-height: 26px; text-align: center; }
        QProgressBar::chunk { background: #344e80; border-radius: 5px; }
        QSlider::groove:horizontal { height: 5px; background: #364359; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #709bff; border-radius: 2px; }
        QSlider::handle:horizontal { background: #8eb1ff; width: 14px; margin: -5px 0; border-radius: 7px; }
        QSlider::sub-page:horizontal:disabled, QSlider::handle:horizontal:disabled { background: #526078; }
        QSplitter::handle { background: #2b3547; }
        QTableWidget { background: #151c27; alternate-background-color: #202a3a; border: 1px solid #364359; gridline-color: #2b3547; selection-background-color: #365fd1; selection-color: white; }
        QHeaderView::section { background: #222d3e; border: none; padding: 9px; font-weight: 600; }
        QMenu { background: #1b2230; border: 1px solid #364359; padding: 5px; }
        QMenu::item { padding: 8px 24px; }
        QMenu::item:selected { background: #273754; color: #8eb1ff; }
        QMenu::item:disabled { color: #78859a; }
        QMenu::separator { height: 1px; background: #364359; margin: 4px 8px; }
        QToolTip { color: #e5ebf5; background: #273754; border: 1px solid #46556e; padding: 6px; }
    """.replace("$ICON_DIR", (Path(__file__).parent / "icons").resolve().as_posix()))


def label(text, role="muted"):
    result = QLabel(text)
    result.setTextFormat(Qt.PlainText)
    result.setProperty("role", role)
    result.setWordWrap(True)
    return result


def heading(title, subtitle):
    layout = QVBoxLayout()
    layout.setSpacing(3)
    layout.addWidget(label(title, "title"))
    layout.addWidget(label(subtitle))
    return layout


class ImagePreview(QLabel):
    """Resize a cached preview without reloading files or touching captions."""
    def __init__(self):
        super().__init__()
        self.source = QPixmap()
        self.setObjectName("imagePreview")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(260, 220)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)

    def set_source(self, pixmap):
        self.source = pixmap
        if pixmap.isNull():
            self.setText("Unable to preview this image.")
        else:
            self.render_preview()

    def clear(self):
        self.source = QPixmap()
        super().clear()

    def render_preview(self):
        if not self.source.isNull():
            self.setPixmap(self.source.scaled(self.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_preview()
