import sys

from PySide6.QtWidgets import QApplication

from src.ui.tagging_window import TaggingWindow
from src.training.records import default_session_path


def main() -> int:
    app = QApplication(sys.argv)
    window = TaggingWindow(session_path=default_session_path())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
