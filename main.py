import sys
import os

# Добавляем текущую директорию в путь для импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from config import APP_TITLE
from ui.main_window import MainWindow


def main():
    """Точка входа в приложение."""
    
    # Настройки Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    
    # Создаем и показываем главное окно
    window = MainWindow()
    window.show()
    
    # Запуск цикла событий
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
