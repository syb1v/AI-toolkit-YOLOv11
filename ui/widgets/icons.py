"""
Модуль с кроссплатформенными иконками для UI.

Использует стандартные QStyle иконки Qt, которые корректно отображаются
на всех платформах (Windows, macOS, Linux).
"""
from PyQt5.QtWidgets import QStyle, QApplication
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt


class Icons:
    """Централизованный доступ к стандартным иконкам Qt."""
    
    # Файлы и папки
    @staticmethod
    def folder() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_DirIcon)
    
    @staticmethod
    def folder_open() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_DirOpenIcon)
    
    @staticmethod
    def file() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_FileIcon)
    
    @staticmethod
    def file_new() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_FileDialogNewFolder)
    
    # Действия
    @staticmethod
    def play() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_MediaPlay)
    
    @staticmethod
    def stop() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_MediaStop)
    
    @staticmethod
    def refresh() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_BrowserReload)
    
    @staticmethod
    def save() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_DialogSaveButton)
    
    @staticmethod
    def delete() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_TrashIcon)
    
    @staticmethod
    def add() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_FileDialogNewFolder)
    
    # Навигация
    @staticmethod
    def arrow_left() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_ArrowLeft)
    
    @staticmethod
    def arrow_right() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_ArrowRight)
    
    @staticmethod
    def arrow_up() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_ArrowUp)
    
    @staticmethod
    def arrow_down() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_ArrowDown)
    
    @staticmethod
    def undo() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_ArrowBack)
    
    @staticmethod
    def redo() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_ArrowForward)
    
    # Статус
    @staticmethod
    def success() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_DialogApplyButton)
    
    @staticmethod
    def warning() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_MessageBoxWarning)
    
    @staticmethod
    def error() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_MessageBoxCritical)
    
    @staticmethod
    def info() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_MessageBoxInformation)
    
    @staticmethod
    def question() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_MessageBoxQuestion)
    
    # Действия с окнами
    @staticmethod
    def close() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_DialogCloseButton)
    
    @staticmethod
    def help() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_DialogHelpButton)
    
    # Специальные
    @staticmethod
    def view() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_FileDialogContentsView)
    
    @staticmethod
    def computer() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_ComputerIcon)
    
    @staticmethod
    def drive() -> QIcon:
        return QApplication.style().standardIcon(QStyle.SP_DriveHDIcon)


def create_color_dot(color: str, size: int = 12) -> QPixmap:
    """
    Создаёт пиксель-карту с цветной точкой.
    
    Args:
        color: Цвет в формате hex (#RRGGBB)
        size: Размер точки в пикселях
        
    Returns:
        QPixmap с цветной точкой
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    
    return pixmap


def create_status_indicator(status: str, size: int = 16) -> QPixmap:
    """
    Создаёт индикатор статуса.
    
    Args:
        status: 'success', 'warning', 'error', 'info', 'neutral'
        size: Размер индикатора
        
    Returns:
        QPixmap с индикатором
    """
    colors = {
        'success': '#3fb950',
        'warning': '#d29922', 
        'error': '#f85149',
        'info': '#58a6ff',
        'neutral': '#8b949e'
    }
    
    color = colors.get(status, colors['neutral'])
    return create_color_dot(color, size)


# Символы для использования в текстовых метках
# Эти символы безопасны для всех платформ и шрифтов
TEXT_SYMBOLS = {
    'arrow_up': '▲',
    'arrow_down': '▼',
    'arrow_left': '◀',
    'arrow_right': '▶',
    'bullet': '●',
    'check': '✓',
    'cross': '✗',
    'dash': '—',
    'plus': '+',
    'minus': '-',
    'star': '★',
    'circle': '○',
    'filled_circle': '●',
    'square': '■',
    'triangle_up': '△',
    'triangle_down': '▽',
}
