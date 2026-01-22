"""
Виджет лога для отображения сообщений.
"""
from PyQt5.QtWidgets import QPlainTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QTextCharFormat, QColor, QFont
from datetime import datetime


class LogWidget(QWidget):
    """
    Виджет для отображения логов.
    Поддерживает цветовое выделение уровней.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Кнопки управления
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        self._btn_clear = QPushButton("Очистить")
        self._btn_clear.clicked.connect(self.clear)
        btn_layout.addWidget(self._btn_clear)
        
        self._btn_copy = QPushButton("Копировать")
        self._btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_layout.addWidget(self._btn_copy)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Текстовое поле
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setProperty("class", "log")
        self._text.setFont(QFont("JetBrains Mono", 10))
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._text)
    
    @pyqtSlot(str)
    def log(self, message: str, level: str = "info"):
        """
        Добавляет сообщение в лог.
        
        Args:
            message: текст сообщения
            level: уровень - 'info', 'success', 'warning', 'error'
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Определяем цвет и префикс
        colors = {
            "info": "#a0a0a0",
            "success": "#4ade80",
            "warning": "#fbbf24",
            "error": "#f87171",
        }
        prefixes = {
            "info": "[INFO]",
            "success": "[OK]",
            "warning": "[WARN]",
            "error": "[ERR]",
        }
        
        color = colors.get(level, colors["info"])
        prefix = prefixes.get(level, "")
        
        # Форматированное сообщение
        formatted = f"[{timestamp}] {prefix} {message}"
        
        # Добавляем с цветом
        cursor = self._text.textCursor()
        format = QTextCharFormat()
        format.setForeground(QColor(color))
        
        cursor.movePosition(cursor.End)
        cursor.insertText(formatted + "\n", format)
        
        # Прокрутка вниз
        self._text.verticalScrollBar().setValue(
            self._text.verticalScrollBar().maximum()
        )
    
    def info(self, message: str):
        """Добавляет информационное сообщение."""
        self.log(message, "info")
    
    def success(self, message: str):
        """Добавляет сообщение об успехе."""
        self.log(message, "success")
    
    def warning(self, message: str):
        """Добавляет предупреждение."""
        self.log(message, "warning")
    
    def error(self, message: str):
        """Добавляет сообщение об ошибке."""
        self.log(message, "error")
    
    def clear(self):
        """Очищает лог."""
        self._text.clear()
    
    def _copy_to_clipboard(self):
        """Копирует содержимое в буфер обмена."""
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text.toPlainText())
    
    def get_text(self) -> str:
        """Возвращает текст лога."""
        return self._text.toPlainText()
