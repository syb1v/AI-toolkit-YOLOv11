"""
Главное окно приложения AI Toolkit YOLO v2.
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QAction, QStatusBar, QMessageBox, QStyle
)

from config import APP_TITLE, STYLE_FILE
from core.class_manager import ClassManager
from core.hardware_detector import get_hardware_detector
from ui.tabs.labeling_tab import LabelingTab
from ui.tabs.training_tab import TrainingTab
from ui.tabs.analysis_tab import AnalysisTab


class MainWindow(QMainWindow):
    """Главное окно приложения."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1280, 800)
        self.resize(1400, 900)
        
        # Убираем меню
        self.menuBar().hide()
        
        # Менеджеры
        self._class_manager = ClassManager()
        self._load_class_manager()
        
        # UI
        self._init_ui()
        self._create_statusbar()
        self._connect_signals()
        self._apply_styles()
        self._create_statusbar()
        self._connect_signals()
        self._apply_styles()
    
    def _load_class_manager(self):
        try:
            self._class_manager.load()
        except Exception as e:
            QMessageBox.warning(
                self, "Предупреждение",
                f"Не удалось загрузить наборы классов:\n{e}"
            )
    
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Вкладки
        # Вкладки
        self._tabs = QTabWidget()
        # self._tabs.setDocumentMode(True)  # Убираем, так как вызывает артефакты с QSS
        
        # Вкладка разметки
        self._labeling_tab = LabelingTab(self._class_manager)
        self._tabs.addTab(self._labeling_tab, self.style().standardIcon(QStyle.SP_FileIcon), "Разметка")
        
        # Вкладка обучения
        self._training_tab = TrainingTab()
        self._tabs.addTab(self._training_tab, self.style().standardIcon(QStyle.SP_ComputerIcon), "Обучение")
        
        # Вкладка анализа
        self._analysis_tab = AnalysisTab()
        self._tabs.addTab(self._analysis_tab, self.style().standardIcon(QStyle.SP_FileDialogInfoView), "Анализ")
        
        layout.addWidget(self._tabs)
    
    def _create_menu(self):
        menubar = self.menuBar()
        
        # Файл
        file_menu = menubar.addMenu("Файл")
        
        exit_action = QAction("Выход", self)
        exit_action.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Помощь
        help_menu = menubar.addMenu("Помощь")
        
        hw_action = QAction("Оборудование", self)
        hw_action.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        hw_action.triggered.connect(self._show_hardware_info)
        help_menu.addAction(hw_action)
        
        about_action = QAction("О программе", self)
        about_action.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        
        hw = get_hardware_detector()
        device = hw.best_device
        self._statusbar.showMessage(f"{device.name}")
    
    def _connect_signals(self):
        self._labeling_tab.dataset_exported.connect(self._on_dataset_exported)
        self._training_tab.training_finished.connect(self._on_training_finished)
    
    def _apply_styles(self):
        if STYLE_FILE.exists():
            with open(STYLE_FILE, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
    
    def _on_dataset_exported(self, path: str):
        self._training_tab.set_dataset_path(path)
        self._statusbar.showMessage(f"Датасет экспортирован: {path}", 5000)
        # Переключаемся на вкладку обучения
        self._tabs.setCurrentIndex(1)
    
    def _on_training_finished(self, weights_path: str):
        self._analysis_tab.load_model_from_path(weights_path)
        self._statusbar.showMessage("Обучение завершено!", 5000)
        # Переключаемся на вкладку анализа
        self._tabs.setCurrentIndex(2)
    
    def _show_hardware_info(self):
        hw = get_hardware_detector()
        QMessageBox.information(
            self, "Оборудование",
            hw.get_status_report()
        )
    
    def _show_about(self):
        QMessageBox.about(
            self, "О программе",
            f"<h2>{APP_TITLE}</h2>"
            "<p>Инструмент полного цикла для Computer Vision.</p>"
            
            "<h3>Возможности:</h3>"
            "<ul>"
            "<li>Разметка изображений (Box/Polygon)</li>"
            "<li>Обучение моделей YOLOv11</li>"
            "<li>Анализ и дообучение</li>"
            "</ul>"
            
            "<h3>Поддержка устройств:</h3>"
            "<ul>"
            "<li>NVIDIA GPU (CUDA)</li>"
            "<li>AMD GPU (ROCm)</li>"
            "<li>Intel GPU (oneAPI)</li>"
            "<li>Apple Silicon (MPS)</li>"
            "<li>CPU</li>"
            "</ul>"
            
            "<p>© 2024 AI Toolkit Team</p>"
        )
    
    def closeEvent(self, event):
        if self._class_manager.is_dirty:
            reply = QMessageBox.question(
                self, "Сохранение",
                "Сохранить изменения в наборах классов?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self._class_manager.save()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        event.accept()
