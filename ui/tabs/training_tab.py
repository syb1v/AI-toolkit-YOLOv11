"""
Вкладка обучения модели YOLO.
"""
import os
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QGroupBox, QProgressBar,
    QScrollArea, QGridLayout, QTabWidget, QInputDialog, QStyle,
    QPlainTextEdit, QSizePolicy
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from ui.widgets.log_widget import LogWidget
from core.project_manager import ProjectManager
from core.trainer import YOLOTrainer
from core.hardware_detector import get_hardware_detector
from models.presets import (
    PRESETS, AugmentationLevel, AUGMENTATION_PRESETS,
    get_augmentation_level_name, AUGMENTATION_DESCRIPTIONS
)
from config import YOLO_MODELS, YOLO_SEGMENT_MODELS, TASK_TYPES, DATA_YAML_FILE


class TrainingTab(QWidget):
    """Вкладка обучения модели YOLO."""
    
    training_finished = pyqtSignal(str)
    sig_log_msg = pyqtSignal(str, str)
    sig_progress_update = pyqtSignal(object)
    sig_training_complete = pyqtSignal(object)
    sig_training_error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_manager = ProjectManager()
        self._trainer: Optional[YOLOTrainer] = None
        self._hardware = get_hardware_detector()
        self._current_aug_level = AugmentationLevel.MEDIUM
        
        # Таймер для обновления elapsed time
        self._elapsed_timer = QTimer()
        self._elapsed_timer.timeout.connect(self._update_elapsed_time)
        self._training_start_time = 0.0
        
        # Connect signals
        self.sig_log_msg.connect(self._on_log_slot)
        self.sig_progress_update.connect(self._on_progress_slot)
        self.sig_training_complete.connect(self._on_finished_slot)
        self.sig_training_error.connect(self._on_error_slot)
        self._init_ui()
        self._update_device_combo()
        self._load_projects()
    
    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель - настройки
        left = self._create_settings_panel()
        splitter.addWidget(left)
        
        # Правая панель - логи
        right = self._create_monitoring_panel()
        splitter.addWidget(right)
        
        splitter.setSizes([450, 550])
        layout.addWidget(splitter)
    
    def _create_settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        panel = QFrame()
        panel.setProperty("cssClass", "card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # === Заголовок ===
        header = QLabel("Обучение модели YOLO")
        header.setProperty("cssClass", "title")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #58a6ff;")
        layout.addWidget(header)
        
        # === 1. Проект ===
        grp1 = QGroupBox("Проект")
        lay1 = QVBoxLayout(grp1)
        
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._btn_new = QPushButton("Создать")
        self._btn_new.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self._btn_new.setToolTip("Создать новый проект обучения")
        self._btn_new.clicked.connect(self._create_project)
        btn_row.addWidget(self._btn_new)
        
        self._btn_load = QPushButton("Открыть")
        self._btn_load.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self._btn_load.setToolTip("Открыть существующий проект")
        self._btn_load.clicked.connect(self._load_project)
        btn_row.addWidget(self._btn_load)
        lay1.addLayout(btn_row)
        
        self._combo_projects = QComboBox()
        self._combo_projects.setToolTip("Выберите проект для работы")
        self._combo_projects.currentTextChanged.connect(self._on_project_selected)
        lay1.addWidget(self._combo_projects)
        
        self._lbl_project = QLabel("Выберите или создайте проект")
        self._lbl_project.setProperty("cssClass", "hint")
        lay1.addWidget(self._lbl_project)
        layout.addWidget(grp1)
        
        # === 2. Датасет ===
        grp2 = QGroupBox("Датасет")
        lay2 = QVBoxLayout(grp2)
        
        self._btn_dataset = QPushButton("Выбрать data.yaml")
        self._btn_dataset.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self._btn_dataset.setToolTip("Укажите путь к файлу data.yaml вашего датасета")
        self._btn_dataset.clicked.connect(self._select_dataset)
        lay2.addWidget(self._btn_dataset)
        
        self._lbl_dataset = QLabel("Датасет не выбран")
        self._lbl_dataset.setWordWrap(True)
        self._lbl_dataset.setProperty("cssClass", "hint")
        lay2.addWidget(self._lbl_dataset)
        layout.addWidget(grp2)
        
        # === 3. Модель и устройство ===
        grp3 = QGroupBox("Модель и устройство")
        lay3 = QGridLayout(grp3)
        lay3.setSpacing(10)
        
        lay3.addWidget(QLabel("Тип задачи:"), 0, 0)
        self._combo_task = QComboBox()
        for k, v in TASK_TYPES.items():
            self._combo_task.addItem(v, k)
        self._combo_task.setToolTip(
            "Детекция — обнаружение объектов прямоугольниками (Bounding Box).\n"
            "Используйте для подсчёта объектов, определения их положения.\n\n"
            "Сегментация — выделение объектов полигонами (Mask).\n"
            "Используйте для точного определения границ объектов."
        )
        self._combo_task.currentIndexChanged.connect(self._on_task_changed)
        lay3.addWidget(self._combo_task, 0, 1)
        
        lay3.addWidget(QLabel("Модель YOLO:"), 1, 0)
        self._combo_model = QComboBox()
        self._update_model_combo()
        self._combo_model.setToolTip(
            "Размер модели влияет на скорость и точность:\n\n"
            "• Nano — самая быстрая, для слабых устройств и real-time\n"
            "• Small — баланс скорости и точности\n"
            "• Medium — рекомендуется для большинства задач\n"
            "• Large — высокая точность, требует больше памяти\n"
            "• XLarge — максимальная точность, требует мощный GPU"
        )
        lay3.addWidget(self._combo_model, 1, 1)
        
        lay3.addWidget(QLabel("Устройство:"), 2, 0)
        self._combo_device = QComboBox()
        self._combo_device.setToolTip(
            "Устройство для обучения модели:\n\n"
            "• GPU (cuda) — рекомендуется, в 10-50 раз быстрее CPU\n"
            "• CPU — медленнее, но работает везде"
        )
        lay3.addWidget(self._combo_device, 2, 1)
        
        self._lbl_device = QLabel()
        self._lbl_device.setProperty("cssClass", "success")
        lay3.addWidget(self._lbl_device, 3, 0, 1, 2)
        layout.addWidget(grp3)
        
        # === 4. Пресет ===
        grp4 = QGroupBox("Пресет обучения")
        lay4 = QVBoxLayout(grp4)
        
        self._combo_preset = QComboBox()
        for pid, p in PRESETS.items():
            self._combo_preset.addItem(f"{p.name}", pid)
        self._combo_preset.setToolTip("Готовые наборы настроек для разных сценариев")
        self._combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        lay4.addWidget(self._combo_preset)
        
        self._lbl_preset = QLabel()
        self._lbl_preset.setWordWrap(True)
        self._lbl_preset.setProperty("cssClass", "hint")
        lay4.addWidget(self._lbl_preset)
        layout.addWidget(grp4)
        
        # === 5. Аугментация ===
        grp5 = QGroupBox("Аугментация данных")
        lay5 = QVBoxLayout(grp5)
        
        aug_hint = QLabel("Чем больше аугментация — тем лучше модель справляется\nс разнообразными и нестабильными данными")
        aug_hint.setProperty("cssClass", "hint")
        aug_hint.setWordWrap(True)
        lay5.addWidget(aug_hint)
        
        self._combo_augmentation = QComboBox()
        for level in AugmentationLevel:
            self._combo_augmentation.addItem(get_augmentation_level_name(level), level)
        self._combo_augmentation.setCurrentIndex(2)  # Medium по умолчанию
        self._combo_augmentation.setToolTip(
            "Аугментация — автоматическое изменение изображений для улучшения модели:\n\n"
            "• Без — изображения используются как есть\n"
            "• Лёгкая — HSV ±5%, поворот ±5°, отражение, mosaic 50%\n"
            "• Средняя — HSV ±7%, поворот ±10°, mosaic + mixup 10%\n"
            "• Сильная — все аугментации: HSV, повороты, mosaic, mixup, copy-paste\n\n"
            "Чем меньше данных — тем сильнее аугментация."
        )
        self._combo_augmentation.currentIndexChanged.connect(self._on_augmentation_changed)
        lay5.addWidget(self._combo_augmentation)
        
        self._lbl_augmentation = QLabel(AUGMENTATION_DESCRIPTIONS[AugmentationLevel.MEDIUM])
        self._lbl_augmentation.setProperty("cssClass", "hint")
        lay5.addWidget(self._lbl_augmentation)
        layout.addWidget(grp5)
        
        # === 6. Параметры ===
        grp6 = QGroupBox("Параметры обучения")
        lay6 = QGridLayout(grp6)
        lay6.setSpacing(10)
        
        lay6.addWidget(QLabel("Эпохи:"), 0, 0)
        self._spin_epochs = QSpinBox()
        self._spin_epochs.setRange(1, 10000)
        self._spin_epochs.setValue(100)
        self._spin_epochs.setToolTip(
            "Эпоха — один полный проход по всему датасету.\n\n"
            "• Больше эпох = лучше качество, но дольше обучение\n"
            "• Слишком много эпох = переобучение (модель 'зубрит')\n\n"
            "Рекомендации: 50-100 для теста, 100-300 для production"
        )
        lay6.addWidget(self._spin_epochs, 0, 1)
        
        lay6.addWidget(QLabel("Batch Size:"), 1, 0)
        self._spin_batch = QSpinBox()
        self._spin_batch.setRange(-1, 256)
        self._spin_batch.setValue(16)
        self._spin_batch.setToolTip(
            "Количество изображений, обрабатываемых одновременно.\n\n"
            "• -1 = автоопределение оптимального размера\n"
            "• Больше batch = быстрее обучение, но больше памяти GPU\n\n"
            "При ошибке 'CUDA out of memory' уменьшите это значение.\n"
            "Рекомендации: 8GB VRAM → 16, 4GB → 8, CPU → 4"
        )
        lay6.addWidget(self._spin_batch, 1, 1)
        
        lay6.addWidget(QLabel("Размер фото:"), 2, 0)
        self._spin_imgsz = QSpinBox()
        self._spin_imgsz.setRange(32, 1920)
        self._spin_imgsz.setSingleStep(32)
        self._spin_imgsz.setValue(640)
        self._spin_imgsz.setToolTip(
            "Размер изображений для обучения (в пикселях).\n\n"
            "• Должен делиться на 32\n"
            "• 640 — стандарт, баланс скорости и качества\n"
            "• 1280 — для мелких объектов, требует больше памяти\n"
            "• 416/320 — для слабых устройств\n\n"
            "Рекомендуется использовать одинаковый размер для обучения и инференса."
        )
        lay6.addWidget(self._spin_imgsz, 2, 1)
        
        lay6.addWidget(QLabel("Learning Rate:"), 3, 0)
        self._spin_lr = QDoubleSpinBox()
        self._spin_lr.setRange(0.0001, 1.0)
        self._spin_lr.setDecimals(4)
        self._spin_lr.setSingleStep(0.001)
        self._spin_lr.setValue(0.01)
        self._spin_lr.setToolTip(
            "Learning Rate (lr0) — скорость обучения.\n\n"
            "Определяет размер шага при оптимизации весов модели:\n"
            "• 0.01 — стандарт для SGD\n"
            "• 0.001 — для AdamW или малых датасетов\n\n"
            "Слишком большой LR = нестабильное обучение.\n"
            "Слишком маленький = очень медленная сходимость."
        )
        lay6.addWidget(self._spin_lr, 3, 1)
        
        # Блокировка случайной прокрутки (требует клика для активации скролла)
        inputs = [
            self._combo_task, self._combo_model, self._combo_device,
            self._combo_preset, self._combo_augmentation,
            self._spin_epochs, self._spin_batch, self._spin_imgsz, self._spin_lr
        ]
        
        def create_safe_scroll(widget):
            original_wheel = widget.wheelEvent
            def safe_wheel_event(event):
                if widget.hasFocus():
                    original_wheel(event)
                else:
                    event.ignore()
            return safe_wheel_event
            
        for w in inputs:
            w.wheelEvent = create_safe_scroll(w)
            w.setFocusPolicy(Qt.StrongFocus)
        
        layout.addWidget(grp6)
        
        # === 7. Управление ===
        grp7 = QGroupBox("Управление")
        lay7 = QVBoxLayout(grp7)
        
        self._btn_start = QPushButton("Начать обучение")
        self._btn_start.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._btn_start.setProperty("cssClass", "primary")
        self._btn_start.setMinimumHeight(50)
        self._btn_start.setStyleSheet("font-size: 15px; font-weight: bold;")
        self._btn_start.clicked.connect(self._start_training)
        lay7.addWidget(self._btn_start)
        
        self._btn_stop = QPushButton("Остановить")
        self._btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self._btn_stop.setProperty("cssClass", "danger")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_training)
        lay7.addWidget(self._btn_stop)
        
        self._progress = QProgressBar()
        self._progress.setMinimumHeight(30)
        lay7.addWidget(self._progress)
        
        self._lbl_status = QLabel("Готов к обучению")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        self._lbl_status.setStyleSheet("font-size: 13px;")
        lay7.addWidget(self._lbl_status)
        layout.addWidget(grp7)
        
        layout.addStretch()
        scroll.setWidget(panel)
        return scroll
    
    def _create_monitoring_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        tabs = QTabWidget()
        
        # === Вкладка Метрики ===
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setSpacing(12)
        
        # Таймер и ETA
        time_group = QGroupBox("Время")
        time_layout = QHBoxLayout(time_group)
        
        self._elapsed_label = QLabel("Время: 00:00:00")
        self._elapsed_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        time_layout.addWidget(self._elapsed_label)
        
        self._eta_label = QLabel("Осталось: --")
        self._eta_label.setStyleSheet("font-size: 14px;")
        time_layout.addWidget(self._eta_label)
        
        metrics_layout.addWidget(time_group)
        
        # Loss метрики
        loss_group = QGroupBox("Потери (Loss)")
        loss_layout = QGridLayout(loss_group)
        loss_layout.setSpacing(10)
        
        # Train Loss
        loss_layout.addWidget(QLabel("Train Loss:"), 0, 0)
        self._train_loss_label = QLabel("--.----")
        self._train_loss_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        loss_layout.addWidget(self._train_loss_label, 0, 1)
        
        self._train_loss_trend = QLabel("")
        self._train_loss_trend.setStyleSheet("font-size: 16px;")
        loss_layout.addWidget(self._train_loss_trend, 0, 2)
        
        train_hint = QLabel("Ошибка на обучающих данных")
        train_hint.setProperty("cssClass", "hint")
        train_hint.setStyleSheet("color: #8b949e; font-size: 11px;")
        loss_layout.addWidget(train_hint, 1, 0, 1, 3)
        
        # Val Loss
        loss_layout.addWidget(QLabel("Val Loss:"), 2, 0)
        self._val_loss_label = QLabel("--.----")
        self._val_loss_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        loss_layout.addWidget(self._val_loss_label, 2, 1)
        
        self._val_loss_trend = QLabel("")
        self._val_loss_trend.setStyleSheet("font-size: 16px;")
        loss_layout.addWidget(self._val_loss_trend, 2, 2)
        
        val_hint = QLabel("Ошибка на новых данных (главный показатель)")
        val_hint.setProperty("cssClass", "hint")
        val_hint.setStyleSheet("color: #8b949e; font-size: 11px;")
        loss_layout.addWidget(val_hint, 3, 0, 1, 3)
        
        metrics_layout.addWidget(loss_group)
        
        # Состояние модели
        state_group = QGroupBox("Состояние обучения")
        state_layout = QVBoxLayout(state_group)
        
        self._model_state_label = QLabel("● Ожидание модели...")
        self._model_state_label.setAlignment(Qt.AlignCenter)
        self._model_state_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        state_layout.addWidget(self._model_state_label)
        
        state_hint = QLabel(
            "<span style='color:#3fb950'>●</span> Нормальное — модель учится правильно<br>"
            "<span style='color:#d29922'>●</span> Переобучение — модель 'зубрит' данные<br>"
            "<span style='color:#58a6ff'>●</span> Недообучение — нужно больше эпох<br>"
            "<span style='color:#3fb950'>●</span> Плато — модель достигла максимума<br>"
            "<span style='color:#f85149'>●</span> Проблема — проверьте данные или LR"
        )
        state_hint.setStyleSheet("color: #8b949e; font-size: 11px;")
        state_layout.addWidget(state_hint)
        
        metrics_layout.addWidget(state_group)
        
        # Качество
        quality_group = QGroupBox("Качество модели")
        quality_layout = QGridLayout(quality_group)
        
        quality_layout.addWidget(QLabel("mAP50:"), 0, 0)
        self._map50_label = QLabel("--.--")
        self._map50_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #58a6ff;")
        quality_layout.addWidget(self._map50_label, 0, 1)
        
        quality_layout.addWidget(QLabel("mAP50-95:"), 0, 2)
        self._map95_label = QLabel("--.--")
        self._map95_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #58a6ff;")
        quality_layout.addWidget(self._map95_label, 0, 3)
        
        metrics_layout.addWidget(quality_group)
        
        metrics_layout.addStretch()
        tabs.addTab(metrics_widget, "Метрики")
        
        # === Вкладка логов ===
        self._log = LogWidget()
        tabs.addTab(self._log, "Логи")
        
        # === Вкладка помощи ===
        help_widget = QWidget()
        help_layout = QVBoxLayout(help_widget)
        help_text = QLabel(
            "<h3>Советы по обучению</h3>"
            "<p><b>Эпохи:</b> Больше эпох = лучше качество, но дольше.<br>"
            "Начните с 50-100 для теста, увеличьте при необходимости.</p>"
            
            "<p><b>Batch Size:</b> Зависит от памяти GPU.<br>"
            "8GB VRAM → batch 16, 4GB → batch 8, CPU → batch 4</p>"
            
            "<p><b>Размер изображения:</b> 640 — стандарт.<br>"
            "1280 — для мелких объектов (нужно больше VRAM)</p>"
            
            "<p><b>Аугментация:</b><br>"
            "• Мало данных → сильная аугментация<br>"
            "• Много чистых данных → лёгкая/средняя<br>"
            "• Шумные данные → сильная</p>"
            
            "<p><b>Если обучение падает:</b><br>"
            "• Уменьшите batch size<br>"
            "• Уменьшите размер изображения<br>"
            "• Используйте CPU если нет GPU</p>"
            
            "<h4>Что означают метрики:</h4>"
            "<p><b>Train Loss:</b> Ошибка на обучающих данных.<br>"
            "▼ = падает (хорошо), ▲ = растёт (плохо)</p>"
            
            "<p><b>Val Loss:</b> Ошибка на новых данных.<br>"
            "Это главный показатель качества модели.</p>"
            
            "<p><b>mAP50:</b> Точность при IoU=0.5 (0-1, выше = лучше)</p>"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("padding: 16px;")
        help_layout.addWidget(help_text)
        help_layout.addStretch()
        tabs.addTab(help_widget, "Помощь")
        
        layout.addWidget(tabs)
        return panel
    
    def _update_device_combo(self):
        self._combo_device.clear()
        for d in self._hardware.available_devices:
            self._combo_device.addItem(d.get_display_name(), d.device_id)
        best = self._hardware.best_device
        self._lbl_device.setText(f"Рекомендуется: {best.name}")
    
    def _update_model_combo(self):
        self._combo_model.clear()
        task = self._combo_task.currentData()
        models = YOLO_SEGMENT_MODELS if task == "segment" else YOLO_MODELS
        for name, file in models.items():
            self._combo_model.addItem(name, file)
    
    def _load_projects(self):
        self._combo_projects.clear()
        self._combo_projects.addItem("Выберите проект...", None)
        for n in self._project_manager.list_projects():
            self._combo_projects.addItem(n, n)
    
    def _on_task_changed(self):
        self._update_model_combo()
    
    def _on_project_selected(self, name):
        if not name or name == "Выберите проект...":
            return
        proj_name = name
        path = self._project_manager.base_dir / proj_name
        try:
            cfg = self._project_manager.load_project(str(path))
            self._apply_config(cfg)
            self._lbl_project.setText(f"Загружен: {proj_name}")
            self._lbl_project.setProperty("cssClass", "success")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", str(e))
    
    def _on_preset_changed(self):
        pid = self._combo_preset.currentData()
        if pid in PRESETS:
            p = PRESETS[pid]
            self._lbl_preset.setText(p.description)
            if pid != "custom":
                self._spin_epochs.setValue(p.epochs)
                self._spin_batch.setValue(p.batch_size)
                self._spin_imgsz.setValue(p.image_size)
                self._spin_lr.setValue(p.learning_rate)
                # Устанавливаем уровень аугментации
                for i in range(self._combo_augmentation.count()):
                    if self._combo_augmentation.itemData(i) == p.augmentation_level:
                        self._combo_augmentation.setCurrentIndex(i)
                        break
    
    def _on_augmentation_changed(self):
        level = self._combo_augmentation.currentData()
        if level:
            self._current_aug_level = level
            self._lbl_augmentation.setText(AUGMENTATION_DESCRIPTIONS.get(level, ""))
    
    def _apply_config(self, cfg):
        if cfg.data_yaml_path:
            self._lbl_dataset.setText(cfg.data_yaml_path)
            self._lbl_dataset.setProperty("cssClass", "success")
        self._spin_epochs.setValue(cfg.epochs)
        self._spin_batch.setValue(cfg.batch_size)
        self._spin_imgsz.setValue(cfg.image_size)
        self._spin_lr.setValue(cfg.learning_rate)
    
    def _create_project(self):
        name, ok = QInputDialog.getText(self, "Новый проект", "Название проекта:")
        if ok and name:
            self._project_manager.create_project(name)
            self._load_projects()
            self._log.success(f"Создан проект: {name}")
    
    def _load_project(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите папку проекта")
        if path:
            try:
                cfg = self._project_manager.load_project(path)
                self._apply_config(cfg)
                self._load_projects()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))
    
    def _select_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите data.yaml", "", "YAML (*.yaml *.yml)")
        if path:
            self._lbl_dataset.setText(path)
            self._lbl_dataset.setProperty("cssClass", "success")
            if self._project_manager.current_config:
                self._project_manager.current_config.data_yaml_path = path
                self._project_manager.save_project()
    
    def _start_training(self):
        if not self._project_manager.current_project_path:
            QMessageBox.warning(self, "Ошибка", "Сначала создайте или выберите проект")
            return
        
        ds = self._lbl_dataset.text()
        if ds == "Датасет не выбран" or not os.path.exists(ds):
            QMessageBox.warning(self, "Ошибка", "Выберите датасет (data.yaml)")
            return
        
        ds_path = ds
        
        cfg = self._project_manager.current_config
        cfg.model_name = self._combo_model.currentData()
        cfg.data_yaml_path = ds_path
        cfg.device = self._combo_device.currentData()
        cfg.epochs = self._spin_epochs.value()
        cfg.batch_size = self._spin_batch.value()
        cfg.image_size = self._spin_imgsz.value()
        cfg.learning_rate = self._spin_lr.value()
        
        # Получаем аугментацию
        aug_level = self._combo_augmentation.currentData()
        cfg.augmentation = AUGMENTATION_PRESETS.get(aug_level, {}).copy()
        
        self._project_manager.save_project()
        
        runs = self._project_manager.get_runs_dir()
        self._trainer = YOLOTrainer({
            "model_name": cfg.model_name,
            "data_yaml": cfg.data_yaml_path,
            "save_dir": str(runs / "train"),
            "training_args": cfg.get_training_args()
        })
        
        self._trainer.on_progress = self._on_progress
        self._trainer.on_log = self._on_log
        self._trainer.on_finished = self._on_finished
        self._trainer.on_error = self._on_error
        self._trainer.start()
        
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setMaximum(cfg.epochs)
        self._progress.setValue(0)
        self._lbl_status.setText("Обучение запущено...")
        
        # Сброс метрик
        self._elapsed_label.setText("Время: 00:00:00")
        self._eta_label.setText("Осталось: --")
        self._train_loss_label.setText("--.----")
        self._train_loss_trend.setText("")
        self._val_loss_label.setText("--.----")
        self._val_loss_trend.setText("")
        self._map50_label.setText("--.--")
        self._map95_label.setText("--.--")
        self._model_state_label.setText("● Оценка...")
        self._model_state_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #58a6ff;")
        
        self._log.success("Обучение запущено!")
        self._log.info(f"Эпохи: {cfg.epochs}, Batch: {cfg.batch_size}, ImgSize: {cfg.image_size}")
        self._log.info(f"Аугментация: {get_augmentation_level_name(aug_level)}")
    
    def _stop_training(self):
        if self._trainer:
            self._trainer.stop()
            self._log.warning("Остановка обучения...")
    
    def _on_progress(self, p):
        self.sig_progress_update.emit(p)

    def _on_progress_slot(self, p):
        """Обновляет UI на основе прогресса обучения."""
        self._progress.setValue(p.current_epoch)
        self._lbl_status.setText(f"Эпоха {p.current_epoch}/{p.total_epochs}")
        
        # Обновляем время
        if p.start_time and p.start_time != self._training_start_time:
            self._training_start_time = p.start_time
            self._elapsed_timer.start(1000)  # Обновлять каждую секунду
        
        # ETA
        if p.eta_seconds > 0:
            self._eta_label.setText(f"Осталось: {self._format_time(p.eta_seconds)}")
        
        # Train Loss
        if p.train_loss > 0:
            self._train_loss_label.setText(f"{p.train_loss:.4f}")
            trend_text, trend_color = self._get_trend_display(p.train_loss_trend)
            self._train_loss_trend.setText(trend_text)
            self._train_loss_trend.setStyleSheet(f"font-size: 16px; color: {trend_color};")
        
        # Val Loss
        if p.val_loss > 0:
            self._val_loss_label.setText(f"{p.val_loss:.4f}")
            trend_text, trend_color = self._get_trend_display(p.val_loss_trend)
            self._val_loss_trend.setText(trend_text)
            self._val_loss_trend.setStyleSheet(f"font-size: 16px; color: {trend_color};")
        
        # Качество
        if p.map50 > 0:
            self._map50_label.setText(f"{p.map50:.4f}")
        if p.map50_95 > 0:
            self._map95_label.setText(f"{p.map50_95:.4f}")
        
        # Состояние модели
        state_text, state_type = p.get_model_state()
        state_colors = {
            "success": ("#3fb950", "●"),
            "warning": ("#d29922", "●"),
            "danger": ("#f85149", "●"),
            "info": ("#58a6ff", "●")
        }
        color, icon = state_colors.get(state_type, ("#8b949e", "●"))
        self._model_state_label.setText(f"{icon} {state_text}")
        self._model_state_label.setStyleSheet(f"font-size: 18px; font-weight: bold; padding: 10px; color: {color};")
    
    def _get_trend_display(self, trend: str) -> tuple:
        """Возвращает (символ, цвет) для тренда."""
        if trend == "down":
            return ("▼", "#3fb950")  # Зелёный - хорошо
        elif trend == "up":
            return ("▲", "#f85149")  # Красный - плохо
        return ("━", "#8b949e")  # Серый - без изменений
    
    def _update_elapsed_time(self):
        """Обновляет отображение прошедшего времени."""
        import time
        if self._training_start_time > 0:
            elapsed = time.time() - self._training_start_time
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            secs = int(elapsed % 60)
            self._elapsed_label.setText(f"Время: {hours:02d}:{mins:02d}:{secs:02d}")
    
    def _format_time(self, seconds: float) -> str:
        """Форматирует секунды в читаемый вид."""
        if seconds < 60:
            return f"{int(seconds)} сек"
        elif seconds < 3600:
            mins = int(seconds // 60)
            return f"~{mins} мин"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"~{hours}ч {mins}м"
    
    def _on_log(self, msg):
        level = "info"
        if "error" in msg.lower(): level = "error"
        elif "warning" in msg.lower(): level = "warning"
        elif "success" in msg.lower(): level = "success"
        self.sig_log_msg.emit(msg, level)
        
    def _on_log_slot(self, msg, level):
        self._log.log(msg, level)
    
    def _on_finished(self, p):
        self.sig_training_complete.emit(p)

    def _on_finished_slot(self, p):
        # Останавливаем таймер
        self._elapsed_timer.stop()
        self._training_start_time = 0.0
        
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_status.setText("Обучение завершено!")
        self._progress.setValue(self._progress.maximum())
        
        # Обновляем состояние модели
        self._model_state_label.setText("● Обучение завершено")
        self._model_state_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #3fb950;")
        self._eta_label.setText("Готово!")
        
        if p.best_weights:
            self._log.success("Обучение завершено!")
            self._log.success(f"Лучшие веса: {p.best_weights}")
            self.training_finished.emit(p.best_weights)
            QMessageBox.information(
                self, "Готово!",
                f"Модель успешно обучена!\n\nВеса: {p.best_weights}"
            )
            
    def _on_error(self, err):
        self.sig_training_error.emit(err)

    def _on_error_slot(self, err):
        # Останавливаем таймер
        self._elapsed_timer.stop()
        self._training_start_time = 0.0
        
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_status.setText("Ошибка обучения")
        
        # Обновляем состояние модели
        self._model_state_label.setText("● Ошибка")
        self._model_state_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px; color: #f85149;")
        self._eta_label.setText("")
        
        self._log.error(err)
        QMessageBox.critical(self, "Ошибка", err)
    
    def set_dataset_path(self, path):
        yaml_path = os.path.join(path, DATA_YAML_FILE)
        if os.path.exists(yaml_path):
            self._lbl_dataset.setText(yaml_path)
            self._lbl_dataset.setProperty("cssClass", "success")
