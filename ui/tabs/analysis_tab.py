"""
Вкладка анализа изображений.
"""
import os
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QPushButton, QSlider, QFileDialog,
    QMessageBox, QGroupBox, QListWidget, QStyle
)
from PyQt5.QtCore import Qt

from ui.widgets.image_viewer import ImageViewer
from core.predictor import YOLOPredictor, PredictionResult
from utils.feedback_saver import FeedbackSaver
from config import SUPPORTED_IMAGE_FILTER


class AnalysisTab(QWidget):
    """Вкладка анализа изображений."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._predictor: Optional[YOLOPredictor] = None
        self._current_image: Optional[str] = None
        self._last_result: Optional[PredictionResult] = None
        self._feedback_saver = FeedbackSaver()
        self._init_ui()
    
    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая - просмотр
        left = self._create_viewer_panel()
        splitter.addWidget(left)
        
        # Правая - настройки
        right = self._create_controls_panel()
        splitter.addWidget(right)
        
        splitter.setSizes([700, 350])
        layout.addWidget(splitter)
    
    def _create_viewer_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Toolbar
        toolbar = QFrame()
        toolbar.setProperty("cssClass", "toolbar")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 8, 12, 8)
        tb_layout.setSpacing(12)
        
        self._btn_load_img = QPushButton("Открыть фото")
        self._btn_load_img.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self._btn_load_img.setToolTip("Выберите изображение для анализа")
        self._btn_load_img.clicked.connect(self._load_image)
        tb_layout.addWidget(self._btn_load_img)
        
        self._btn_analyze = QPushButton("Анализировать")
        self._btn_analyze.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._btn_analyze.setProperty("cssClass", "primary")
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(self._run_analysis)
        tb_layout.addWidget(self._btn_analyze)
        
        self._lbl_image = QLabel("Загрузите изображение")
        self._lbl_image.setProperty("cssClass", "hint")
        tb_layout.addWidget(self._lbl_image)
        tb_layout.addStretch()
        
        layout.addWidget(toolbar)
        
        # Viewer
        self._viewer = ImageViewer()
        self._viewer.set_drawing_enabled(False)
        layout.addWidget(self._viewer)
        
        # Статус
        self._lbl_status = QLabel("Загрузите модель и изображение для анализа")
        self._lbl_status.setStyleSheet("color: #8b949e; padding: 8px;")
        layout.addWidget(self._lbl_status)
        
        return panel
    
    def _create_controls_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("cssClass", "card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Заголовок
        header = QLabel("Анализ изображений")
        header.setStyleSheet("font-weight: bold; font-size: 15px; color: #58a6ff;")
        layout.addWidget(header)
        
        # Модель
        model_grp = QGroupBox("Модель YOLO")
        model_lay = QVBoxLayout(model_grp)
        
        self._btn_load_model = QPushButton("Загрузить веса (.pt)")
        self._btn_load_model.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self._btn_load_model.setToolTip("Выберите файл обученной модели")
        self._btn_load_model.clicked.connect(self._load_model)
        model_lay.addWidget(self._btn_load_model)
        
        self._lbl_model = QLabel("Модель не загружена")
        self._lbl_model.setProperty("cssClass", "warning")
        model_lay.addWidget(self._lbl_model)
        layout.addWidget(model_grp)
        
        # Настройки
        settings_grp = QGroupBox("Настройки")
        settings_lay = QVBoxLayout(settings_grp)
        
        # Confidence hint
        conf_hint = QLabel("Чем ниже уверенность — тем больше объектов\nбудет обнаружено, но возможны ложные срабатывания")
        conf_hint.setProperty("cssClass", "hint")
        conf_hint.setWordWrap(True)
        settings_lay.addWidget(conf_hint)
        
        # Confidence
        conf_lay = QHBoxLayout()
        conf_lay.addWidget(QLabel("Уверенность:"))
        self._slider_conf = QSlider(Qt.Horizontal)
        self._slider_conf.setRange(1, 100)
        self._slider_conf.setValue(25)
        self._slider_conf.setToolTip(
            "Минимальный порог уверенности (confidence threshold).\n\n"
            "• Ниже значение = больше детекций, но больше ложных срабатываний\n"
            "• Выше значение = только уверенные детекции, меньше объектов\n\n"
            "Рекомендации: 25% для начала, затем подстройка"
        )
        self._slider_conf.valueChanged.connect(self._on_conf_changed)
        conf_lay.addWidget(self._slider_conf)
        self._lbl_conf = QLabel("25%")
        self._lbl_conf.setMinimumWidth(40)
        conf_lay.addWidget(self._lbl_conf)
        settings_lay.addLayout(conf_lay)
        layout.addWidget(settings_grp)
        
        # Результаты
        results_grp = QGroupBox("Результаты")
        results_lay = QVBoxLayout(results_grp)
        
        self._list_results = QListWidget()
        self._list_results.setMaximumHeight(150)
        self._list_results.itemClicked.connect(self._on_result_clicked)
        results_lay.addWidget(self._list_results)
        
        self._lbl_time = QLabel("")
        self._lbl_time.setProperty("cssClass", "hint")
        results_lay.addWidget(self._lbl_time)
        
        self._lbl_total = QLabel("")
        self._lbl_total.setStyleSheet("font-weight: bold;")
        results_lay.addWidget(self._lbl_total)
        layout.addWidget(results_grp)
        
        # Коррекция
        corr_grp = QGroupBox("Режим коррекции")
        corr_lay = QVBoxLayout(corr_grp)
        
        corr_hint = QLabel("Включите для ручного исправления\nпредсказаний модели")
        corr_hint.setProperty("cssClass", "hint")
        corr_lay.addWidget(corr_hint)
        
        self._btn_correct = QPushButton("Включить правку")
        self._btn_correct.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self._btn_correct.setCheckable(True)
        self._btn_correct.setToolTip(
            "Включить режим редактирования предсказаний:\n\n"
            "• Выделяйте и перемещайте объекты\n"
            "• Удаляйте ложные срабатывания\n"
            "• Изменяйте размеры рамок\n\n"
            "Исправления можно сохранить для дообучения модели."
        )
        self._btn_correct.clicked.connect(self._toggle_correction)
        corr_lay.addWidget(self._btn_correct)
        
        self._btn_delete = QPushButton("Удалить выбранное")
        self._btn_delete.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self._btn_delete.setToolTip("Удалить выбранную аннотацию")
        self._btn_delete.clicked.connect(lambda: self._viewer.delete_selected())
        corr_lay.addWidget(self._btn_delete)
        
        # Кнопка сохранения для дообучения
        self._btn_save_feedback = QPushButton("Сохранить для дообучения")
        self._btn_save_feedback.setToolTip(
            "Сохранить исправления для дообучения модели:\n\n"
            "• Изображение копируется в feedback_data/images/\n"
            "• Разметка сохраняется в feedback_data/labels/\n"
            "• Формат YOLO (class x_center y_center width height)\n\n"
            "Используйте эти данные для добавления в датасет при дообучении."
        )
        self._btn_save_feedback.clicked.connect(self._save_for_retraining)
        corr_lay.addWidget(self._btn_save_feedback)
        
        self._lbl_feedback_stats = QLabel("")
        self._lbl_feedback_stats.setProperty("cssClass", "hint")
        self._lbl_feedback_stats.setStyleSheet("color: #8b949e; font-size: 11px;")
        corr_lay.addWidget(self._lbl_feedback_stats)
        
        layout.addWidget(corr_grp)
        
        layout.addStretch()
        return panel
    
    def _on_conf_changed(self, value: int):
        self._lbl_conf.setText(f"{value}%")
    
    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите веса модели", "", "Model (*.pt)"
        )
        if not path:
            return
        
        try:
            self._predictor = YOLOPredictor(path)
            self._lbl_model.setText(f"✅ {Path(path).name}")
            self._lbl_model.setProperty("cssClass", "success")
            self._lbl_status.setText("✅ Модель загружена, выберите изображение")
            self._check_ready()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")
    
    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", SUPPORTED_IMAGE_FILTER
        )
        if not path:
            return
        
        if self._viewer.load_image(path):
            self._current_image = path
            self._lbl_image.setText(f"📷 {Path(path).name}")
            self._lbl_image.setProperty("cssClass", "success")
            self._check_ready()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение")
    
    def _check_ready(self):
        ready = self._predictor is not None and self._current_image is not None
        self._btn_analyze.setEnabled(ready)
        if ready:
            self._lbl_status.setText("Готово к анализу! Нажмите 'Анализировать'")
    
    def _run_analysis(self):
        if not self._predictor or not self._current_image:
            return
        
        self._lbl_status.setText("Выполняется анализ...")
        
        try:
            conf = self._slider_conf.value() / 100.0
            result = self._predictor.predict(self._current_image, conf=conf)
            self._last_result = result
            
            # Отображаем
            self._viewer.set_annotations(result.annotations)
            
            # Список классов
            self._list_results.clear()
            for cls, count in sorted(result.class_counts.items()):
                self._list_results.addItem(f"{cls}: {count}")
            
            # Статистика
            total = sum(result.class_counts.values())
            self._lbl_total.setText(f"Найдено объектов: {total}")
            self._lbl_time.setText(f"Время: {result.inference_time_ms:.1f} мс")
            self._lbl_status.setText(f"Анализ завершён — {total} объектов")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            self._lbl_status.setText("Ошибка анализа")
    
    
    def _on_result_clicked(self, item):
        """Фильтрация по клику на результат."""
        text = item.text()
        if ":" in text:
            class_name = text.split(":")[0].strip()
            self._viewer.set_filter_state(active=True, class_name=class_name)
    
    def _toggle_correction(self, checked: bool):
        self._viewer.set_drawing_enabled(checked)
        if checked:
            self._btn_correct.setText("🔒 Выключить правку")
            self._lbl_status.setText("✏️ Режим правки включён — выберите объект для редактирования")
            self._viewer.set_drawing_mode("select") # По умолчанию 'select' для редактирования
        else:
            self._btn_correct.setText("🔓 Включить правку")
            self._lbl_status.setText("🔍 Режим просмотра")
            self._viewer.set_filter_state(active=False, class_name=None) # Сброс фильтра при выходе
    
    def load_model_from_path(self, path: str):
        """Загружает модель по пути (вызов извне)."""
        if os.path.exists(path):
            try:
                self._predictor = YOLOPredictor(path)
                self._lbl_model.setText(f"✅ {Path(path).name}")
                self._lbl_model.setProperty("cssClass", "success")
                self._lbl_status.setText("✅ Модель загружена автоматически!")
                self._check_ready()
            except:
                pass
    
    def _save_for_retraining(self):
        """Сохраняет исправленные аннотации для дообучения."""
        if not self._current_image:
            QMessageBox.warning(self, "Ошибка", "Нет загруженного изображения")
            return
        
        annotations = self._viewer.get_annotations()
        if not annotations:
            QMessageBox.warning(self, "Ошибка", "Нет аннотаций для сохранения")
            return
        
        # Получаем классы из модели
        if self._predictor and hasattr(self._predictor, 'model'):
            try:
                names = self._predictor.model.names
                if isinstance(names, dict):
                    classes = list(names.values())
                else:
                    classes = list(names)
            except:
                # Собираем классы из аннотаций
                classes = list(set(ann.class_name for ann in annotations))
        else:
            classes = list(set(ann.class_name for ann in annotations))
        
        if not classes:
            QMessageBox.warning(self, "Ошибка", "Не удалось определить классы")
            return
        
        # Получаем размер изображения
        # Получаем размер изображения
        img_size = self._viewer.image_size
        if not img_size or img_size == (0, 0):
            QMessageBox.warning(self, "Ошибка", "Не удалось получить размер изображения")
            return
        
        try:
            img_w, img_h = img_size
            self._feedback_saver.save(
                image_path=self._current_image,
                annotations=annotations,
                classes=classes,
                image_size=(img_w, img_h)
            )
            
            stats = self._feedback_saver.get_stats()
            self._lbl_feedback_stats.setText(
                f"Сохранено: {stats['images']} фото, {stats['labels']} файлов разметки"
            )
            
            QMessageBox.information(self, "Успех", "Данные сохранены для дообучения!")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

