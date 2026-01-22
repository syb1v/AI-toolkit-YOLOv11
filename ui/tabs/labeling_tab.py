"""
Вкладка разметки изображений.
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QGroupBox, QRadioButton, QButtonGroup,
    QSpinBox, QShortcut, QStyle, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QKeySequence

from ui.widgets.image_viewer import ImageViewer
from ui.widgets.class_tree import ClassTreeWidget
from core.class_manager import ClassManager
from models.annotation import Annotation, AnnotationType
from utils.export import DatasetExporter
from config import SUPPORTED_IMAGE_FORMATS, AUTOSAVE_FILE, AUTOSAVE_INTERVAL_MS


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


class LabelingTab(QWidget):
    """Вкладка для разметки изображений."""
    
    dataset_exported = pyqtSignal(str)
    
    def __init__(self, class_manager: ClassManager, parent=None):
        super().__init__(parent)
        
        self._class_manager = class_manager
        self._image_dir: Optional[Path] = None
        self._image_files: List[str] = []
        self._current_index: int = -1
        self._annotations: Dict[str, List[Annotation]] = {}
        
        self._current_class: Optional[str] = None
        self._current_color: str = "#FF0000"
        
        self._init_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._setup_autosave()
    
    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Центр
        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)
        
        # Правая панель
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([280, 650, 320])
        layout.addWidget(splitter)
    
    def _create_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("cssClass", "card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        
        # Заголовок
        header = QLabel("Изображения")
        header.setStyleSheet("font-weight: bold; font-size: 15px; color: #58a6ff;")
        layout.addWidget(header)
        
        # Кнопка открытия
        self._btn_open = QPushButton("Открыть папку...")
        self._btn_open.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self._btn_open.setToolTip("Выберите папку с изображениями (Ctrl+O)")
        self._btn_open.clicked.connect(self._open_folder)
        layout.addWidget(self._btn_open)
        
        # Путь
        self._lbl_path = QLabel("Папка не выбрана")
        self._lbl_path.setProperty("cssClass", "hint")
        self._lbl_path.setWordWrap(True)
        layout.addWidget(self._lbl_path)
        
        # Список файлов
        self._file_list = QListWidget()
        self._file_list.setToolTip("Изображения")
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        layout.addWidget(self._file_list)
        
        # Навигация
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(8)
        
        self._btn_prev = QPushButton("Назад")
        self._btn_prev.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self._btn_prev.setToolTip("Предыдущее (←, A)")
        self._btn_prev.clicked.connect(self._prev_image)
        nav_layout.addWidget(self._btn_prev)
        
        self._lbl_counter = QLabel("0 / 0")
        self._lbl_counter.setAlignment(Qt.AlignCenter)
        self._lbl_counter.setStyleSheet("font-weight: bold; font-size: 14px;")
        nav_layout.addWidget(self._lbl_counter, 1)
        
        self._btn_next = QPushButton("Вперед")
        self._btn_next.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self._btn_next.setToolTip("Следующее (→, D)")
        self._btn_next.clicked.connect(self._next_image)
        nav_layout.addWidget(self._btn_next)
        
        layout.addLayout(nav_layout)
        
        return panel
    
    def _create_center_panel(self) -> QWidget:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Toolbar
        toolbar = QFrame()
        toolbar.setProperty("cssClass", "toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)
        toolbar_layout.setSpacing(12)
        
        # Режим
        toolbar_layout.addWidget(QLabel("Инструменты:"))
        
        self._mode_group = QButtonGroup(self)
        
        self._rb_select = QRadioButton("Выбор")
        self._rb_select.setToolTip(
            "Режим выбора и редактирования (Z)\n\n"
            "• Клик по объекту — выделение\n"
            "• Перетаскивание — перемещение объекта\n"
            "• Ручки по краям — изменение размера"
        )
        self._rb_select.setChecked(True)
        self._mode_group.addButton(self._rb_select)
        toolbar_layout.addWidget(self._rb_select)
        
        self._rb_bbox = QRadioButton("Box")
        self._rb_bbox.setToolTip(
            "Рисование Bounding Box (X)\n\n"
            "Прямоугольники для задач детекции.\n"
            "Клик + перетаскивание для создания."
        )
        self._mode_group.addButton(self._rb_bbox)
        toolbar_layout.addWidget(self._rb_bbox)
        
        self._rb_polygon = QRadioButton("Polygon")
        self._rb_polygon.setToolTip(
            "Рисование полигонов (C)\n\n"
            "Для задач сегментации (точные границы объектов).\n\n"
            "• Клики — расставление точек\n"
            "• Enter или двойной клик — замкнуть фигуру\n"
            "• Esc — отменить текущую фигуру"
        )
        self._mode_group.addButton(self._rb_polygon)
        toolbar_layout.addWidget(self._rb_polygon)
        
        toolbar_layout.addStretch()
        
        self._cb_show_all = QCheckBox("Показать все")
        self._cb_show_all.setToolTip("Если выключено — показывает только объекты выбранного класса")
        self._cb_show_all.setChecked(True)
        self._cb_show_all.stateChanged.connect(self._on_show_all_toggled)
        toolbar_layout.addWidget(self._cb_show_all)
        
        layout.addWidget(toolbar)
        
        # ImageViewer
        self._viewer = ImageViewer()
        self._viewer.set_drawing_enabled(True)
        self._viewer.annotations_changed.connect(self._on_annotations_changed)
        layout.addWidget(self._viewer)
        
        # Статус
        status_layout = QHBoxLayout()
        self._lbl_status = QLabel("Откройте папку с изображениями")
        self._lbl_status.setStyleSheet("color: #8b949e;")
        status_layout.addWidget(self._lbl_status)
        
        self._lbl_mode = QLabel("")
        self._lbl_mode.setStyleSheet("color: #58a6ff; font-weight: bold;")
        status_layout.addWidget(self._lbl_mode)
        
        layout.addLayout(status_layout)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("cssClass", "card")
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Классы
        self._class_tree = ClassTreeWidget(self._class_manager)
        layout.addWidget(self._class_tree)
        
        # Экспорт
        export_group = QGroupBox("Экспорт")
        export_layout = QVBoxLayout(export_group)
        
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Val %:"))
        self._spin_val = QSpinBox()
        self._spin_val.setRange(0, 50)
        self._spin_val.setValue(20)
        self._spin_val.setToolTip("Процент для валидации (10-20%)")
        val_layout.addWidget(self._spin_val)
        export_layout.addLayout(val_layout)
        
        self._btn_export = QPushButton("Экспортировать датасет")
        self._btn_export.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self._btn_export.setProperty("cssClass", "primary")
        self._btn_export.setToolTip("Сохранить в формате YOLO (Ctrl+E)")
        self._btn_export.clicked.connect(self._export_dataset)
        export_layout.addWidget(self._btn_export)
        
        layout.addWidget(export_group)
        
        # Статистика аннотаций
        stats_group = QGroupBox("Статистика")
        stats_layout = QVBoxLayout(stats_group)
        
        self._stats_total_label = QLabel("Аннотаций: 0")
        self._stats_total_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self._stats_total_label)
        
        self._stats_images_label = QLabel("Размечено: 0 / 0 изображений")
        self._stats_images_label.setStyleSheet("color: #8b949e;")
        stats_layout.addWidget(self._stats_images_label)
        
        self._stats_by_class = QListWidget()
        self._stats_by_class.setMaximumHeight(100)
        self._stats_by_class.setStyleSheet("font-size: 12px;")
        stats_layout.addWidget(self._stats_by_class)
        
        layout.addWidget(stats_group)
        
        # Подсказки
        hints = QLabel(
            "<b>Горячие клавиши:</b><br>"
            "Z — выбор, X — box, C — polygon<br>"
            "← → A D — навигация<br>"
            "Delete — удалить<br>"
            "Ctrl+Z/Y — undo/redo<br>"
            "Enter — завершить полигон<br>"
            "F — вписать, Esc — отмена"
        )
        hints.setProperty("cssClass", "hint")
        hints.setWordWrap(True)
        hints.setStyleSheet("padding: 12px;")
        layout.addWidget(hints)
        
        layout.addStretch()
        
        return panel
    
    def _connect_signals(self):
        # Режим рисования
        self._rb_select.toggled.connect(lambda c: self._on_mode_changed("select") if c else None)
        self._rb_bbox.toggled.connect(lambda c: self._on_mode_changed("bbox") if c else None)
        self._rb_polygon.toggled.connect(lambda c: self._on_mode_changed("polygon") if c else None)
        
        # Классы
        self._class_tree.class_selected.connect(self._on_class_selected)
        self._class_tree.classes_modified.connect(self._on_classes_modified)
        
        # Viewer
        self._viewer.annotation_added.connect(self._on_annotation_added)
        self._viewer.annotation_modified.connect(self._on_annotation_modified)
        self._viewer.annotation_deleted.connect(self._on_annotation_deleted)
    
    def _setup_shortcuts(self):
        """Настраивает горячие клавиши."""
        # Файлы
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_folder)
        QShortcut(QKeySequence("Ctrl+E"), self, self._export_dataset)
        
        # Навигация
        QShortcut(QKeySequence(Qt.Key_Left), self, self._prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._next_image)
        QShortcut(QKeySequence(Qt.Key_A), self, self._prev_image)
        QShortcut(QKeySequence(Qt.Key_D), self, self._next_image)
        
        # Инструменты (Z/X/C - без модификаторов, не конфликтуют с Ctrl+Z)
        QShortcut(QKeySequence(Qt.Key_Z), self, lambda: self._rb_select.setChecked(True))
        QShortcut(QKeySequence(Qt.Key_X), self, lambda: self._rb_bbox.setChecked(True))
        QShortcut(QKeySequence(Qt.Key_C), self, lambda: self._rb_polygon.setChecked(True))
        
        # Действия
        QShortcut(QKeySequence(Qt.Key_F), self, lambda: self._viewer.fit_to_view())
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self._delete_selected)
    
    def _setup_autosave(self):
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._do_autosave)
        self._autosave_timer.start(AUTOSAVE_INTERVAL_MS)
        QTimer.singleShot(500, self._try_load_autosave)
    
    # === Режимы ===
    
    def _on_mode_changed(self, mode: str):
        self._viewer.set_drawing_mode(mode)
        modes = {"select": "Выбор", "bbox": "Box", "polygon": "Polygon"}
        self._lbl_mode.setText(modes.get(mode, ""))
    
    def _on_class_selected(self, class_name: str, color: str):
        """Выбран класс."""
        self._viewer.set_current_class(class_name, color)
        
        is_show_all = self._cb_show_all.isChecked()
        if not is_show_all:
             self._viewer.set_filter_state(active=True, class_name=class_name)
        else:
             self._viewer.set_filter_state(active=False, class_name=None)
        
        if self._viewer.drawing_mode == "select":
             self._rb_bbox.setChecked(True)
             self._viewer.set_drawing_mode("bbox")
             self._update_status()
    
    def _on_classes_modified(self):
        """Вызывается при изменении классов."""
        # Обновляем цвета на текущем изображении
        colors = self._class_tree.get_colors()
        for ann in self._viewer.get_annotations():
            if ann.class_name in colors:
                if ann.color != colors[ann.class_name]:
                    ann.color = colors[ann.class_name]
                    self._viewer.update_annotation(ann, save_undo=False)
    
    # === Аннотации ===
    
    def _on_annotation_added(self, annotation: Annotation):
        if self._current_index < 0:
            return
        filename = self._image_files[self._current_index]
        if filename not in self._annotations:
            self._annotations[filename] = []
        self._annotations[filename].append(annotation)
        self._update_file_list_item(self._current_index)
        self._update_status()
    
    def _on_annotation_modified(self, annotation: Annotation):
        if self._current_index < 0:
            return
        filename = self._image_files[self._current_index]
        if filename in self._annotations:
            for i, ann in enumerate(self._annotations[filename]):
                if ann.id == annotation.id:
                    self._annotations[filename][i] = annotation
                    break
    
    def _on_annotation_deleted(self, annotation_id: str):
        if self._current_index < 0:
            return
        filename = self._image_files[self._current_index]
        if filename in self._annotations:
            self._annotations[filename] = [
                a for a in self._annotations[filename] if a.id != annotation_id
            ]
            self._update_file_list_item(self._current_index)
            self._update_status()
    
    def _on_annotations_changed(self):
        """Синхронизирует аннотации из viewer в хранилище."""
        if self._current_index < 0:
            return
        filename = self._image_files[self._current_index]
        self._annotations[filename] = self._viewer.get_annotations().copy()
        self._update_file_list_item(self._current_index)
    
    def _on_file_selected(self, row: int):
        if row != self._current_index and row >= 0:
            self._load_image(row)
    
    def _update_file_list_item(self, index: int):
        if 0 <= index < len(self._image_files):
            item = self._file_list.item(index)
            if item:
                filename = self._image_files[index]
                count = len(self._annotations.get(filename, []))
                if count > 0:
                    item.setText(f"{filename} ({count})")
                    item.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
                else:
                    item.setText(filename)
                    item.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
    
    def _update_status(self):
        if self._current_index < 0:
            return
        filename = self._image_files[self._current_index]
        count = len(self._annotations.get(filename, []))
        self._lbl_status.setText(f"{filename} — {count} аннотаций")
        self._update_stats()
    
    def _update_stats(self):
        """Обновляет панель статистики аннотаций."""
        # Общее количество аннотаций
        total_annotations = sum(len(anns) for anns in self._annotations.values())
        self._stats_total_label.setText(f"Аннотаций: {total_annotations}")
        
        # Количество размеченных изображений
        labeled_count = sum(1 for anns in self._annotations.values() if len(anns) > 0)
        total_images = len(self._image_files)
        self._stats_images_label.setText(f"Размечено: {labeled_count} / {total_images}")
        
        # По классам
        class_counts = {}
        for anns in self._annotations.values():
            for ann in anns:
                class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1
        
        self._stats_by_class.clear()
        for cls, count in sorted(class_counts.items()):
            color = self._class_tree.get_colors().get(cls, "#888888")
            self._stats_by_class.addItem(f"● {cls}: {count}")
    
    # === Undo/Redo ===
    
    def _undo(self):
        if self._viewer.undo():
            self._on_annotations_changed()
    
    def _redo(self):
        if self._viewer.redo():
            self._on_annotations_changed()
    
    def _delete_selected(self):
        self._viewer.delete_selected()
    
    # === Файлы ===
    
    def _open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if path:
            self._load_folder(Path(path))
    
    def _load_folder(self, path: Path):
        self._image_dir = path
        self._lbl_path.setText(str(path))
        
        files = []
        for f in path.iterdir():
            if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
                files.append(f.name)
        
        files.sort(key=natural_sort_key)
        self._image_files = files
        
        self._file_list.clear()
        for f in files:
            count = len(self._annotations.get(f, []))
            item = QListWidgetItem(f"{f}")
            if count > 0:
                item.setText(f"{f} ({count})")
                item.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
            else:
                item.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
            self._file_list.addItem(item)
        
        if files:
            self._load_image(0)
            self._lbl_status.setText(f"{len(files)} изображений")
        else:
            QMessageBox.warning(self, "Внимание", "В папке нет изображений")
        
        self._update_counter()
    
    def _load_image(self, index: int):
        if not (0 <= index < len(self._image_files)):
            return
        
        # Сохраняем текущие аннотации
        if 0 <= self._current_index < len(self._image_files):
            old_filename = self._image_files[self._current_index]
            self._annotations[old_filename] = self._viewer.get_annotations().copy()
        
        self._current_index = index
        filename = self._image_files[index]
        path = self._image_dir / filename
        
        if self._viewer.load_image(str(path)):
            # Загружаем аннотации
            annotations = self._annotations.get(filename, [])
            self._viewer.set_annotations(annotations)
            self._update_status()
        else:
            self._lbl_status.setText(f"Ошибка: {filename}")
        
        self._file_list.setCurrentRow(index)
        self._update_counter()
    
    def _prev_image(self):
        if self._current_index > 0:
            self._load_image(self._current_index - 1)
    
    def _next_image(self):
        if self._current_index < len(self._image_files) - 1:
            self._load_image(self._current_index + 1)
    
    def _update_counter(self):
        total = len(self._image_files)
        current = self._current_index + 1 if self._current_index >= 0 else 0
        self._lbl_counter.setText(f"{current} / {total}")
    
    # === Экспорт ===
    def _on_show_all_toggled(self, state: int):
        """Переключение режима 'Показать все'."""
        is_active = (state != Qt.Checked)
        class_name = None
        
        current_item = self._class_tree.selectedItems()
        if current_item:
            class_name = current_item[0].text(0)
            
        self._viewer.set_filter_state(active=is_active, class_name=class_name)
    
    def _export_dataset(self):
        if not self._image_dir:
            QMessageBox.warning(self, "Внимание", "Сначала откройте папку")
            return
        
        # Сохраняем текущие
        if self._current_index >= 0:
            filename = self._image_files[self._current_index]
            self._annotations[filename] = self._viewer.get_annotations().copy()
        
        class_set = self._class_tree.get_current_set()
        if not class_set:
            QMessageBox.warning(self, "Внимание", "Выберите набор классов")
            return
        
        output_path = QFileDialog.getExistingDirectory(self, "Папка для датасета")
        if not output_path:
            return
        
        try:
            class_mapping = self._class_tree.get_class_mapping()
            val_split = self._spin_val.value() / 100.0
            
            exporter = DatasetExporter(
                source_dir=str(self._image_dir),
                output_dir=output_path,
                annotations=self._annotations,
                class_mapping=class_mapping,
                val_split=val_split
            )
            
            stats = exporter.export()
            
            QMessageBox.information(
                self, "Успех!",
                f"Датасет экспортирован!\n\n"
                f"Путь: {stats.export_path}\n"
                f"Всего: {stats.total_images} изображений\n"
                f"Train: {stats.train_images}, Val: {stats.val_images}\n"
                f"Аннотаций: {stats.total_annotations}"
            )
            
            self.dataset_exported.emit(output_path)
            
            if os.path.exists(AUTOSAVE_FILE):
                os.remove(AUTOSAVE_FILE)
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка: {e}")
    
    # === Автосохранение ===
    
    def _do_autosave(self):
        if not self._image_dir:
            return
        
        # Сохраняем текущие
        if 0 <= self._current_index < len(self._image_files):
            filename = self._image_files[self._current_index]
            self._annotations[filename] = self._viewer.get_annotations().copy()
        
        if not self._annotations:
            return
        
        data = {
            "image_dir": str(self._image_dir),
            "current_index": self._current_index,
            "annotations": {}
        }
        
        for filename, anns in self._annotations.items():
            data["annotations"][filename] = [
                {
                    "id": a.id,
                    "class_name": a.class_name,
                    "type": a.annotation_type.value,
                    "coords": a.coords,
                    "color": a.color
                }
                for a in anns
            ]
        
        try:
            with open(AUTOSAVE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Autosave error: {e}")
    
    def _try_load_autosave(self):
        if not os.path.exists(AUTOSAVE_FILE):
            return
        
        reply = QMessageBox.question(
            self, "Восстановление",
            "Найдены несохранённые данные.\nВосстановить?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            os.remove(AUTOSAVE_FILE)
            return
        
        try:
            with open(AUTOSAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._annotations = {}
            for filename, anns_data in data.get("annotations", {}).items():
                self._annotations[filename] = [
                    Annotation(
                        id=a["id"],
                        class_name=a["class_name"],
                        annotation_type=AnnotationType(a["type"]),
                        coords=a["coords"],
                        color=a["color"]
                    )
                    for a in anns_data
                ]
            
            image_dir = Path(data.get("image_dir", ""))
            if image_dir.exists():
                self._load_folder(image_dir)
                idx = data.get("current_index", 0)
                if 0 <= idx < len(self._image_files):
                    self._load_image(idx)
                    
        except Exception as e:
            print(f"Autosave load error: {e}")
