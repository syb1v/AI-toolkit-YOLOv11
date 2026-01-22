"""
Виджет для просмотра и аннотирования изображений.
Полностью переработанный с правильной логикой редактирования.
"""
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import copy

from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsPixmapItem, QWidget, QVBoxLayout,
    QApplication, QGraphicsPathItem
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPixmap, QPen, QBrush, QColor, QPainter, QPolygonF,
    QFont, QCursor, QWheelEvent, QMouseEvent, QKeyEvent, QPainterPath
)

from models.annotation import Annotation, AnnotationType
from config import MIN_BBOX_SIZE, HANDLE_SIZE, POLYGON_CLOSE_DISTANCE


class DrawingMode(Enum):
    """Режим рисования."""
    SELECT = "select"      # Выбор и редактирование
    BBOX = "bbox"          # Рисование прямоугольников
    POLYGON = "polygon"    # Рисование полигонов


class EditHandle(QGraphicsEllipseItem):
    """Точка редактирования для изменения размера."""
    
    def __init__(self, handle_id: str, parent=None):
        super().__init__(parent)
        self.handle_id = handle_id  # 'nw', 'ne', 'sw', 'se' или индекс точки
        self.setRect(-HANDLE_SIZE/2, -HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE)
        self.setBrush(QBrush(QColor("#58a6ff")))
        self.setPen(QPen(QColor("#ffffff"), 2))
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setZValue(1000)
        self.setCursor(QCursor(Qt.CrossCursor))


@dataclass
class UndoState:
    """Состояние для Undo/Redo."""
    annotations: List[Annotation]
    description: str


class ImageViewer(QWidget):
    """
    Виджет просмотра и аннотирования изображений.
    
    Сигналы:
        annotation_added: Annotation
        annotation_modified: Annotation  
        annotation_deleted: str (id)
        annotation_selected: Annotation | None
        annotations_changed: список изменился
    """
    
    annotation_added = pyqtSignal(object)
    annotation_modified = pyqtSignal(object)
    annotation_deleted = pyqtSignal(str)
    annotation_selected = pyqtSignal(object)
    annotations_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Состояние
        self._image_path: Optional[str] = None
        self._pixmap: Optional[QPixmap] = None
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        
        # Аннотации
        self._annotations: Dict[str, Annotation] = {}
        self._graphics_items: Dict[str, QGraphicsItem] = {}
        self._label_items: Dict[str, QGraphicsTextItem] = {}
        
        # Выбор
        self._selected_ids: set[str] = set()
        self._handles: List[EditHandle] = []
        
        # Режим
        self._mode = DrawingMode.SELECT
        self._drawing_enabled = False
        self._current_class: Optional[str] = None
        self._current_color: str = "#FF0000"
        
        # Фильтр
        self._filter_active: bool = False
        self._filter_class: Optional[str] = None  # Класс для фильтрации
        
        # Рисование
        self._is_drawing = False
        self._draw_start: Optional[QPointF] = None
        self._temp_item: Optional[QGraphicsItem] = None
        self._polygon_points: List[QPointF] = []
        self._polygon_preview_item: Optional[QGraphicsPolygonItem] = None
        self._polygon_dots: List[QGraphicsEllipseItem] = []
        
        # Редактирование
        self._is_dragging = False
        self._drag_start: Optional[QPointF] = None
        self._drag_handle: Optional[EditHandle] = None
        self._drag_start: Optional[QPointF] = None
        self._drag_handle: Optional[EditHandle] = None
        self._original_annotations_map: Dict[str, Annotation] = {}
        
        # Undo/Redo
        self._undo_stack: List[UndoState] = []
        self._redo_stack: List[UndoState] = []
        self._max_undo = 50
        
        # UI
        self._init_ui()
        self.setFocusPolicy(Qt.StrongFocus)
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._scene = QGraphicsScene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setRenderHint(QPainter.SmoothPixmapTransform)
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setDragMode(QGraphicsView.NoDrag)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._view.setBackgroundBrush(QBrush(QColor("#0d1117")))
        
        # События мыши
        self._view.wheelEvent = self._on_wheel
        self._view.mousePressEvent = self._on_mouse_press
        self._view.mouseMoveEvent = self._on_mouse_move
        self._view.mouseReleaseEvent = self._on_mouse_release
        self._view.mouseDoubleClickEvent = self._on_double_click
        
        layout.addWidget(self._view)
    
    # === Публичные методы ===
    
    def load_image(self, path: str) -> bool:
        """Загружает изображение."""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        
        self.clear()
        
        self._image_path = path
        self._pixmap = pixmap
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setZValue(-1000)
        self._scene.addItem(self._pixmap_item)
        
        self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        return True
    
    def clear(self):
        """Очищает сцену."""
        self._clear_temp_drawing()
        self._clear_handles()
        
        for item in self._graphics_items.values():
            self._scene.removeItem(item)
        for item in self._label_items.values():
            self._scene.removeItem(item)
        
        self._scene.clear()
        
        self._pixmap_item = None
        self._pixmap = None
        self._image_path = None
        self._annotations.clear()
        self._graphics_items.clear()
        self._label_items.clear()
        self._selected_ids.clear()
        self._selected_ids.clear()
        self._filter_active = False
        self._filter_class = None
        self._undo_stack.clear()
        self._redo_stack.clear()
    
    def set_drawing_enabled(self, enabled: bool):
        """Включает/выключает режим рисования."""
        self._drawing_enabled = enabled
        if not enabled:
            self._clear_temp_drawing()
            self._mode = DrawingMode.SELECT
    
    def set_drawing_mode(self, mode: str):
        """Устанавливает режим: 'select', 'bbox', 'polygon'."""
        self._clear_temp_drawing()
        if mode == "bbox":
            self._mode = DrawingMode.BBOX
        elif mode == "polygon":
            self._mode = DrawingMode.POLYGON
        else:
            self._mode = DrawingMode.SELECT
    
    @property
    def drawing_mode(self) -> str:
        """Возвращает текущий режим рисования."""
        # Convert enum back to string used in set_drawing_mode or simplified
        if self._mode == DrawingMode.BBOX:
            return "bbox"
        elif self._mode == DrawingMode.POLYGON:
            return "polygon"
        return "select"
    
    def set_current_class(self, class_name: str, color: str):
        """Устанавливает текущий класс."""
        self._current_class = class_name
        self._current_color = color
    
    
    def set_filter_state(self, active: bool, class_name: Optional[str]):
        """
        Устанавливает состояние фильтра.
        active: включен ли фильтр ("Показать все" выключено)
        class_name: имя класса для отображения (если check, то только этот класс)
        """
        self._filter_active = active
        self._filter_class = class_name
        self._update_visibility()
    
    def _update_visibility(self):
        """Обновляет видимость аннотаций."""
        if not self._filter_active:
            # Если фильтр выключен - показываем все
            for aid, item in self._graphics_items.items():
                item.setVisible(True)
            for aid, label in self._label_items.items():
                label.setVisible(True)
            return

        # Если фильтр включен
        target_class = self._filter_class
        
        # Сначала снимаем выделение со скрываемых объектов
        if self._selected_ids:
            ids_to_deselect = []
            for aid in self._selected_ids:
                if aid in self._annotations:
                    ann = self._annotations[aid]
                    visible = (ann.class_name == target_class) if target_class else False
                    if not visible:
                        ids_to_deselect.append(aid)
            
            for aid in ids_to_deselect:
                self.select_annotation(aid, toggle=True)
        
        for aid, ann in self._annotations.items():
            visible = False
            if target_class is not None:
                visible = (ann.class_name == target_class)
            
            if aid in self._graphics_items:
                self._graphics_items[aid].setVisible(visible)
            if aid in self._label_items:
                self._label_items[aid].setVisible(visible)
    
    def add_annotation(self, annotation: Annotation, emit_signal: bool = True, save_undo: bool = True):
        """Добавляет аннотацию."""
        if save_undo:
            self._save_undo_state("Добавление аннотации")
        
        self._annotations[annotation.id] = annotation
        self._create_graphics_item(annotation)
        
        if self._filter_active:
             # Если добавляемый объект не соответствует фильтру или фильтр "ничего"
             visible = False
             if self._filter_class is not None:
                 visible = (annotation.class_name == self._filter_class)
             
             if not visible:
                 if annotation.id in self._graphics_items:
                     self._graphics_items[annotation.id].setVisible(False)
                 if annotation.id in self._label_items:
                     self._label_items[annotation.id].setVisible(False)
        
        if emit_signal:
            self.annotation_added.emit(annotation)
            self.annotations_changed.emit()
    
    def remove_annotation(self, annotation_id: str, save_undo: bool = True):
        """Удаляет аннотацию."""
        if annotation_id not in self._annotations:
            return
        
        if save_undo:
            self._save_undo_state("Удаление аннотации")
        
        # Удаляем графику
        if annotation_id in self._graphics_items:
            self._scene.removeItem(self._graphics_items.pop(annotation_id))
        if annotation_id in self._label_items:
            self._scene.removeItem(self._label_items.pop(annotation_id))
        
        del self._annotations[annotation_id]
        
        if annotation_id in self._selected_ids:
            self._selected_ids.remove(annotation_id)
            if not self._selected_ids:
                self._clear_handles()
            else:
                self._update_handles_for_selection()
        
        self.annotation_deleted.emit(annotation_id)
        self.annotations_changed.emit()
    
    def update_annotation(self, annotation: Annotation, save_undo: bool = True):
        """Обновляет аннотацию."""
        if annotation.id not in self._annotations:
            return
        
        if save_undo:
            self._save_undo_state("Изменение аннотации")
        
        self._annotations[annotation.id] = annotation
        self._update_graphics_item(annotation)
        
        self.annotation_modified.emit(annotation)
        self.annotations_changed.emit()
    
    def get_annotations(self) -> List[Annotation]:
        """Возвращает список аннотаций."""
        return list(self._annotations.values())
    
    def set_annotations(self, annotations: List[Annotation]):
        """Устанавливает список аннотаций."""
        # Очищаем старые
        for aid in list(self._graphics_items.keys()):
            self._scene.removeItem(self._graphics_items.pop(aid))
        for aid in list(self._label_items.keys()):
            self._scene.removeItem(self._label_items.pop(aid))
        
        self._annotations.clear()
        self._selected_ids.clear()
        self._clear_handles()
        
        # Добавляем новые
        for ann in annotations:
            self._annotations[ann.id] = ann
            self._create_graphics_item(ann)
        
        # Применяем текущий фильтр после загрузки
        if self._filter_active:
            self._update_visibility()
    
    def select_annotation(self, annotation_id: Optional[str], toggle: bool = False):
        """Выбирает аннотацию.
        
        Args:
            annotation_id: ID аннотации или None.
            toggle: Если True, инвертирует выбор (для Ctrl+Click).
        """
        old_selection = set(self._selected_ids)
        
        if annotation_id is None:
            if not toggle:
                self._selected_ids.clear()
        else:
            if toggle:
                if annotation_id in self._selected_ids:
                    self._selected_ids.remove(annotation_id)
                else:
                    self._selected_ids.add(annotation_id)
            else:
                self._selected_ids.clear()
                self._selected_ids.add(annotation_id)
        
        # Определяем, что изменилось, чтобы обновить стиль
        changed_ids = old_selection.symmetric_difference(self._selected_ids)
        
        for aid in changed_ids:
            if aid in self._annotations and aid in self._graphics_items:
                ann = self._annotations[aid]
                item = self._graphics_items[aid]
                self._update_item_style(item, ann, selected=(aid in self._selected_ids))
        
        self._update_handles_for_selection()
        
        # Сигнал для последнего выбранного или None
        last_selected = None
        if annotation_id and annotation_id in self._selected_ids:
            last_selected = self._annotations[annotation_id]
        elif self._selected_ids:
            # Если сняли выделение, берем любой другой (или последний добавленный)
            # В данном случае просто берем любой, так как порядок в set не гарантирован
            # Но для UI лучше вернуть что-то валидное
            last_id = next(iter(self._selected_ids))
            last_selected = self._annotations[last_id]
            
        self.annotation_selected.emit(last_selected)

    def _update_handles_for_selection(self):
        """Обновляет ручки редактирования."""
        self._clear_handles()
        
        # Показываем ручки только если выбран ровно один объект
        if len(self._selected_ids) == 1:
            aid = next(iter(self._selected_ids))
            if aid in self._annotations:
                self._create_handles(self._annotations[aid])
    
    def get_selected_annotation(self) -> Optional[Annotation]:
        """Возвращает последнюю выбранную аннотацию (для совместимости)."""
        if self._selected_ids:
            # Возвращаем произвольный из выбранных (обычно нужен для свойств)
            aid = next(iter(self._selected_ids))
            return self._annotations.get(aid)
        return None
        
    def get_selected_annotations(self) -> List[Annotation]:
        """Возвращает список всех выбранных аннотаций."""
        return [self._annotations[aid] for aid in self._selected_ids if aid in self._annotations]
    
    def delete_selected(self):
        """Удаляет все выбранные аннотации."""
        if not self._selected_ids:
            return
            
        self._save_undo_state("Удаление объектов")
        
        # Копируем ID, так как remove_annotation меняет set
        ids_to_delete = list(self._selected_ids)
        for aid in ids_to_delete:
            self.remove_annotation(aid, save_undo=False)
        
        self._selected_ids.clear()
        self._clear_handles()
        self.annotation_selected.emit(None)
    

    
    def update_annotation_color(self, class_name: str, new_color: str):
        """Обновляет цвет всех аннотаций класса."""
        for ann in self._annotations.values():
            if ann.class_name == class_name:
                ann.color = new_color
                self._update_graphics_item(ann)
    
    def remove_annotations_by_class(self, class_name: str):
        """Удаляет все аннотации класса."""
        to_remove = [aid for aid, ann in self._annotations.items() if ann.class_name == class_name]
        if to_remove:
            self._save_undo_state(f"Удаление аннотаций класса {class_name}")
            for aid in to_remove:
                self.remove_annotation(aid, save_undo=False)
    
    def fit_to_view(self):
        """Подгоняет изображение."""
        if self._pixmap_item:
            self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
    
    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        if self._pixmap:
            return (self._pixmap.width(), self._pixmap.height())
        return None
    
    # === Undo/Redo ===
    
    def _save_undo_state(self, description: str):
        """Сохраняет состояние для Undo."""
        state = UndoState(
            annotations=[copy.deepcopy(a) for a in self._annotations.values()],
            description=description
        )
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
    
    def undo(self) -> bool:
        """Отменяет последнее действие."""
        if not self._undo_stack:
            return False
        
        # Сохраняем текущее состояние в redo
        current = UndoState(
            annotations=[copy.deepcopy(a) for a in self._annotations.values()],
            description="Current"
        )
        self._redo_stack.append(current)
        
        # Восстанавливаем
        state = self._undo_stack.pop()
        self.set_annotations(state.annotations)
        self.annotations_changed.emit()
        return True
    
    def redo(self) -> bool:
        """Повторяет отменённое действие."""
        if not self._redo_stack:
            return False
        
        # Сохраняем в undo
        current = UndoState(
            annotations=[copy.deepcopy(a) for a in self._annotations.values()],
            description="Current"
        )
        self._undo_stack.append(current)
        
        # Восстанавливаем
        state = self._redo_stack.pop()
        self.set_annotations(state.annotations)
        self.annotations_changed.emit()
        return True
    
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0
    
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0
    
    # === Графические элементы ===
    
    def _create_graphics_item(self, ann: Annotation):
        """Создаёт графический элемент для аннотации."""
        color = QColor(ann.color)
        
        if ann.annotation_type == AnnotationType.BBOX:
            x, y, w, h = ann.coords
            item = QGraphicsRectItem(x, y, w, h)
        else:
            points = [QPointF(p[0], p[1]) for p in ann.coords]
            item = QGraphicsPolygonItem(QPolygonF(points))
        
        # Стиль
        pen = QPen(color, 2)
        item.setPen(pen)
        fill_color = QColor(color)
        fill_color.setAlpha(40)
        item.setBrush(QBrush(fill_color))
        item.setZValue(10)
        
        self._scene.addItem(item)
        self._graphics_items[ann.id] = item
        
        # Метка
        label = QGraphicsTextItem(ann.class_name)
        label.setDefaultTextColor(QColor("#ffffff"))
        font = QFont("Segoe UI", 10, QFont.Bold)
        label.setFont(font)
        label.setZValue(20)
        
        # Позиция метки
        if ann.annotation_type == AnnotationType.BBOX:
            label.setPos(ann.coords[0] + 2, ann.coords[1] - 22)
        else:
            label.setPos(ann.coords[0][0], ann.coords[0][1] - 22)
        
        self._scene.addItem(label)
        self._label_items[ann.id] = label
        
        # Применяем фильтр
        if self._filter_active:
            visible = False
            if self._filter_class is not None:
                visible = (ann.class_name == self._filter_class)
            
            if not visible:
                item.setVisible(False)
                label.setVisible(False)
    
    def _update_graphics_item(self, ann: Annotation):
        """Обновляет графический элемент."""
        if ann.id not in self._graphics_items:
            return
        
        item = self._graphics_items[ann.id]
        color = QColor(ann.color)
        
        if ann.annotation_type == AnnotationType.BBOX:
            x, y, w, h = ann.coords
            if isinstance(item, QGraphicsRectItem):
                item.setRect(x, y, w, h)
        else:
            points = [QPointF(p[0], p[1]) for p in ann.coords]
            if isinstance(item, QGraphicsPolygonItem):
                item.setPolygon(QPolygonF(points))
        
        # Обновляем стиль
        is_selected = (ann.id in self._selected_ids)
        self._update_item_style(item, ann, is_selected)
        
        # Обновляем метку
        if ann.id in self._label_items:
            label = self._label_items[ann.id]
            label.setPlainText(ann.class_name)
            if ann.annotation_type == AnnotationType.BBOX:
                label.setPos(ann.coords[0] + 2, ann.coords[1] - 22)
            else:
                label.setPos(ann.coords[0][0], ann.coords[0][1] - 22)
    
    def _update_item_style(self, item: QGraphicsItem, ann: Annotation, selected: bool):
        """Обновляет стиль элемента."""
        color = QColor(ann.color)
        
        if selected:
            pen = QPen(QColor("#58a6ff"), 3, Qt.SolidLine)
            fill_color = QColor(ann.color)
            fill_color.setAlpha(60)
        else:
            pen = QPen(color, 2)
            fill_color = QColor(color)
            fill_color.setAlpha(40)
        
        if isinstance(item, (QGraphicsRectItem, QGraphicsPolygonItem)):
            item.setPen(pen)
            item.setBrush(QBrush(fill_color))
    
    def _create_handles(self, ann: Annotation):
        """Создаёт ручки редактирования."""
        self._clear_handles()
        
        if ann.annotation_type == AnnotationType.BBOX:
            x, y, w, h = ann.coords
            positions = [
                ('nw', x, y), ('ne', x + w, y),
                ('sw', x, y + h), ('se', x + w, y + h),
            ]
            for hid, hx, hy in positions:
                handle = EditHandle(hid)
                handle.setPos(hx, hy)
                self._scene.addItem(handle)
                self._handles.append(handle)
        else:
            for i, (px, py) in enumerate(ann.coords):
                handle = EditHandle(str(i))
                handle.setPos(px, py)
                self._scene.addItem(handle)
                self._handles.append(handle)
    
    def _clear_handles(self):
        """Удаляет ручки."""
        for h in self._handles:
            self._scene.removeItem(h)
        self._handles.clear()
    
    def _hide_all_annotations(self):
        """Скрывает все аннотации при рисовании."""
        for aid, item in self._graphics_items.items():
            item.setVisible(False)
        for aid, label in self._label_items.items():
            label.setVisible(False)
    
    def _show_all_annotations(self):
        """Показывает все аннотации после рисования."""
        self._update_visibility()
    
    # === Временное рисование ===
    
    def _clear_temp_drawing(self):
        """Очищает временные элементы рисования."""
        if self._temp_item:
            self._scene.removeItem(self._temp_item)
            self._temp_item = None
        
        if self._polygon_preview_item:
            self._scene.removeItem(self._polygon_preview_item)
            self._polygon_preview_item = None
        
        for dot in self._polygon_dots:
            self._scene.removeItem(dot)
        self._polygon_dots.clear()
        
        self._polygon_points.clear()
        self._draw_start = None
        self._is_drawing = False
        
        # Показываем все аннотации обратно
        self._show_all_annotations()
    
    # === Обработчики событий ===
    
    def _on_wheel(self, event: QWheelEvent):
        """Масштабирование."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self._view.scale(factor, factor)
    
    def _on_mouse_press(self, event: QMouseEvent):
        """Нажатие мыши."""
        # Панорамирование средней кнопкой
        if event.button() == Qt.MiddleButton:
            self._view.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = QMouseEvent(event.type(), event.localPos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            QGraphicsView.mousePressEvent(self._view, fake)
            return
        
        if event.button() != Qt.LeftButton:
            return
        
        scene_pos = self._view.mapToScene(event.pos())
        
        # Проверяем клик по ручке редактирования
        for handle in self._handles:
            if handle.contains(handle.mapFromScene(scene_pos)):
                self._start_handle_drag(handle, scene_pos)
                return
        
        # Если рисуем — не проверяем клики по объектам
        if self._is_drawing:
            if self._mode == DrawingMode.POLYGON:
                self._add_polygon_point(scene_pos)
            return
        
            # Проверяем клик по существующей аннотации
        if self._mode == DrawingMode.SELECT or not self._drawing_enabled:
            clicked_item = self._find_annotation_at(scene_pos)
            
            # Проверяем Ctrl
            is_ctrl = bool(event.modifiers() & Qt.ControlModifier)
            
            if clicked_item:
                self.select_annotation(clicked_item, toggle=is_ctrl)
                # Drag начинаем только если кликнули по уже выбранному или только что выбрали
                if clicked_item in self._selected_ids:
                     self._start_object_drag(scene_pos)
                return
        
        # Снимаем выделение (если не кликнули по объекту)
        # Если нажат Ctrl, то не снимаем выделение при клике в пустоту? 
        # Обычно клик в пустоту снимает выделение даже с Ctrl.
        # Но можно сделать чтобы Ctrl оставлял. Пусть пока снимает.
        self.select_annotation(None)
        
        # Начинаем рисование
        if self._drawing_enabled and self._current_class:
            if self._mode == DrawingMode.BBOX:
                self._start_bbox_drawing(scene_pos)
            elif self._mode == DrawingMode.POLYGON:
                self._start_polygon_drawing(scene_pos)
    
    def _on_mouse_move(self, event: QMouseEvent):
        """Движение мыши."""
        scene_pos = self._view.mapToScene(event.pos())
        
        # Перетаскивание ручки
        if self._drag_handle:
            self._update_handle_drag(scene_pos)
            return
        
        # Перетаскивание объекта
        if self._is_dragging and self._selected_ids:
            self._update_object_drag(scene_pos)
            return
        
        # Рисование bbox
        if self._is_drawing and self._mode == DrawingMode.BBOX:
            self._update_bbox_preview(scene_pos)
        
        # Рисование polygon (превью линии к курсору)
        elif self._polygon_points and self._mode == DrawingMode.POLYGON:
            self._update_polygon_preview(scene_pos)
        
        QGraphicsView.mouseMoveEvent(self._view, event)
    
    def _on_mouse_release(self, event: QMouseEvent):
        """Отпускание мыши."""
        if event.button() == Qt.MiddleButton:
            self._view.setDragMode(QGraphicsView.NoDrag)
            return
        
        if event.button() != Qt.LeftButton:
            return
        
        scene_pos = self._view.mapToScene(event.pos())
        
        # Завершаем перетаскивание ручки
        if self._drag_handle:
            self._finish_handle_drag()
            return
        
        # Завершаем перетаскивание объекта
        if self._is_dragging:
            self._finish_object_drag()
            return
        
        # Завершаем рисование bbox
        if self._is_drawing and self._mode == DrawingMode.BBOX:
            self._finish_bbox_drawing(scene_pos)
        
        QGraphicsView.mouseReleaseEvent(self._view, event)
    
    def _on_double_click(self, event: QMouseEvent):
        """Двойной клик — завершает полигон."""
        if self._mode == DrawingMode.POLYGON and len(self._polygon_points) >= 3:
            self._finish_polygon_drawing()
        QGraphicsView.mouseDoubleClickEvent(self._view, event)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Горячие клавиши."""
        key = event.key()
        modifiers = event.modifiers()
        
        # Ctrl+Z — Undo
        if key == Qt.Key_Z and modifiers == Qt.ControlModifier:
            self.undo()
            return
        
        # Ctrl+Shift+Z или Ctrl+Y — Redo
        if (key == Qt.Key_Z and modifiers == (Qt.ControlModifier | Qt.ShiftModifier)) or \
           (key == Qt.Key_Y and modifiers == Qt.ControlModifier):
            self.redo()
            return
        
        # Delete / Backspace — удалить выбранное
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            return
        
        # Escape — отменить рисование / снять выделение
        if key == Qt.Key_Escape:
            if self._is_drawing or self._polygon_points:
                self._clear_temp_drawing()
            else:
                self.select_annotation(None)
            return
        
        # Enter — завершить полигон
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self._mode == DrawingMode.POLYGON and len(self._polygon_points) >= 3:
                self._finish_polygon_drawing()
            return
        
        super().keyPressEvent(event)
    
    # === Поиск аннотации ===
    
    def _find_annotation_at(self, pos: QPointF) -> Optional[str]:
        """Находит аннотацию под курсором."""
        for aid, item in self._graphics_items.items():
            if isinstance(item, QGraphicsRectItem):
                if item.rect().contains(pos):
                    return aid
            elif isinstance(item, QGraphicsPolygonItem):
                if item.polygon().containsPoint(pos, Qt.OddEvenFill):
                    return aid
        return None
    
    # === Рисование BBOX ===
    
    def _start_bbox_drawing(self, pos: QPointF):
        """Начало рисования bbox."""
        self._is_drawing = True
        self._draw_start = pos
        
        # Скрываем другие аннотации при рисовании
        self._hide_all_annotations()
        
        self._temp_item = QGraphicsRectItem()
        color = QColor(self._current_color)
        self._temp_item.setPen(QPen(color, 2, Qt.DashLine))
        fill = QColor(color)
        fill.setAlpha(30)
        self._temp_item.setBrush(QBrush(fill))
        self._temp_item.setZValue(500)
        self._scene.addItem(self._temp_item)
    
    def _update_bbox_preview(self, pos: QPointF):
        """Обновление превью bbox."""
        if not self._temp_item or not self._draw_start:
            return
        
        x1, y1 = self._draw_start.x(), self._draw_start.y()
        x2, y2 = pos.x(), pos.y()
        
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        self._temp_item.setRect(rect)
    
    def _finish_bbox_drawing(self, pos: QPointF):
        """Завершение рисования bbox."""
        if not self._draw_start:
            return
        
        x1, y1 = self._draw_start.x(), self._draw_start.y()
        x2, y2 = pos.x(), pos.y()
        
        w, h = abs(x2 - x1), abs(y2 - y1)
        
        if w >= MIN_BBOX_SIZE and h >= MIN_BBOX_SIZE:
            ann = Annotation(
                class_name=self._current_class,
                annotation_type=AnnotationType.BBOX,
                coords=[min(x1, x2), min(y1, y2), w, h],
                color=self._current_color
            )
            self.add_annotation(ann)
        
        self._clear_temp_drawing()
    
    # === Рисование POLYGON ===
    
    def _start_polygon_drawing(self, pos: QPointF):
        """Начало рисования полигона."""
        self._polygon_points = [pos]
        self._is_drawing = True
        
        # Скрываем другие аннотации при рисовании
        self._hide_all_annotations()
        
        # Рисуем первую точку
        self._add_polygon_dot(pos)
    
    def _add_polygon_point(self, pos: QPointF):
        """Добавляет точку полигона."""
        # Проверяем замыкание (клик близко к первой точке)
        if len(self._polygon_points) >= 3:
            first = self._polygon_points[0]
            dist = math.sqrt((pos.x() - first.x())**2 + (pos.y() - first.y())**2)
            if dist < POLYGON_CLOSE_DISTANCE:
                self._finish_polygon_drawing()
                return
        
        self._polygon_points.append(pos)
        self._add_polygon_dot(pos)
        self._update_polygon_preview(pos)
    
    def _add_polygon_dot(self, pos: QPointF):
        """Добавляет визуальную точку."""
        dot = QGraphicsEllipseItem(-4, -4, 8, 8)
        dot.setPos(pos)
        dot.setBrush(QBrush(QColor(self._current_color)))
        dot.setPen(QPen(QColor("#ffffff"), 2))
        dot.setZValue(600)
        self._scene.addItem(dot)
        self._polygon_dots.append(dot)
    
    def _update_polygon_preview(self, cursor_pos: QPointF):
        """Обновляет превью полигона."""
        if len(self._polygon_points) < 1:
            return
        
        if self._polygon_preview_item:
            self._scene.removeItem(self._polygon_preview_item)
        
        # Рисуем полигон с текущей позицией курсора
        points = self._polygon_points + [cursor_pos]
        polygon = QPolygonF(points)
        
        self._polygon_preview_item = QGraphicsPolygonItem(polygon)
        color = QColor(self._current_color)
        self._polygon_preview_item.setPen(QPen(color, 2, Qt.DashLine))
        fill = QColor(color)
        fill.setAlpha(30)
        self._polygon_preview_item.setBrush(QBrush(fill))
        self._polygon_preview_item.setZValue(500)
        self._scene.addItem(self._polygon_preview_item)
    
    def _finish_polygon_drawing(self):
        """Завершает рисование полигона."""
        if len(self._polygon_points) < 3:
            self._clear_temp_drawing()
            return
        
        coords = [[p.x(), p.y()] for p in self._polygon_points]
        
        ann = Annotation(
            class_name=self._current_class,
            annotation_type=AnnotationType.POLYGON,
            coords=coords,
            color=self._current_color
        )
        self.add_annotation(ann)
        self._clear_temp_drawing()
    
    # === Перетаскивание объекта ===
    
    def _start_object_drag(self, pos: QPointF):
        """Начало перетаскивания объекта."""
        self._is_dragging = True
        self._drag_start = pos
        self._original_annotations_map.clear()
        
        # Запоминаем состояние всех выбранных объектов
        for aid in self._selected_ids:
            if aid in self._annotations:
                self._original_annotations_map[aid] = copy.deepcopy(self._annotations[aid])
    
    def _update_object_drag(self, pos: QPointF):
        """Обновление перетаскивания."""
        if not self._selected_ids or not self._drag_start or not self._original_annotations_map:
            return
        
        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()
        
        for aid, orig in self._original_annotations_map.items():
            if aid not in self._annotations:
                continue
                
            ann = self._annotations[aid]
            
            if ann.annotation_type == AnnotationType.BBOX:
                ann.coords = [orig.coords[0] + dx, orig.coords[1] + dy, orig.coords[2], orig.coords[3]]
            else:
                ann.coords = [[p[0] + dx, p[1] + dy] for p in orig.coords]
            
            self._update_graphics_item(ann)
            
            # Обновляем ручки только если выбран один (так как ручки только у одного)
            if len(self._selected_ids) == 1:
                self._create_handles(ann)
    
    def _finish_object_drag(self):
        """Завершение перетаскивания."""
        if self._selected_ids and self._original_annotations_map:
            # Проверяем были ли изменения
            changed = False
            for aid, orig in self._original_annotations_map.items():
                if aid in self._annotations:
                    if self._annotations[aid].coords != orig.coords:
                        changed = True
                        break
            
            if changed:
                self._save_undo_state("Перемещение объектов")
                # Эмитим сигналы
                for aid in self._selected_ids:
                    if aid in self._annotations:
                         self.annotation_modified.emit(self._annotations[aid])
                self.annotations_changed.emit()
        
        self._is_dragging = False
        self._drag_start = None
        self._original_annotations_map.clear()
    
    # === Перетаскивание ручки ===
    
    def _start_handle_drag(self, handle: EditHandle, pos: QPointF):
        """Начало перетаскивания ручки."""
        self._drag_handle = handle
        self._drag_start = pos
        # Ручки работают только при выделении одного объекта
        if len(self._selected_ids) == 1:
            aid = next(iter(self._selected_ids))
            if aid in self._annotations:
                self._original_annotations_map = {aid: copy.deepcopy(self._annotations[aid])}
    
    def _update_handle_drag(self, pos: QPointF):
        """Обновление перетаскивания ручки."""
        if len(self._selected_ids) != 1 or not self._drag_handle or not self._original_annotations_map:
            return
            
        aid = next(iter(self._selected_ids))
        if aid not in self._annotations or aid not in self._original_annotations_map:
            return
            
        ann = self._annotations[aid]
        orig = self._original_annotations_map[aid]
        hid = self._drag_handle.handle_id
        
        if ann.annotation_type == AnnotationType.BBOX:
            ox, oy, ow, oh = orig.coords
            
            if hid == 'nw':
                new_x, new_y = pos.x(), pos.y()
                new_w = ox + ow - new_x
                new_h = oy + oh - new_y
                if new_w > MIN_BBOX_SIZE and new_h > MIN_BBOX_SIZE:
                    ann.coords = [new_x, new_y, new_w, new_h]
            elif hid == 'ne':
                new_y = pos.y()
                new_w = pos.x() - ox
                new_h = oy + oh - new_y
                if new_w > MIN_BBOX_SIZE and new_h > MIN_BBOX_SIZE:
                    ann.coords = [ox, new_y, new_w, new_h]
            elif hid == 'sw':
                new_x = pos.x()
                new_w = ox + ow - new_x
                new_h = pos.y() - oy
                if new_w > MIN_BBOX_SIZE and new_h > MIN_BBOX_SIZE:
                    ann.coords = [new_x, oy, new_w, new_h]
            elif hid == 'se':
                new_w = pos.x() - ox
                new_h = pos.y() - oy
                if new_w > MIN_BBOX_SIZE and new_h > MIN_BBOX_SIZE:
                    ann.coords = [ox, oy, new_w, new_h]
        else:
            # Полигон — двигаем конкретную точку
            try:
                idx = int(hid)
                if 0 <= idx < len(ann.coords):
                    ann.coords[idx] = [pos.x(), pos.y()]
            except ValueError:
                pass
        
        self._update_graphics_item(ann)
        self._create_handles(ann)
        self._drag_handle.setPos(pos)
    
    def _finish_handle_drag(self):
        """Завершение перетаскивания ручки."""
        if len(self._selected_ids) == 1 and self._original_annotations_map:
            aid = next(iter(self._selected_ids))
            if aid in self._annotations and aid in self._original_annotations_map:
                current = self._annotations[aid]
                orig = self._original_annotations_map[aid]
                if current.coords != orig.coords:
                    self._save_undo_state("Изменение размера")
                    self.annotation_modified.emit(current)
                    self.annotations_changed.emit()
        
        self._drag_handle = None
        self._drag_start = None
        self._original_annotations_map.clear()
