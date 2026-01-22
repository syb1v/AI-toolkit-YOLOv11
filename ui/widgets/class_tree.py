"""
Виджет дерева классов.
"""
from typing import Optional, List, Callable
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QComboBox, QMenu, QAction, QInputDialog, QMessageBox,
    QColorDialog, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap, QPainter, QBrush

from core.class_manager import ClassManager, ClassInfo


class ClassTreeWidget(QWidget):
    """
    Виджет управления классами.
    
    Сигналы:
        class_selected: str - выбран класс
        class_set_changed: str - изменен набор классов
        classes_modified - изменен список классов
    """
    
    class_selected = pyqtSignal(str, str)  # class_name, color
    class_set_changed = pyqtSignal(str)  # set_name
    classes_modified = pyqtSignal()
    
    def __init__(self, class_manager: ClassManager, parent=None):
        super().__init__(parent)
        self._manager = class_manager
        self._current_set: Optional[str] = None
        self._init_ui()
        self._refresh_sets()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Заголовок
        header = QLabel("Классы объектов")
        header.setStyleSheet("font-weight: bold; font-size: 15px; color: #58a6ff;")
        layout.addWidget(header)
        
        # Выбор набора
        set_layout = QHBoxLayout()
        set_layout.setSpacing(6)
        
        self._combo_sets = QComboBox()
        self._combo_sets.setToolTip("Выберите набор классов для работы")
        self._combo_sets.currentTextChanged.connect(self._on_set_changed)
        set_layout.addWidget(self._combo_sets, 1)
        
        self._btn_add_set = QPushButton("+")
        self._btn_add_set.setProperty("cssClass", "small-icon")
        self._btn_add_set.setFixedWidth(36)
        self._btn_add_set.setToolTip("Создать новый набор классов")
        self._btn_add_set.clicked.connect(self._add_set)
        set_layout.addWidget(self._btn_add_set)
        
        self._btn_del_set = QPushButton("-")
        self._btn_del_set.setProperty("cssClass", "small-icon")
        self._btn_del_set.setFixedWidth(36)
        self._btn_del_set.setToolTip("Удалить текущий набор классов")
        self._btn_del_set.clicked.connect(self._delete_set)
        set_layout.addWidget(self._btn_del_set)
        
        layout.addLayout(set_layout)
        
        # Дерево классов
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.setToolTip("Кликните для выбора класса\nПКМ — меню редактирования")
        layout.addWidget(self._tree)
        
        # Кнопки управления классами
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self._btn_add_class = QPushButton("+ Класс")
        self._btn_add_class.setToolTip("Добавить новый класс")
        self._btn_add_class.clicked.connect(self._add_class)
        btn_layout.addWidget(self._btn_add_class)
        
        self._btn_add_child = QPushButton("Подкласс")
        self._btn_add_child.setToolTip("Добавить подкласс к выбранному")
        self._btn_add_child.clicked.connect(self._add_child_class)
        btn_layout.addWidget(self._btn_add_child)
        
        layout.addLayout(btn_layout)
        
        # Подсказка
        hint = QLabel("ПКМ на классе → цвет, имя, удаление")
        hint.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(hint)
    
    def _create_color_icon(self, color: str) -> QIcon:
        """Создает иконку с цветом."""
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QBrush(QColor(color)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 16, 16, 3, 3)
        painter.end()
        return QIcon(pixmap)
    
    def _refresh_sets(self):
        """Обновляет список наборов."""
        self._combo_sets.blockSignals(True)
        current = self._combo_sets.currentText()
        
        self._combo_sets.clear()
        sets = self._manager.get_set_names()
        self._combo_sets.addItems(sets)
        
        # Восстанавливаем выбор
        if current in sets:
            self._combo_sets.setCurrentText(current)
        elif sets:
            self._combo_sets.setCurrentIndex(0)
        
        self._combo_sets.blockSignals(False)
        self._on_set_changed(self._combo_sets.currentText())
    
    def _refresh_tree(self):
        """Обновляет дерево классов."""
        self._tree.clear()
        
        if not self._current_set:
            return
        
        classes = self._manager.get_classes(self._current_set)
        
        # Строим дерево
        items_map = {}
        
        # Сначала добавляем корневые
        for cls in classes:
            if cls.parent is None:
                item = QTreeWidgetItem([cls.name])
                item.setIcon(0, self._create_color_icon(cls.color))
                item.setData(0, Qt.UserRole, cls.color)
                self._tree.addTopLevelItem(item)
                items_map[cls.name] = item
        
        # Потом добавляем детей
        for cls in classes:
            if cls.parent and cls.parent in items_map:
                parent_item = items_map[cls.parent]
                item = QTreeWidgetItem([cls.name])
                item.setIcon(0, self._create_color_icon(cls.color))
                item.setData(0, Qt.UserRole, cls.color)
                parent_item.addChild(item)
                items_map[cls.name] = item
        
        self._tree.expandAll()
    
    def _on_set_changed(self, set_name: str):
        """Обработчик смены набора."""
        self._current_set = set_name if set_name else None
        self._refresh_tree()
        if set_name:
            self.class_set_changed.emit(set_name)
    
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Обработчик клика по классу."""
        class_name = item.text(0)
        color = item.data(0, Qt.UserRole) or "#FF0000"
        self.class_selected.emit(class_name, color)
    
    def _add_set(self):
        """Добавляет новый набор."""
        name, ok = QInputDialog.getText(
            self, "Новый набор", "Название набора классов:",
        )
        if ok and name:
            if self._manager.create_set(name):
                self._manager.save()
                self._refresh_sets()
                self._combo_sets.setCurrentText(name)
                self.classes_modified.emit()
            else:
                QMessageBox.warning(self, "Ошибка", "Набор с таким именем уже существует")
    
    def _delete_set(self):
        """Удаляет текущий набор."""
        if not self._current_set:
            return
        
        reply = QMessageBox.question(
            self, "Удаление",
            f"Удалить набор '{self._current_set}' и все его классы?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._manager.delete_set(self._current_set)
            self._manager.save()
            self._refresh_sets()
            self.classes_modified.emit()
    
    def _add_class(self):
        """Добавляет класс."""
        if not self._current_set:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите или создайте набор классов")
            return
        
        name, ok = QInputDialog.getText(self, "Новый класс", "Название класса:")
        if ok and name:
            if self._manager.add_class(self._current_set, name):
                self._manager.save()
                self._refresh_tree()
                self.classes_modified.emit()
            else:
                QMessageBox.warning(self, "Ошибка", "Класс с таким именем уже существует")
    
    def _add_child_class(self):
        """Добавляет подкласс к выбранному."""
        if not self._current_set:
            return
        
        current = self._tree.currentItem()
        if not current:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите родительский класс")
            return
        
        parent_name = current.text(0)
        name, ok = QInputDialog.getText(self, "Подкласс", f"Название подкласса для '{parent_name}':")
        
        if ok and name:
            if self._manager.add_class(self._current_set, name, parent_name):
                self._manager.save()
                self._refresh_tree()
                self.classes_modified.emit()
            else:
                QMessageBox.warning(self, "Ошибка", "Класс с таким именем уже существует")
    
    def _show_context_menu(self, pos):
        """Показывает контекстное меню."""
        item = self._tree.itemAt(pos)
        if not item:
            return
        
        class_name = item.text(0)
        
        menu = QMenu(self)
        
        # Изменить цвет
        color_action = QAction("Изменить цвет", self)
        color_action.triggered.connect(lambda: self._change_color(class_name))
        menu.addAction(color_action)
        
        # Переименовать
        rename_action = QAction("Переименовать", self)
        rename_action.triggered.connect(lambda: self._rename_class(class_name))
        menu.addAction(rename_action)
        
        menu.addSeparator()
        
        # Удалить
        delete_action = QAction("Удалить", self)
        delete_action.triggered.connect(lambda: self._delete_class(class_name))
        menu.addAction(delete_action)
        
        menu.exec_(self._tree.mapToGlobal(pos))
    
    def _change_color(self, class_name: str):
        """Изменяет цвет класса."""
        current_color = self._manager.get_class_color(self._current_set, class_name)
        color = QColorDialog.getColor(QColor(current_color), self, "Выберите цвет")
        
        if color.isValid():
            self._manager.set_class_color(self._current_set, class_name, color.name())
            self._manager.save()
            self._refresh_tree()
            self.classes_modified.emit()
    
    def _rename_class(self, class_name: str):
        """Переименовывает класс."""
        new_name, ok = QInputDialog.getText(
            self, "Переименование", 
            f"Новое имя для '{class_name}':",
            text=class_name
        )
        
        if ok and new_name and new_name != class_name:
            if self._manager.rename_class(self._current_set, class_name, new_name):
                self._manager.save()
                self._refresh_tree()
                self.classes_modified.emit()
            else:
                QMessageBox.warning(self, "Ошибка", "Класс с таким именем уже существует")
    
    def _delete_class(self, class_name: str):
        """Удаляет класс."""
        reply = QMessageBox.question(
            self, "Удаление",
            f"Удалить класс '{class_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._manager.delete_class(self._current_set, class_name)
            self._manager.save()
            self._refresh_tree()
            self.classes_modified.emit()
    
    def get_current_set(self) -> Optional[str]:
        """Возвращает имя текущего набора."""
        return self._current_set
    
    def get_class_names(self) -> List[str]:
        """Возвращает список имен классов текущего набора."""
        if self._current_set:
            return self._manager.get_class_names(self._current_set)
        return []
    
    def get_class_mapping(self):
        """Возвращает словарь {class_name: class_id}."""
        if self._current_set:
            return self._manager.get_class_id_mapping(self._current_set)
        return {}
    
    def get_colors(self):
        """Возвращает словарь {class_name: color}."""
        if self._current_set:
            return self._manager.get_all_colors(self._current_set)
        return {}
    
    def refresh(self):
        """Обновляет виджет."""
        self._refresh_sets()

    def selectedItems(self) -> List[QTreeWidgetItem]:
        """Возвращает список выбранных элементов."""
        return self._tree.selectedItems()
