"""
Менеджер классов для аннотаций.
Управляет наборами классов, их иерархией и цветами.
"""
import json
import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from config import CLASS_SETS_FILE, DEFAULT_CLASS_COLORS


@dataclass
class ClassInfo:
    """Информация о классе."""
    name: str
    color: str = "#FF0000"
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": self.color,
            "parent": self.parent,
            "children": self.children.copy()
        }
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ClassInfo":
        return cls(
            name=name,
            color=data.get("color", "#FF0000"),
            parent=data.get("parent"),
            children=data.get("children", [])
        )


class ClassManager:
    """
    Менеджер наборов классов.
    
    Структура хранения:
    {
        "SetName1": {
            "class1": {"color": "#FF0000", "parent": null, "children": []},
            "class2": {"color": "#00FF00", "parent": null, "children": ["child1"]},
            "child1": {"color": "#0000FF", "parent": "class2", "children": []}
        },
        "SetName2": {...}
    }
    """
    
    def __init__(self, filepath: str = CLASS_SETS_FILE):
        self.filepath = Path(filepath)
        self._data: Dict[str, Dict[str, Dict]] = {}
        self._is_dirty = False
        self._used_colors: Set[str] = set()
    
    def load(self) -> bool:
        """
        Загружает данные из файла.
        Returns: True если успешно, False если файл не существует.
        Raises: ValueError при ошибке парсинга.
        """
        if not self.filepath.exists():
            self._data = {}
            return False
        
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            self._is_dirty = False
            self._update_used_colors()
            return True
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка чтения JSON: {e}")
        except OSError as e:
            raise OSError(f"Ошибка доступа к файлу: {e}")
    
    def save(self) -> None:
        """Сохраняет данные в файл."""
        try:
            # Создаем директорию если нужно
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            self._is_dirty = False
        except OSError as e:
            raise OSError(f"Ошибка сохранения: {e}")
    
    def _update_used_colors(self):
        """Обновляет множество используемых цветов."""
        self._used_colors = set()
        for class_set in self._data.values():
            for class_data in class_set.values():
                if color := class_data.get("color"):
                    self._used_colors.add(color.upper())
    
    def _generate_unique_color(self) -> str:
        """Генерирует уникальный цвет для нового класса."""
        # Сначала пробуем использовать цвета из палитры
        for color in DEFAULT_CLASS_COLORS:
            if color.upper() not in self._used_colors:
                self._used_colors.add(color.upper())
                return color
        
        # Если все использованы, генерируем новый
        # Используем golden ratio для хорошего распределения
        golden_ratio = 0.618033988749895
        h = (len(self._used_colors) * golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
        color = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
        self._used_colors.add(color.upper())
        return color
    
    @property
    def is_dirty(self) -> bool:
        """Есть ли несохраненные изменения."""
        return self._is_dirty
    
    # === Операции с наборами ===
    
    def get_set_names(self) -> List[str]:
        """Возвращает список имен наборов классов."""
        return sorted(self._data.keys())
    
    def create_set(self, name: str) -> bool:
        """Создает новый набор классов."""
        if name in self._data:
            return False
        self._data[name] = {}
        self._is_dirty = True
        return True
    
    def delete_set(self, name: str) -> bool:
        """Удаляет набор классов."""
        if name not in self._data:
            return False
        del self._data[name]
        self._is_dirty = True
        self._update_used_colors()
        return True
    
    def rename_set(self, old_name: str, new_name: str) -> bool:
        """Переименовывает набор."""
        if old_name not in self._data or new_name in self._data:
            return False
        self._data[new_name] = self._data.pop(old_name)
        self._is_dirty = True
        return True
    
    def get_set(self, name: str) -> Optional[Dict[str, Dict]]:
        """Возвращает набор классов по имени."""
        return self._data.get(name)
    
    # === Операции с классами ===
    
    def get_classes(self, set_name: str) -> List[ClassInfo]:
        """Возвращает список классов в наборе."""
        if set_name not in self._data:
            return []
        
        return [
            ClassInfo.from_dict(name, data)
            for name, data in self._data[set_name].items()
        ]
    
    def get_root_classes(self, set_name: str) -> List[ClassInfo]:
        """Возвращает только корневые классы (без родителя)."""
        return [c for c in self.get_classes(set_name) if c.parent is None]
    
    def get_class_names(self, set_name: str) -> List[str]:
        """Возвращает список имен классов."""
        if set_name not in self._data:
            return []
        return list(self._data[set_name].keys())
    
    def add_class(
        self, 
        set_name: str, 
        class_name: str, 
        parent_name: Optional[str] = None,
        color: Optional[str] = None
    ) -> bool:
        """Добавляет новый класс в набор."""
        if set_name not in self._data:
            return False
        
        # Проверяем уникальность имени
        if class_name in self._data[set_name]:
            return False
        
        # Проверяем существование родителя
        if parent_name and parent_name not in self._data[set_name]:
            return False
        
        # Генерируем цвет если не задан
        if not color:
            color = self._generate_unique_color()
        
        # Создаем класс
        self._data[set_name][class_name] = {
            "color": color,
            "parent": parent_name,
            "children": []
        }
        
        # Добавляем в children родителя
        if parent_name:
            self._data[set_name][parent_name]["children"].append(class_name)
        
        self._is_dirty = True
        return True
    
    def delete_class(self, set_name: str, class_name: str) -> bool:
        """Удаляет класс и всех его потомков."""
        if set_name not in self._data or class_name not in self._data[set_name]:
            return False
        
        class_data = self._data[set_name][class_name]
        
        # Рекурсивно удаляем потомков
        for child in class_data.get("children", []):
            self.delete_class(set_name, child)
        
        # Удаляем из children родителя
        if parent := class_data.get("parent"):
            if parent in self._data[set_name]:
                self._data[set_name][parent]["children"].remove(class_name)
        
        # Удаляем сам класс
        del self._data[set_name][class_name]
        self._is_dirty = True
        self._update_used_colors()
        return True
    
    def rename_class(self, set_name: str, old_name: str, new_name: str) -> bool:
        """Переименовывает класс."""
        if set_name not in self._data:
            return False
        if old_name not in self._data[set_name]:
            return False
        if new_name in self._data[set_name]:
            return False
        
        class_data = self._data[set_name].pop(old_name)
        self._data[set_name][new_name] = class_data
        
        # Обновляем ссылки в родителе
        if parent := class_data.get("parent"):
            if parent in self._data[set_name]:
                children = self._data[set_name][parent]["children"]
                idx = children.index(old_name)
                children[idx] = new_name
        
        # Обновляем ссылки в потомках
        for child in class_data.get("children", []):
            if child in self._data[set_name]:
                self._data[set_name][child]["parent"] = new_name
        
        self._is_dirty = True
        return True
    
    def set_class_color(self, set_name: str, class_name: str, color: str) -> bool:
        """Устанавливает цвет класса."""
        if set_name not in self._data or class_name not in self._data[set_name]:
            return False
        
        self._data[set_name][class_name]["color"] = color
        self._is_dirty = True
        self._update_used_colors()
        return True
    
    def get_class_color(self, set_name: str, class_name: str) -> str:
        """Возвращает цвет класса."""
        if set_name in self._data and class_name in self._data[set_name]:
            return self._data[set_name][class_name].get("color", "#FF0000")
        return "#FF0000"
    
    def get_all_colors(self, set_name: str = None) -> Dict[str, str]:
        """Возвращает словарь {class_name: color}."""
        colors = {}
        
        if set_name:
            sets_to_scan = [(set_name, self._data.get(set_name, {}))]
        else:
            sets_to_scan = list(self._data.items())
        
        for _, class_set in sets_to_scan:
            for name, data in class_set.items():
                colors[name] = data.get("color", "#FF0000")
        
        return colors
    
    def get_class_id_mapping(self, set_name: str) -> Dict[str, int]:
        """Возвращает словарь {class_name: class_id} для YOLO."""
        names = sorted(self.get_class_names(set_name))
        return {name: i for i, name in enumerate(names)}
    
    def get_id_class_mapping(self, set_name: str) -> Dict[int, str]:
        """Возвращает словарь {class_id: class_name} для YOLO."""
        names = sorted(self.get_class_names(set_name))
        return {i: name for i, name in enumerate(names)}
    
    # === Импорт/Экспорт ===
    
    def import_from_yolo_yaml(self, yaml_path: str, set_name: str = None) -> str:
        """
        Импортирует классы из YOLO data.yaml.
        Returns: имя созданного набора.
        """
        import yaml
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        names = data.get("names", {})
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        
        if not set_name:
            set_name = Path(yaml_path).stem
        
        # Создаем набор с уникальным именем
        base_name = set_name
        counter = 1
        while set_name in self._data:
            set_name = f"{base_name}_{counter}"
            counter += 1
        
        self.create_set(set_name)
        
        for class_id in sorted(names.keys()):
            class_name = names[class_id]
            self.add_class(set_name, class_name)
        
        return set_name
    
    def export_to_yolo_names(self, set_name: str) -> Dict[int, str]:
        """Экспортирует классы в формат names для YOLO."""
        return self.get_id_class_mapping(set_name)
