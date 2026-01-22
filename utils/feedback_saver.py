"""
Утилита для сохранения исправленных аннотаций для дообучения.

Сохраняет изображение и аннотации в формате YOLO для последующего дообучения модели.
"""
import os
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class FeedbackSaver:
    """Сохраняет исправленные аннотации для дообучения модели."""
    
    def __init__(self, base_dir: Path = None):
        """
        Args:
            base_dir: Базовая директория для feedback_data.
                      По умолчанию используется CWD.
        """
        self.base_dir = base_dir or Path.cwd()
        self.feedback_dir = self.base_dir / "feedback_data"
    
    def save(self, image_path: str, annotations: List[Any], 
             classes: List[str], image_size: tuple) -> str:
        """
        Сохраняет изображение и аннотации для дообучения.
        
        Args:
            image_path: Путь к исходному изображению
            annotations: Список объектов Annotation
            classes: Список классов модели
            image_size: (width, height) изображения
            
        Returns:
            Путь к сохранённому датасету
        """
        # Создаём структуру папок
        images_dir = self.feedback_dir / "images" / "train"
        labels_dir = self.feedback_dir / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        orig_name = Path(image_path).stem
        new_name = f"{orig_name}_{timestamp}"
        ext = Path(image_path).suffix
        
        # Копируем изображение
        dst_image = images_dir / f"{new_name}{ext}"
        shutil.copy2(image_path, dst_image)
        
        # Конвертируем аннотации в формат YOLO
        yolo_labels = []
        width, height = image_size
        
        for ann in annotations:
            class_name = getattr(ann, 'class_name', 'unknown')
            if class_name not in classes:
                continue
            
            class_id = classes.index(class_name)
            coords = getattr(ann, 'coords', [])
            # Fix: use annotation_type instead of type, and handle enum value
            ann_type_enum = getattr(ann, 'annotation_type', None)
            ann_type = ann_type_enum.value if hasattr(ann_type_enum, 'value') else str(ann_type_enum)
            
            # YOLO format: class x_center y_center width height (normalized)
            if ann_type == 'bbox' and len(coords) >= 4:
                # Annotation stores [x, y, w, h]
                x, y, w_pix, h_pix = coords[:4]
                
                # Center
                x_center = (x + w_pix / 2) / width
                y_center = (y + h_pix / 2) / height
                
                # Normalized width/height
                w_norm = w_pix / width
                h_norm = h_pix / height
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # Polygon format: class x1 y1 x2 y2 ... (normalized)
            elif ann_type == 'polygon':
                # Annotation stores list of lists: [[x, y], [x, y], ...]
                points = []
                # Check format of coords
                if coords and isinstance(coords[0], (list, tuple)):
                    # [[x,y], [x,y]]
                    for pt in coords:
                        px = pt[0] / width
                        py = pt[1] / height
                        points.append(f"{px:.6f} {py:.6f}")
                else:
                    # [x, y, x, y] flat list (fallback)
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            px = coords[i] / width
                            py = coords[i+1] / height
                            points.append(f"{px:.6f} {py:.6f}")
                            
                if points:
                    yolo_labels.append(f"{class_id} {' '.join(points)}")
        
        # Сохраняем лейблы
        dst_label = labels_dir / f"{new_name}.txt"
        with open(dst_label, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        # Создаём/обновляем data.yaml
        self._update_data_yaml(classes)
        
        return str(self.feedback_dir)
    
    def _update_data_yaml(self, classes: List[str]):
        """Создаёт или обновляет data.yaml."""
        yaml_path = self.feedback_dir / "data.yaml"
        
        data = {
            "path": str(self.feedback_dir.absolute()),
            "train": "images/train",
            "val": "images/train",  # Используем те же данные для валидации
            "nc": len(classes),
            "names": classes
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику сохранённых данных."""
        images_dir = self.feedback_dir / "images" / "train"
        if not images_dir.exists():
            return {"images": 0, "labels": 0}
        
        images = list(images_dir.glob("*.[jJpP][pPnN][gG]"))
        labels_dir = self.feedback_dir / "labels" / "train"
        labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        
        return {
            "images": len(images),
            "labels": len(labels)
        }
