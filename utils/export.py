"""
Экспорт датасетов в формат YOLO.
"""
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from PIL import Image

from models.annotation import Annotation
from config import (
    IMAGES_DIR, LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR, 
    DATA_YAML_FILE
)


@dataclass
class ExportStats:
    """Статистика экспорта."""
    total_images: int = 0
    train_images: int = 0
    val_images: int = 0
    total_annotations: int = 0
    classes_count: int = 0
    export_path: str = ""


class DatasetExporter:
    """
    Экспортирует размеченные изображения в формат YOLO.
    """
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        annotations: Dict[str, List[Annotation]],
        class_mapping: Dict[str, int],
        val_split: float = 0.2
    ):
        """
        Args:
            source_dir: папка с исходными изображениями
            output_dir: папка для экспорта датасета
            annotations: {filename: [Annotation, ...]}
            class_mapping: {class_name: class_id}
            val_split: доля валидационных данных (0-1)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.annotations = annotations
        self.class_mapping = class_mapping
        self.val_split = val_split
        
        self._stats = ExportStats()
    
    def export(self, shuffle: bool = True) -> ExportStats:
        """
        Выполняет экспорт датасета.
        
        Args:
            shuffle: перемешать файлы перед разделением
        
        Returns:
            статистика экспорта
        """
        # Создаем структуру папок
        self._create_directories()
        
        # Получаем список файлов с аннотациями
        files = list(self.annotations.keys())
        
        if shuffle:
            random.shuffle(files)
        
        # Разделяем на train/val
        split_idx = int(len(files) * (1 - self.val_split))
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        # Экспортируем
        self._export_split(train_files, is_val=False)
        self._export_split(val_files, is_val=True)
        
        # Создаем data.yaml
        self._create_data_yaml()
        
        # Статистика
        self._stats.total_images = len(files)
        self._stats.train_images = len(train_files)
        self._stats.val_images = len(val_files)
        self._stats.classes_count = len(self.class_mapping)
        self._stats.export_path = str(self.output_dir)
        
        return self._stats
    
    def _create_directories(self):
        """Создает структуру папок."""
        dirs = [
            self.output_dir / IMAGES_DIR,
            self.output_dir / LABELS_DIR,
            self.output_dir / VAL_IMAGES_DIR,
            self.output_dir / VAL_LABELS_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _export_split(self, files: List[str], is_val: bool):
        """Экспортирует группу файлов."""
        img_dir = VAL_IMAGES_DIR if is_val else IMAGES_DIR
        lbl_dir = VAL_LABELS_DIR if is_val else LABELS_DIR
        
        for filename in files:
            # Копируем изображение
            src_path = self.source_dir / filename
            dst_path = self.output_dir / img_dir / filename
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            
            # Получаем размеры изображения
            try:
                with Image.open(src_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Ошибка чтения {filename}: {e}")
                continue
            
            # Создаем файл аннотаций
            label_filename = Path(filename).stem + ".txt"
            label_path = self.output_dir / lbl_dir / label_filename
            
            annotations = self.annotations.get(filename, [])
            self._stats.total_annotations += len(annotations)
            
            with open(label_path, "w") as f:
                for ann in annotations:
                    class_id = self.class_mapping.get(ann.class_name)
                    if class_id is None:
                        continue
                    
                    line = ann.to_yolo_format(img_width, img_height, class_id)
                    if line:
                        f.write(line + "\n")
    
    def _create_data_yaml(self):
        """Создает data.yaml для YOLO."""
        # Инвертируем class_mapping для получения {id: name}
        names = {v: k for k, v in self.class_mapping.items()}
        
        data = {
            "path": str(self.output_dir.absolute()),
            "train": IMAGES_DIR,
            "val": VAL_IMAGES_DIR,
            "names": names
        }
        
        yaml_path = self.output_dir / DATA_YAML_FILE
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def count_images_in_dataset(dataset_path: str) -> Tuple[int, int]:
    """
    Подсчитывает количество изображений в датасете YOLO.
    Returns: (train_count, val_count)
    """
    dataset = Path(dataset_path)
    
    train_dir = dataset / IMAGES_DIR
    val_dir = dataset / VAL_IMAGES_DIR
    
    train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*"))) if val_dir.exists() else 0
    
    return train_count, val_count


def load_dataset_info(yaml_path: str) -> Optional[Dict]:
    """
    Загружает информацию о датасете из data.yaml.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except:
        return None
