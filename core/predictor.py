"""
Модуль инференса YOLO моделей.
"""
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from models.annotation import Annotation, AnnotationType


@dataclass
class PredictionResult:
    """Результат предсказания."""
    annotations: List[Annotation]
    class_counts: Dict[str, int]
    inference_time_ms: float
    image_size: Tuple[int, int]
    model_name: str


class YOLOPredictor:
    """
    Инференс YOLO моделей.
    
    Использование:
        predictor = YOLOPredictor("path/to/best.pt")
        result = predictor.predict("image.jpg", conf=0.25)
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: путь к весам модели (.pt)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_name = self.model_path.name
        self._class_names: Dict[int, str] = {}
        
        self._load_model()
    
    def _load_model(self):
        """Загружает модель."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self._class_names = self.model.names
        except Exception as e:
            raise ValueError(f"Ошибка загрузки модели: {e}")
    
    @property
    def class_names(self) -> Dict[int, str]:
        """Словарь имен классов {id: name}."""
        return self._class_names
    
    @property
    def is_segment_model(self) -> bool:
        """Является ли модель сегментационной."""
        if self.model:
            return hasattr(self.model, 'task') and self.model.task == 'segment'
        return 'seg' in self.model_name.lower()
    
    def predict(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        conf_max: Optional[float] = None,
        device: str = ""
    ) -> PredictionResult:
        """
        Выполняет предсказание на изображении.
        
        Args:
            image_path: путь к изображению
            conf: минимальный порог уверенности
            iou: порог IoU для NMS
            max_det: максимальное количество детекций
            conf_max: максимальный порог уверенности (фильтрация сверху)
            device: устройство для инференса
        
        Returns:
            PredictionResult с аннотациями
        """
        if not self.model:
            raise RuntimeError("Модель не загружена")
        
        import time
        start_time = time.time()
        
        # Запуск предсказания
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device if device else None,
            verbose=False
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        # Обработка результатов
        result = results[0]
        img_h, img_w = result.orig_shape
        
        annotations: List[Annotation] = []
        class_counts: Dict[str, int] = {}
        
        boxes = result.boxes
        masks = result.masks
        
        for i, box in enumerate(boxes):
            confidence = float(box.conf[0])
            
            # Фильтрация по верхнему порогу
            if conf_max is not None and confidence > conf_max:
                continue
            
            class_id = int(box.cls[0])
            class_name = self._class_names.get(class_id, f"class_{class_id}")
            
            # Счетчик классов
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Создаем аннотацию
            if masks is not None and i < len(masks.xy):
                # Сегментация - polygon
                try:
                    polygon = masks.xy[i].tolist()
                    if len(polygon) >= 3:  # Минимум 3 точки для полигона
                        annotation = Annotation(
                            id=str(uuid.uuid4()),
                            class_name=class_name,
                            annotation_type=AnnotationType.POLYGON,
                            coords=polygon,
                            confidence=confidence,
                            color=self.get_class_color(class_name)
                        )
                        annotations.append(annotation)
                        continue
                except:
                    pass
            
            # Детекция - bbox
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            
            annotation = Annotation(
                id=str(uuid.uuid4()),
                class_name=class_name,
                annotation_type=AnnotationType.BBOX,
                coords=[x1, y1, w, h],
                confidence=confidence,
                color=self.get_class_color(class_name)
            )
            annotations.append(annotation)
        
        return PredictionResult(
            annotations=annotations,
            class_counts=class_counts,
            inference_time_ms=inference_time,
            image_size=(img_w, img_h),
            model_name=self.model_name
        )
    
    def predict_batch(
        self,
        image_paths: List[str],
        **kwargs
    ) -> List[PredictionResult]:
        """
        Выполняет предсказание на списке изображений.
        """
        return [self.predict(path, **kwargs) for path in image_paths]
    
    def get_class_color(self, class_name: str) -> str:
        """Возвращает цвет для класса (хеш на основе имени)."""
        import hashlib
        hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:6], 16)
        r = (hash_val >> 16) & 0xFF
        g = (hash_val >> 8) & 0xFF
        b = hash_val & 0xFF
        return f"#{r:02X}{g:02X}{b:02X}"


def load_predictor(model_path: str) -> Optional[YOLOPredictor]:
    """
    Безопасно загружает предиктор.
    Returns: YOLOPredictor или None при ошибке.
    """
    try:
        return YOLOPredictor(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None
