"""
Модель данных для аннотаций.
Унифицированный формат для bbox и polygon.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple  # Добавили Tuple, убрали _typeshed
import uuid


class AnnotationType(Enum):
    """Тип аннотации."""
    BBOX = "bbox"      # Bounding box [x, y, width, height]
    POLYGON = "polygon"  # Polygon [[x1,y1], [x2,y2], ...]


@dataclass
class Annotation:
    """
    Унифицированная модель аннотации.

    Для BBOX: coords = [x, y, width, height] (левый верхний угол + размеры)
    Для POLYGON: coords = [[x1, y1], [x2, y2], ...] (список точек)

    Все координаты хранятся в ПИКСЕЛЯХ изображения (абсолютные).
    Нормализация происходит только при экспорте в YOLO формат.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    class_name: str = ""
    annotation_type: AnnotationType = AnnotationType.BBOX
    coords: List = field(default_factory=list)
    color: str = "#FF0000"
    confidence: Optional[float] = None  # Для результатов инференса

    def to_yolo_format(self, img_width: int, img_height: int, class_id: int) -> str:
        """
        Конвертирует аннотацию в строку YOLO формата.

        Для BBOX: <class_id> <cx> <cy> <w> <h>
        Для POLYGON: <class_id> <x1> <y1> <x2> <y2> ...

        Все координаты нормализованы [0, 1].
        """
        if self.annotation_type == AnnotationType.BBOX:
            x, y, w, h = self.coords
            cx = (x + w / 2) / img_width
            cy = (y + h / 2) / img_height
            nw = w / img_width
            nh = h / img_height
            return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

        elif self.annotation_type == AnnotationType.POLYGON:
            points_str = " ".join(
                f"{x / img_width:.6f} {y / img_height:.6f}"
                for x, y in self.coords
            )
            return f"{class_id} {points_str}"

        return ""

    @classmethod
    def from_yolo_format(
        cls,
        line: str,
        img_width: int,
        img_height: int,
        class_names: dict,
        is_segment: bool = False
    ) -> Optional["Annotation"]:
        """
        Создает аннотацию из строки YOLO формата.
        """
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        class_id = int(parts[0])
        class_name = class_names.get(class_id, f"class_{class_id}")
        values = [float(v) for v in parts[1:]]

        if is_segment or len(values) > 4:
            # Polygon format: x1 y1 x2 y2 ...
            coords = []
            for i in range(0, len(values), 2):
                if i + 1 < len(values):
                    x = values[i] * img_width
                    y = values[i + 1] * img_height
                    coords.append([x, y])
            return cls(
                class_name=class_name,
                annotation_type=AnnotationType.POLYGON,
                coords=coords
            )
        else:
            # BBox format: cx cy w h
            cx, cy, nw, nh = values[:4]
            w = nw * img_width
            h = nh * img_height
            x = cx * img_width - w / 2
            y = cy * img_height - h / 2
            return cls(
                class_name=class_name,
                annotation_type=AnnotationType.BBOX,
                coords=[x, y, w, h]
            )

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Возвращает bounding box для любого типа аннотации.
        Returns: (x, y, width, height)
        """
        if self.annotation_type == AnnotationType.BBOX:
            return tuple(self.coords)

        elif self.annotation_type == AnnotationType.POLYGON and self.coords:
            xs = [p[0] for p in self.coords]
            ys = [p[1] for p in self.coords]
            x = min(xs)
            y = min(ys)
            w = max(xs) - x
            h = max(ys) - y
            return x, y, w, h

        return 0.0, 0.0, 0.0, 0.0

    def move(self, dx: float, dy: float):
        """Перемещает аннотацию на (dx, dy)."""
        if self.annotation_type == AnnotationType.BBOX:
            self.coords[0] += dx
            self.coords[1] += dy
        elif self.annotation_type == AnnotationType.POLYGON:
            self.coords = [[p[0] + dx, p[1] + dy] for p in self.coords]

    def copy(self) -> "Annotation":
        """Создает копию аннотации с новым ID."""
        return Annotation(
            id=str(uuid.uuid4()),
            class_name=self.class_name,
            annotation_type=self.annotation_type,
            coords=self.coords.copy() if isinstance(self.coords, list) else self.coords,
            color=self.color,
            confidence=self.confidence
        )