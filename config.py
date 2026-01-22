"""
Глобальные константы и настройки приложения.
"""
from pathlib import Path

# === Версия ===
APP_VERSION = "2.0.0"
APP_NAME = "AI Toolkit YOLO"
APP_TITLE = f"{APP_NAME} v{APP_VERSION}"

# === Пути ===
BASE_DIR = Path(__file__).parent.resolve()
ASSETS_DIR = BASE_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
STYLE_FILE = ASSETS_DIR / "style.qss"

# Рабочие директории (относительно проекта пользователя)
MODELS_DIR = "models"
DATASETS_DIR = "datasets"
FEEDBACK_DIR = "feedback_data"
RUNS_DIR = "runs"

# Структура датасета YOLO
IMAGES_DIR = "images"
LABELS_DIR = "labels"
VAL_IMAGES_DIR = "images_val"
VAL_LABELS_DIR = "labels_val"

# === Файлы ===
CONFIG_FILE = "config.yaml"
DATA_YAML_FILE = "data.yaml"
CLASS_SETS_FILE = "class_sets.json"
AUTOSAVE_FILE = "autosave.json"
SETTINGS_FILE = "settings.json"

# YOLO weights
BEST_MODEL_FILE = "best.pt"
LAST_MODEL_FILE = "last.pt"

# === Форматы ===
SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff")
SUPPORTED_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff)"

# === UI Константы ===
MIN_WINDOW_WIDTH = 1280
MIN_WINDOW_HEIGHT = 800
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 900

# Цвета для классов (если не задан пользователем)
DEFAULT_CLASS_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FF4500",
    "#9370DB", "#20B2AA", "#FFD700", "#DC143C", "#00FA9A",
]

# === Настройки аннотаций ===
MIN_BBOX_SIZE = 5  # Минимальный размер bbox в пикселях
HANDLE_SIZE = 8    # Размер ручек изменения размера
POLYGON_CLOSE_DISTANCE = 15  # Дистанция для замыкания полигона

# === Настройки обучения ===
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 640
DEFAULT_LEARNING_RATE = 0.01

# === YOLO модели ===
YOLO_MODELS = {
    "YOLOv11 Nano (Быстрая)": "yolo11n.pt",
    "YOLOv11 Small": "yolo11s.pt",
    "YOLOv11 Medium": "yolo11m.pt",
    "YOLOv11 Large": "yolo11l.pt",
    "YOLOv11 XLarge (Точная)": "yolo11x.pt",
}

YOLO_SEGMENT_MODELS = {
    "YOLOv11 Nano Segment": "yolo11n-seg.pt",
    "YOLOv11 Small Segment": "yolo11s-seg.pt",
    "YOLOv11 Medium Segment": "yolo11m-seg.pt",
    "YOLOv11 Large Segment": "yolo11l-seg.pt",
    "YOLOv11 XLarge Segment": "yolo11x-seg.pt",
}

# === Типы задач ===
TASK_TYPES = {
    "detect": "Детекция (Bounding Box)",
    "segment": "Сегментация (Polygon Mask)",
}

# === Автосохранение ===
AUTOSAVE_INTERVAL_MS = 30000  # 30 секунд
