"""
Пресеты настроек обучения для разных сценариев.
Включает уровни аугментации: отсутствует, лёгкая, средняя, сильная.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class AugmentationLevel(Enum):
    """Уровень аугментации данных."""
    NONE = "none"           # Без аугментации
    LIGHT = "light"         # Лёгкая
    MEDIUM = "medium"       # Средняя
    HEAVY = "heavy"         # Сильная (для нестабильных данных)


# Настройки аугментации по уровням
AUGMENTATION_PRESETS: Dict[AugmentationLevel, Dict[str, float]] = {
    AugmentationLevel.NONE: {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    AugmentationLevel.LIGHT: {
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    AugmentationLevel.MEDIUM: {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0001,
        "flipud": 0.1,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.0,
    },
    AugmentationLevel.HEAVY: {
        "hsv_h": 0.02,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
        "degrees": 20.0,
        "translate": 0.2,
        "scale": 0.7,
        "shear": 5.0,
        "perspective": 0.0005,
        "flipud": 0.2,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.2,
        "erasing": 0.4,
    },
}

# Описания уровней аугментации
AUGMENTATION_DESCRIPTIONS = {
    AugmentationLevel.NONE: "Без аугментации — для идеально чистых данных",
    AugmentationLevel.LIGHT: "Лёгкая — минимальные изменения, базовые преобразования",
    AugmentationLevel.MEDIUM: "Средняя — сбалансированная аугментация для большинства задач",
    AugmentationLevel.HEAVY: "Сильная — максимальная вариативность для нестабильных данных",
}


@dataclass
class TrainingPreset:
    """Пресет настроек обучения."""
    name: str
    description: str
    icon: str  # Эмоджи
    
    # Основные параметры
    epochs: int
    batch_size: int
    image_size: int
    
    # Оптимизатор
    learning_rate: float
    optimizer: str
    weight_decay: float
    momentum: float
    
    # Early stopping
    patience: int
    
    # Уровень аугментации
    augmentation_level: AugmentationLevel
    
    # Для каких размеров датасета
    min_images: int = 0
    max_images: Optional[int] = None
    
    # Рекомендуемое устройство
    recommended_device: str = "auto"
    
    def get_augmentation(self) -> Dict[str, float]:
        """Возвращает настройки аугментации."""
        return AUGMENTATION_PRESETS.get(
            self.augmentation_level, 
            AUGMENTATION_PRESETS[AugmentationLevel.MEDIUM]
        ).copy()
    
    def to_training_args(self) -> Dict[str, Any]:
        """Конвертирует пресет в аргументы для YOLO trainer."""
        args = {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "lr0": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "patience": self.patience,
        }
        # Добавляем аугментацию
        args.update(self.get_augmentation())
        return args


# === Определение пресетов ===

PRESETS: Dict[str, TrainingPreset] = {
    "quick_test": TrainingPreset(
        name="Быстрый тест",
        icon="",
        description="Для проверки работоспособности.\n"
                    "10 эпох, минимальные настройки.\n"
                    "Используйте для первого запуска.",
        epochs=10,
        batch_size=16,
        image_size=640,
        learning_rate=0.01,
        optimizer="auto",
        weight_decay=0.0005,
        momentum=0.937,
        patience=5,
        augmentation_level=AugmentationLevel.LIGHT,
        min_images=10,
        max_images=100,
    ),
    
    "small_dataset": TrainingPreset(
        name="Малый датасет (50-200)",
        icon="",
        description="Для небольших датасетов.\n"
                    "Сильная аугментация против переобучения.\n"
                    "Медленное обучение, много эпох.",
        epochs=150,
        batch_size=8,
        image_size=640,
        learning_rate=0.001,
        optimizer="AdamW",
        weight_decay=0.001,
        momentum=0.9,
        patience=30,
        augmentation_level=AugmentationLevel.HEAVY,
        min_images=50,
        max_images=200,
    ),
    
    "medium_dataset": TrainingPreset(
        name="Средний датасет (200-1000)",
        icon="",
        description="Сбалансированные настройки.\n"
                    "Хороший баланс скорости и качества.\n"
                    "Подходит для большинства задач.",
        epochs=100,
        batch_size=16,
        image_size=640,
        learning_rate=0.01,
        optimizer="SGD",
        weight_decay=0.0005,
        momentum=0.937,
        patience=50,
        augmentation_level=AugmentationLevel.MEDIUM,
        min_images=200,
        max_images=1000,
    ),
    
    "large_dataset": TrainingPreset(
        name="Большой датасет (1000+)",
        icon="",
        description="Для больших датасетов.\n"
                    "Меньше аугментации, больше эпох.\n"
                    "Требуется GPU с 8GB+ VRAM.",
        epochs=200,
        batch_size=32,
        image_size=640,
        learning_rate=0.01,
        optimizer="SGD",
        weight_decay=0.0005,
        momentum=0.937,
        patience=50,
        augmentation_level=AugmentationLevel.LIGHT,
        min_images=1000,
        recommended_device="cuda",
    ),
    
    "high_quality": TrainingPreset(
        name="Максимальное качество",
        icon="",
        description="Максимально возможное качество.\n"
                    "Долгое обучение, большие изображения.\n"
                    "! Требуется мощный GPU (12GB+ VRAM)!",
        epochs=300,
        batch_size=8,
        image_size=1280,
        learning_rate=0.01,
        optimizer="SGD",
        weight_decay=0.0005,
        momentum=0.937,
        patience=100,
        augmentation_level=AugmentationLevel.MEDIUM,
        recommended_device="cuda",
    ),
    
    "unstable_data": TrainingPreset(
        name="Нестабильные данные",
        icon="",
        description="Для разнородных/зашумлённых данных.\n"
                    "Максимальная аугментация.\n"
                    "Помогает при плохом качестве фото.",
        epochs=200,
        batch_size=8,
        image_size=640,
        learning_rate=0.001,
        optimizer="AdamW",
        weight_decay=0.01,
        momentum=0.9,
        patience=50,
        augmentation_level=AugmentationLevel.HEAVY,
    ),
    
    "cpu_friendly": TrainingPreset(
        name="Для CPU / Слабого GPU",
        icon="",
        description="Для работы на CPU или слабом GPU.\n"
                    "Маленький batch, меньший размер.\n"
                    "Подходит для ноутбуков.",
        epochs=50,
        batch_size=4,
        image_size=416,
        learning_rate=0.001,
        optimizer="AdamW",
        weight_decay=0.0005,
        momentum=0.9,
        patience=20,
        augmentation_level=AugmentationLevel.LIGHT,
        recommended_device="cpu",
    ),
    
    "custom": TrainingPreset(
        name="Свои настройки",
        icon="",
        description="Полностью настраиваемые параметры.\n"
                    "Для опытных пользователей.",
        epochs=100,
        batch_size=16,
        image_size=640,
        learning_rate=0.01,
        optimizer="auto",
        weight_decay=0.0005,
        momentum=0.937,
        patience=50,
        augmentation_level=AugmentationLevel.MEDIUM,
    ),
}


def get_recommended_preset(num_images: int) -> str:
    """Возвращает ID рекомендованного пресета."""
    if num_images < 50:
        return "quick_test"
    elif num_images < 200:
        return "small_dataset"
    elif num_images < 1000:
        return "medium_dataset"
    else:
        return "large_dataset"


def get_augmentation_level_name(level: AugmentationLevel) -> str:
    """Возвращает название уровня аугментации."""
    names = {
        AugmentationLevel.NONE: "Без аугментации",
        AugmentationLevel.LIGHT: "Лёгкая",
        AugmentationLevel.MEDIUM: "Средняя",
        AugmentationLevel.HEAVY: "Сильная",
    }
    return names.get(level, str(level))
