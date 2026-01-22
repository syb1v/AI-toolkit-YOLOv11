"""
Менеджер проектов для хранения настроек моделей.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

from config import CONFIG_FILE, RUNS_DIR, MODELS_DIR


@dataclass
class ProjectConfig:
    """Конфигурация проекта обучения."""
    
    # Метаданные
    __dataclass_fields__ = None
    name: str = "New Project"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Модель
    model_name: str = "yolo11n.pt"
    task_type: str = "detect"  # detect или segment
    
    # Данные
    data_yaml_path: Optional[str] = None
    
    # Обучение
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    device: str = "auto"
    
    # Оптимизатор
    learning_rate: float = 0.01
    optimizer: str = "auto"
    weight_decay: float = 0.0005
    momentum: float = 0.937
    
    # Early stopping
    patience: int = 50
    
    # Аугментация
    augmentation: Dict[str, float] = field(default_factory=lambda: {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    })
    
    # Пресет
    preset_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь для сохранения."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        """Создает из словаря."""
        # Фильтруем только известные поля
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def get_training_args(self) -> Dict[str, Any]:
        """Возвращает аргументы для YOLO trainer."""
        args = {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.image_size,
            "device": self.device if self.device != "auto" else "",
            "lr0": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "patience": self.patience,
        }
        
        # Добавляем аугментацию
        args.update(self.augmentation)
        
        return args


class ProjectManager:
    """
    Менеджер проектов обучения.
    Каждый проект - папка с config.yaml и runs/.
    """
    
    def __init__(self, base_dir: str = MODELS_DIR):
        self.base_dir = Path(base_dir)
        self.current_project_path: Optional[Path] = None
        self.current_config: Optional[ProjectConfig] = None
    
    def ensure_base_dir(self):
        """Создает базовую директорию если нужно."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def list_projects(self) -> List[str]:
        """Возвращает список имен существующих проектов."""
        self.ensure_base_dir()
        projects = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / CONFIG_FILE).exists():
                projects.append(item.name)
        return sorted(projects)
    
    def create_project(self, name: str) -> Path:
        """
        Создает новый проект.
        Returns: путь к папке проекта.
        """
        self.ensure_base_dir()
        
        # Генерируем уникальное имя
        project_path = self.base_dir / name
        counter = 1
        while project_path.exists():
            project_path = self.base_dir / f"{name}_{counter}"
            counter += 1
        
        # Создаем структуру
        project_path.mkdir(parents=True)
        (project_path / RUNS_DIR).mkdir()
        
        # Создаем конфиг
        config = ProjectConfig(name=project_path.name)
        self._save_config(project_path, config)
        
        return project_path
    
    def load_project(self, path: str) -> ProjectConfig:
        """
        Загружает проект.
        Returns: конфигурация проекта.
        """
        project_path = Path(path)
        config_path = project_path / CONFIG_FILE
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        self.current_project_path = project_path
        self.current_config = ProjectConfig.from_dict(data)
        
        return self.current_config
    
    def save_project(self, config: Optional[ProjectConfig] = None) -> None:
        """Сохраняет текущий проект."""
        if config:
            self.current_config = config
        
        if not self.current_project_path or not self.current_config:
            raise ValueError("No project loaded")
        
        self.current_config.updated_at = datetime.now().isoformat()
        self._save_config(self.current_project_path, self.current_config)
    
    @staticmethod
    def _save_config(project_path: Path, config: ProjectConfig) -> None:
        """Сохраняет конфиг в файл."""
        config_path = project_path / CONFIG_FILE
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config.to_dict(), f, sort_keys=False, allow_unicode=True)
    
    def delete_project(self, name: str) -> bool:
        """Удаляет проект."""
        import shutil
        project_path = self.base_dir / name
        if project_path.exists():
            shutil.rmtree(project_path)
            if self.current_project_path == project_path:
                self.current_project_path = None
                self.current_config = None
            return True
        return False
    
    def get_runs_dir(self) -> Optional[Path]:
        """Возвращает путь к директории runs текущего проекта."""
        if self.current_project_path:
            runs_dir = self.current_project_path / RUNS_DIR
            runs_dir.mkdir(exist_ok=True)
            return runs_dir
        return None
    
    def get_best_weights_path(self) -> Optional[Path]:
        """Возвращает путь к лучшим весам если есть."""
        runs_dir = self.get_runs_dir()
        if runs_dir:
            # Ищем последнюю папку train* или detect* и т.д.
            train_dirs = sorted(runs_dir.glob("**/weights/best.pt"))
            if train_dirs:
                return train_dirs[-1]
        return None
    
    def get_last_weights_path(self) -> Optional[Path]:
        """Возвращает путь к последним весам."""
        runs_dir = self.get_runs_dir()
        if runs_dir:
            last_dirs = sorted(runs_dir.glob("**/weights/last.pt"))
            if last_dirs:
                return last_dirs[-1]
        return None
    
    def apply_preset(self, preset_id: str) -> None:
        """Применяет пресет к текущему проекту."""
        from models.presets import PRESETS
        
        if not self.current_config:
            raise ValueError("No project loaded")
        
        if preset_id not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_id}")
        
        preset = PRESETS[preset_id]
        
        self.current_config.preset_id = preset_id
        self.current_config.epochs = preset.epochs
        self.current_config.batch_size = preset.batch_size
        self.current_config.image_size = preset.image_size
        self.current_config.learning_rate = preset.learning_rate
        self.current_config.optimizer = preset.optimizer
        self.current_config.weight_decay = preset.weight_decay
        self.current_config.momentum = preset.momentum
        self.current_config.patience = preset.patience
        self.current_config.augmentation = preset.get_augmentation()
        
        self.save_project()
