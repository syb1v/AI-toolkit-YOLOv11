"""Core business logic for AI Toolkit."""
from .hardware_detector import HardwareDetector, DeviceInfo
from .class_manager import ClassManager
from .trainer import YOLOTrainer
from .predictor import YOLOPredictor
from .project_manager import ProjectManager

__all__ = [
    'HardwareDetector', 'DeviceInfo',
    'ClassManager',
    'YOLOTrainer', 'YOLOPredictor',
    'ProjectManager',
]
