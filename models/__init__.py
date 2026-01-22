"""Data models for AI Toolkit."""
from .annotation import Annotation, AnnotationType
from .presets import (
    TrainingPreset, PRESETS, 
    AugmentationLevel, AUGMENTATION_PRESETS,
    get_augmentation_level_name
)

__all__ = [
    'Annotation', 'AnnotationType', 
    'TrainingPreset', 'PRESETS',
    'AugmentationLevel', 'AUGMENTATION_PRESETS',
    'get_augmentation_level_name'
]
