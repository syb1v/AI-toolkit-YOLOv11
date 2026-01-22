"""
Модуль обучения YOLO моделей.
Поддерживает колбэки для отслеживания прогресса.
"""
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class TrainingStatus(Enum):
    """Статус обучения."""
    IDLE = "idle"
    STARTING = "starting"
    TRAINING = "training"
    PAUSED = "paused"
    STOPPED = "stopped"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class TrainingProgress:
    """Прогресс обучения."""
    status: TrainingStatus = TrainingStatus.IDLE
    current_epoch: int = 0
    total_epochs: int = 0
    
    # Метрики текущей эпохи
    train_loss: float = 0.0
    val_loss: float = 0.0
    
    # Тренды Loss (для отображения ▼▲)
    prev_train_loss: float = 0.0
    prev_val_loss: float = 0.0
    train_loss_trend: str = ""  # "up", "down", "stable"
    val_loss_trend: str = ""
    
    # Метрики качества
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    # Время
    eta_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    start_time: float = 0.0  # timestamp начала обучения
    
    # Путь к результатам
    save_dir: Optional[str] = None
    best_weights: Optional[str] = None
    
    # Ошибка
    error_message: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        if self.total_epochs == 0:
            return 0.0
        return (self.current_epoch / self.total_epochs) * 100
    
    def get_model_state(self) -> tuple:
        """
        Анализирует состояние модели на основе трендов loss.
        Возвращает (текст, цвет: 'success'/'warning'/'danger'/'info').
        """
        if self.current_epoch < 2:
            return ("Оценка...", "info")
        
        t_trend = self.train_loss_trend
        v_trend = self.val_loss_trend
        
        # Переобучение: train падает, val растёт
        if t_trend == "down" and v_trend == "up":
            if self.val_loss > self.train_loss * 1.5:
                return ("Переобучение", "warning")
            return ("Возможно переобучение", "warning")
        
        # Проблема: оба растут
        if t_trend == "up" and v_trend == "up":
            return ("Проблема (Loss растёт)", "danger")
        
        # Нормальное обучение: оба падают
        if t_trend == "down" and v_trend == "down":
            if self.val_loss <= self.train_loss * 1.5:
                return ("Нормальное (Обучается)", "success")
        
        # Недообучение: высокий train loss
        if self.train_loss > 0.5:
            return ("Недообучение", "info")
        
        # Схождение: val стабилен, train падает
        if v_trend == "stable" and t_trend == "down":
            return ("Схождение (Плато)", "success")
        
        return ("Нормальное (Обучается)", "info")


class YOLOTrainer:
    """
    Обучение YOLO моделей с поддержкой колбэков.
    
    Использование:
        trainer = YOLOTrainer(config)
        trainer.on_progress = lambda p: print(p)
        trainer.on_log = lambda msg: print(msg)
        trainer.start()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: словарь с настройками:
                - model_name: str - имя базовой модели (yolo11n.pt)
                - data_yaml: str - путь к data.yaml
                - save_dir: str - куда сохранять результаты
                - training_args: dict - аргументы для model.train()
        """
        self.config = config
        self.model_name = config.get("model_name", "yolo11n.pt")
        self.data_yaml = config.get("data_yaml")
        self.save_dir = config.get("save_dir", "runs")
        self.training_args = config.get("training_args", {})
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._progress = TrainingProgress()
        
        # Колбэки
        self.on_progress: Optional[Callable[[TrainingProgress], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_finished: Optional[Callable[[TrainingProgress], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    def _log(self, message: str):
        """Отправляет сообщение в лог."""
        if self.on_log:
            self.on_log(message)
    
    def _update_progress(self, **kwargs):
        """Обновляет прогресс и вызывает колбэк."""
        for key, value in kwargs.items():
            if hasattr(self._progress, key):
                setattr(self._progress, key, value)
        
        if self.on_progress:
            self.on_progress(self._progress)
    
    def _on_train_epoch_end(self, trainer):
        """Колбэк YOLO - конец эпохи обучения."""
        if self._stop_event.is_set():
            trainer.stop = True
            self._log("⏹️ Получен сигнал остановки, завершаем...")
            return
        
        epoch = trainer.epoch + 1
        
        # Извлекаем метрики
        metrics = {}
        if hasattr(trainer, 'metrics'):
            m = trainer.metrics
            if hasattr(m, 'box'):
                metrics['map50'] = getattr(m.box, 'map50', 0.0)
                metrics['map50_95'] = getattr(m.box, 'map', 0.0)
        
        # Извлекаем loss
        train_loss = 0.0
        val_loss = 0.0
        if hasattr(trainer, 'loss'):
            train_loss = float(trainer.loss.mean()) if trainer.loss is not None else 0.0
        
        # Пытаемся получить val_loss из validator
        if hasattr(trainer, 'validator') and trainer.validator:
            v = trainer.validator
            if hasattr(v, 'loss') and v.loss is not None:
                val_loss = float(v.loss.mean())
            elif hasattr(trainer, 'metrics') and hasattr(trainer.metrics, 'fitness'):
                # Используем fitness как приближение
                val_loss = 1.0 - getattr(trainer.metrics, 'fitness', 0.5)
        
        # Рассчитываем тренды
        prev_train = self._progress.train_loss
        prev_val = self._progress.val_loss
        
        def get_trend(new_val, old_val):
            if old_val == 0.0:
                return "stable"
            diff = new_val - old_val
            threshold = abs(old_val) * 0.02  # 2% изменение
            if diff < -threshold:
                return "down"
            elif diff > threshold:
                return "up"
            return "stable"
        
        train_trend = get_trend(train_loss, prev_train) if epoch > 1 else "stable"
        val_trend = get_trend(val_loss, prev_val) if epoch > 1 and prev_val > 0 else "stable"
        
        # Рассчитываем время
        elapsed = time.time() - self._progress.start_time if self._progress.start_time else 0
        eta = 0.0
        if epoch > 0 and self._progress.total_epochs > epoch:
            time_per_epoch = elapsed / epoch
            remaining_epochs = self._progress.total_epochs - epoch
            eta = remaining_epochs * time_per_epoch
        
        # Обновляем прогресс
        self._update_progress(
            current_epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss if val_loss > 0 else self._progress.val_loss,
            prev_train_loss=prev_train,
            prev_val_loss=prev_val,
            train_loss_trend=train_trend,
            val_loss_trend=val_trend,
            map50=metrics.get('map50', 0.0),
            map50_95=metrics.get('map50_95', 0.0),
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )
        
        # Формируем красивый лог с индикаторами
        train_arrow = "▼" if train_trend == "down" else ("▲" if train_trend == "up" else "━")
        val_arrow = "▼" if val_trend == "down" else ("▲" if val_trend == "up" else "━")
        
        eta_str = self._format_time(eta) if eta > 0 else "--:--"
        
        self._log(f"📊 Epoch {epoch}/{self._progress.total_epochs} | "
                  f"Train: {train_loss:.4f}{train_arrow} | "
                  f"mAP50: {metrics.get('map50', 0):.4f} | "
                  f"ETA: {eta_str}")
    
    def _format_time(self, seconds: float) -> str:
        """Форматирует секунды в читаемый вид."""
        if seconds < 60:
            return f"{int(seconds)} сек"
        elif seconds < 3600:
            mins = int(seconds // 60)
            return f"~{mins} мин"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"~{hours}ч {mins}м"
    
    def _train_worker(self):
        """Рабочий поток обучения."""
        try:
            from ultralytics import YOLO
            
            self._update_progress(status=TrainingStatus.STARTING)
            self._log(f"🚀 Инициализация модели: {self.model_name}")
            
            # Загружаем модель
            model = YOLO(self.model_name)
            
            # Регистрируем колбэки
            model.add_callback("on_train_epoch_end", self._on_train_epoch_end)
            
            # Подготавливаем аргументы
            save_dir = Path(self.save_dir)
            project = str(save_dir.parent)
            name = save_dir.name
            
            train_args = {
                "data": self.data_yaml,
                "project": project,
                "name": name,
                "exist_ok": True,
                "plots": True,
                "verbose": True,
                **self.training_args
            }
            
            self._update_progress(
                status=TrainingStatus.TRAINING,
                total_epochs=train_args.get("epochs", 100),
                save_dir=str(save_dir),
                start_time=time.time()  # Запоминаем время начала
            )
            
            self._log(f"📁 Данные: {self.data_yaml}")
            self._log(f"📂 Сохранение: {save_dir}")
            self._log(f"⚙️ Эпохи: {train_args.get('epochs')}, "
                      f"Batch: {train_args.get('batch', 16)}, "
                      f"ImgSz: {train_args.get('imgsz', 640)}")
            self._log("=" * 50)
            self._log("▶️ Запуск обучения...")
            
            # Запуск обучения
            results = model.train(**train_args)
            
            # Успешное завершение
            best_weights = str(results.save_dir / "weights" / "best.pt")
            
            self._update_progress(
                status=TrainingStatus.FINISHED,
                best_weights=best_weights,
                current_epoch=self._progress.total_epochs
            )
            
            self._log("=" * 50)
            self._log(f"✅ Обучение завершено!")
            self._log(f"📦 Лучшие веса: {best_weights}")
            
            if self.on_finished:
                self.on_finished(self._progress)
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            self._log(f"❌ ОШИБКА: {error_msg}")
            self._log(traceback.format_exc())
            
            self._update_progress(
                status=TrainingStatus.ERROR,
                error_message=error_msg
            )
            
            if self.on_error:
                self.on_error(error_msg)
    
    def start(self):
        """Запускает обучение в отдельном потоке."""
        if self._thread and self._thread.is_alive():
            self._log("Обучение уже запущено")
            return
        
        if not self.data_yaml:
            raise ValueError("data_yaml не указан")
        
        self._stop_event.clear()
        self._progress = TrainingProgress()
        
        self._thread = threading.Thread(target=self._train_worker, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Останавливает обучение."""
        if self._thread and self._thread.is_alive():
            self._log("🛑 Отправляю сигнал остановки...")
            self._stop_event.set()
            self._update_progress(status=TrainingStatus.STOPPED)
    
    def is_running(self) -> bool:
        """Проверяет, идет ли обучение."""
        return self._thread is not None and self._thread.is_alive()
    
    @property
    def progress(self) -> TrainingProgress:
        """Текущий прогресс обучения."""
        return self._progress
