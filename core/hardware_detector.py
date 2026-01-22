"""
Детектор оборудования для определения доступных GPU/CPU.
Поддержка: NVIDIA CUDA, AMD ROCm, Intel oneAPI, Apple MPS, CPU.
"""
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import os


class DeviceType(Enum):
    """Тип вычислительного устройства."""
    CPU = "cpu"
    CUDA = "cuda"           # NVIDIA GPU
    ROCM = "rocm"           # AMD GPU (Linux)
    MPS = "mps"             # Apple Silicon
    ONEAPI = "xpu"          # Intel GPU
    DIRECTML = "directml"   # Windows DirectML (AMD/Intel)


@dataclass
class DeviceInfo:
    """Информация об устройстве."""
    device_type: DeviceType
    device_id: str              # "cpu", "cuda:0", "mps", etc.
    name: str                   # Человекочитаемое имя
    memory_total: Optional[int] = None  # MB
    memory_free: Optional[int] = None   # MB
    compute_capability: Optional[str] = None  # Для CUDA
    is_available: bool = True
    
    def get_display_name(self) -> str:
        """Возвращает имя для отображения в UI."""
        if self.memory_total:
            mem_gb = self.memory_total / 1024
            return f"{self.name} ({mem_gb:.1f} GB)"
        return self.name
    
    def get_pytorch_device(self) -> str:
        """Возвращает строку устройства для PyTorch."""
        return self.device_id


class HardwareDetector:
    """
    Определяет доступное оборудование для обучения ML моделей.
    """
    
    def __init__(self):
        self._devices: List[DeviceInfo] = []
        self._torch_available = False
        self._cuda_available = False
        self._mps_available = False
        self._xpu_available = False
        self._rocm_available = False
        self._detect()
    
    def _detect(self):
        """Определяет все доступные устройства."""
        self._devices = []
        
        # Проверяем наличие PyTorch
        try:
            import torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            # Добавляем только CPU без PyTorch
            self._devices.append(DeviceInfo(
                device_type=DeviceType.CPU,
                device_id="cpu",
                name=f"CPU ({platform.processor() or 'Unknown'})",
                is_available=False  # PyTorch не установлен
            ))
            return
        
        # CPU всегда доступен
        cpu_name = self._get_cpu_name()
        self._devices.append(DeviceInfo(
            device_type=DeviceType.CPU,
            device_id="cpu",
            name=f"CPU: {cpu_name}",
            is_available=True
        ))
        
        # NVIDIA CUDA
        self._detect_cuda(torch)
        
        # Apple MPS (Apple Silicon)
        self._detect_mps(torch)
        
        # Intel oneAPI (XPU)
        self._detect_xpu(torch)
        
        # AMD ROCm (Linux)
        self._detect_rocm(torch)

    @staticmethod
    def _get_cpu_name() -> str:
        """Получает имя процессора."""
        system = platform.system()

        if system == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            except:
                pass
        elif system == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                )
                return winreg.QueryValueEx(key, "ProcessorNameString")[0]
            except:
                pass
        elif system == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass

        return platform.processor() or "Unknown CPU"

    def _detect_cuda(self, torch):
        """Определяет NVIDIA GPU через CUDA."""
        try:
            if torch.cuda.is_available():
                self._cuda_available = True
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    
                    # Получаем память
                    try:
                        mem_total = props.total_memory // (1024 * 1024)
                        # Для free памяти нужно переключиться на устройство
                        torch.cuda.set_device(i)
                        mem_free = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                    except:
                        mem_total = None
                        mem_free = None
                    
                    self._devices.append(DeviceInfo(
                        device_type=DeviceType.CUDA,
                        device_id=f"cuda:{i}" if i > 0 else "cuda",
                        name=f"NVIDIA {props.name}",
                        memory_total=mem_total,
                        memory_free=mem_free,
                        compute_capability=f"{props.major}.{props.minor}",
                        is_available=True
                    ))
        except Exception as e:
            print(f"CUDA detection error: {e}")
    
    def _detect_mps(self, torch):
        """Определяет Apple Silicon MPS."""
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._mps_available = True
                
                # Получаем имя чипа Apple
                chip_name = "Apple Silicon"
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        chip_name = result.stdout.strip()
                except:
                    pass
                
                self._devices.append(DeviceInfo(
                    device_type=DeviceType.MPS,
                    device_id="mps",
                    name=f"Apple {chip_name} (Metal GPU)",
                    is_available=True
                ))
        except Exception as e:
            print(f"MPS detection error: {e}")
    
    def _detect_xpu(self, torch):
        """Определяет Intel GPU через oneAPI."""
        try:
            # Проверяем наличие intel_extension_for_pytorch
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                self._xpu_available = True
                for i in range(torch.xpu.device_count()):
                    try:
                        name = torch.xpu.get_device_name(i)
                    except:
                        name = f"Intel GPU {i}"
                    
                    self._devices.append(DeviceInfo(
                        device_type=DeviceType.ONEAPI,
                        device_id=f"xpu:{i}" if i > 0 else "xpu",
                        name=f"Intel {name}",
                        is_available=True
                    ))
        except ImportError:
            pass
        except Exception as e:
            print(f"XPU detection error: {e}")
    
    def _detect_rocm(self, torch):
        """Определяет AMD GPU через ROCm."""
        # ROCm использует тот же API что и CUDA в PyTorch
        # Проверяем наличие ROCm через переменные окружения или hip
        if platform.system() != "Linux":
            return
        
        try:
            # Проверяем, установлен ли ROCm
            rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
            if os.path.exists(rocm_path):
                # Если CUDA тоже доступна, проверяем что это именно ROCm
                hip_version = subprocess.run(
                    ["hipconfig", "--version"],
                    capture_output=True, text=True
                )
                if hip_version.returncode == 0 and not self._cuda_available:
                    self._rocm_available = True
                    # ROCm устройства доступны через torch.cuda
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            try:
                                props = torch.cuda.get_device_properties(i)
                                name = props.name
                            except:
                                name = f"AMD GPU {i}"
                            
                            self._devices.append(DeviceInfo(
                                device_type=DeviceType.ROCM,
                                device_id=f"cuda:{i}" if i > 0 else "cuda",
                                name=f"AMD {name} (ROCm)",
                                is_available=True
                            ))
        except Exception as e:
            print(f"ROCm detection error: {e}")
    
    @property
    def devices(self) -> List[DeviceInfo]:
        """Список всех обнаруженных устройств."""
        return self._devices
    
    @property
    def available_devices(self) -> List[DeviceInfo]:
        """Список доступных для использования устройств."""
        return [d for d in self._devices if d.is_available]
    
    @property
    def has_gpu(self) -> bool:
        """Есть ли доступный GPU."""
        return any(
            d.device_type != DeviceType.CPU and d.is_available 
            for d in self._devices
        )
    
    @property
    def best_device(self) -> DeviceInfo:
        """Возвращает лучшее доступное устройство."""
        # Приоритет: CUDA > MPS > XPU > ROCm > CPU
        priority = {
            DeviceType.CUDA: 0,
            DeviceType.MPS: 1,
            DeviceType.ONEAPI: 2,
            DeviceType.ROCM: 3,
            DeviceType.CPU: 10
        }
        
        available = self.available_devices
        if not available:
            # Возвращаем CPU по умолчанию
            return DeviceInfo(
                device_type=DeviceType.CPU,
                device_id="cpu",
                name="CPU (PyTorch not installed)",
                is_available=False
            )
        
        return min(available, key=lambda d: priority.get(d.device_type, 99))
    
    def get_device_by_id(self, device_id: str) -> Optional[DeviceInfo]:
        """Получает устройство по ID."""
        for device in self._devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_status_report(self) -> str:
        """Возвращает текстовый отчет о доступных устройствах."""
        lines = ["=== Обнаруженные устройства ===\n"]
        
        if not self._torch_available:
            lines.append("! PyTorch не установлен!\n")
            lines.append("Установите PyTorch для вашей платформы:\n")
            lines.append("  pip install torch torchvision\n")
            return "\n".join(lines)
        
        for device in self._devices:
            status = "✅" if device.is_available else "❌"
            lines.append(f"{status} {device.get_display_name()}")
            if device.device_type == DeviceType.CUDA and device.compute_capability:
                lines.append(f"   Compute Capability: {device.compute_capability}")
        
        lines.append(f"\n🎯 Рекомендуемое устройство: {self.best_device.name}")
        
        return "\n".join(lines)
    
    def get_pytorch_install_command(self) -> str:
        """Возвращает команду установки PyTorch для текущей системы."""
        system = platform.system()
        
        if self._cuda_available:
            return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        elif self._mps_available or system == "Darwin":
            return "pip install torch torchvision"
        elif self._rocm_available:
            return "pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0"
        elif system == "Linux" and self._check_intel_gpu():
            return "pip install torch torchvision intel-extension-for-pytorch"
        else:
            return "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    
    @staticmethod
    def _check_intel_gpu() -> bool:
        """Проверяет наличие Intel GPU."""
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True
            )
            return "Intel" in result.stdout and ("VGA" in result.stdout or "Display" in result.stdout)
        except:
            return False


# Singleton для использования во всем приложении
_hardware_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """Возвращает singleton экземпляр HardwareDetector."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector
