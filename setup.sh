#!/bin/bash
# ============================================================================
# AI Toolkit YOLO v2 - Автоматическая установка
# Поддержка: NVIDIA CUDA, AMD ROCm, Intel oneAPI, Apple MPS, CPU
# Совместимость: Linux (Ubuntu/Debian/Arch/Fedora), macOS
# ============================================================================

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Переход в директорию скрипта
cd "$(dirname "$0")"

# ============================================================================
# Функции вывода
# ============================================================================

print_header() {
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BLUE}AI Toolkit YOLO v2 - Автоматическая настройка${NC}           ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ============================================================================
# Определение системы
# ============================================================================

detect_os() {
    OS="unknown"
    DISTRO="unknown"
    
    case "$(uname -s)" in
        Linux*)
            OS="linux"
            if [ -f /etc/os-release ]; then
                DISTRO=$(grep "^ID=" /etc/os-release | cut -d= -f2 | tr -d '"')
            fi
            ;;
        Darwin*)
            OS="macos"
            DISTRO="macos"
            ;;
        MINGW*|CYGWIN*|MSYS*)
            OS="windows"
            DISTRO="windows"
            ;;
        *)
            print_error "Неподдерживаемая операционная система: $(uname -s)"
            exit 1
            ;;
    esac
    
    print_info "Операционная система: ${OS} (${DISTRO})"
}

# ============================================================================
# Определение GPU
# ============================================================================

detect_gpu() {
    GPU_TYPE="cpu"
    GPU_NAME="CPU"
    
    # Проверяем NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_TYPE="nvidia"
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "NVIDIA GPU")
            print_step "Обнаружен GPU: ${GPU_NAME}"
            return
        fi
    fi
    
    # Проверяем AMD (ROCm) на Linux
    if [ "$OS" = "linux" ]; then
        if command -v rocminfo &> /dev/null; then
            if rocminfo 2>/dev/null | grep -q "Name:"; then
                GPU_TYPE="amd"
                GPU_NAME=$(rocminfo 2>/dev/null | grep "Marketing Name" | head -n1 | cut -d: -f2 | xargs || echo "AMD GPU")
                print_step "Обнаружен AMD GPU (ROCm): ${GPU_NAME}"
                return
            fi
        fi
        
        # Альтернативная проверка AMD через lspci
        if command -v lspci &> /dev/null; then
            if lspci 2>/dev/null | grep -i "VGA\|3D" | grep -qi "AMD\|Radeon"; then
                GPU_TYPE="amd"
                GPU_NAME=$(lspci 2>/dev/null | grep -i "VGA\|3D" | grep -i "AMD\|Radeon" | head -n1 | sed 's/.*: //')
                print_step "Обнаружен AMD GPU: ${GPU_NAME}"
                print_warn "ROCm не установлен. Для GPU-ускорения установите ROCm."
                return
            fi
        fi
        
        # Проверяем Intel GPU
        if command -v lspci &> /dev/null; then
            if lspci 2>/dev/null | grep -i "VGA\|3D" | grep -qi "Intel"; then
                GPU_TYPE="intel"
                GPU_NAME=$(lspci 2>/dev/null | grep -i "VGA\|3D" | grep -i "Intel" | head -n1 | sed 's/.*: //')
                print_step "Обнаружена Intel графика: ${GPU_NAME}"
                return
            fi
        fi
    fi
    
    # Apple Silicon
    if [ "$OS" = "macos" ]; then
        CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        if [[ "$CHIP" == *"Apple"* ]]; then
            GPU_TYPE="mps"
            GPU_NAME="$CHIP (Metal GPU)"
            print_step "Обнаружен Apple Silicon: ${GPU_NAME}"
            return
        fi
    fi
    
    print_info "GPU не обнаружен, будет использоваться CPU"
}

# ============================================================================
# Проверка Python
# ============================================================================

check_python() {
    PYTHON_CMD=""
    
    # Пробуем разные варианты
    for cmd in python3 python; do
        if command -v $cmd &> /dev/null; then
            VERSION=$($cmd --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
            MAJOR=$(echo $VERSION | cut -d. -f1)
            MINOR=$(echo $VERSION | cut -d. -f2)
            
            if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
                PYTHON_CMD=$cmd
                print_step "Python найден: $cmd (версия $VERSION)"
                return
            fi
        fi
    done
    
    if [ -z "$PYTHON_CMD" ]; then
        print_error "Python 3.10+ не найден!"
        echo ""
        echo "Установите Python 3.10 или выше:"
        case "$DISTRO" in
            ubuntu|debian)
                echo "  sudo apt update && sudo apt install python3 python3-venv python3-pip"
                ;;
            arch|manjaro)
                echo "  sudo pacman -S python python-pip"
                ;;
            fedora)
                echo "  sudo dnf install python3 python3-pip"
                ;;
            macos)
                echo "  brew install python@3.12"
                ;;
            *)
                echo "  Используйте пакетный менеджер вашей системы"
                ;;
        esac
        exit 1
    fi
}

# ============================================================================
# Установка системных зависимостей
# ============================================================================

install_system_deps() {
    if [ "$OS" = "linux" ]; then
        print_info "Проверка системных зависимостей для Qt..."
        
        case "$DISTRO" in
            ubuntu|debian|linuxmint|pop)
                # Проверяем наличие PyQt5
                if ! dpkg -l python3-pyqt5 &> /dev/null; then
                    print_warn "Рекомендуется установить системный PyQt5:"
                    echo "  sudo apt install python3-pyqt5 libxcb-xinerama0 libxkbcommon-x11-0"
                else
                    print_step "Системный PyQt5 установлен"
                fi
                ;;
            arch|manjaro)
                if ! pacman -Q python-pyqt5 &> /dev/null 2>&1; then
                    print_warn "Рекомендуется установить системный PyQt5:"
                    echo "  sudo pacman -S python-pyqt5"
                else
                    print_step "Системный PyQt5 установлен"
                fi
                ;;
            fedora)
                if ! rpm -q python3-qt5 &> /dev/null 2>&1; then
                    print_warn "Рекомендуется установить системный PyQt5:"
                    echo "  sudo dnf install python3-qt5"
                else
                    print_step "Системный PyQt5 установлен"
                fi
                ;;
        esac
    fi
}

# ============================================================================
# Создание виртуального окружения
# ============================================================================

create_venv() {
    VENV_DIR=".venv"
    
    if [ -d "$VENV_DIR" ]; then
        echo ""
        print_warn "Виртуальное окружение уже существует."
        read -p "Пересоздать? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            print_info "Используется существующее окружение"
            return
        fi
    fi
    
    print_info "Создание виртуального окружения..."
    
    # На Linux используем --system-site-packages для доступа к системному PyQt5
    if [ "$OS" = "linux" ]; then
        $PYTHON_CMD -m venv "$VENV_DIR" --system-site-packages
        print_step "Venv создан с доступом к системным пакетам"
    else
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_step "Venv создан"
    fi
}

# ============================================================================
# Установка PyTorch
# ============================================================================

install_pytorch() {
    print_info "Установка PyTorch для ${GPU_TYPE}..."
    
    # Активируем venv
    source .venv/bin/activate
    
    # Обновляем pip
    pip install --upgrade pip -q
    
    case "$GPU_TYPE" in
        nvidia)
            print_info "Устанавливаю PyTorch с поддержкой CUDA 12.1..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            ;;
        amd)
            if [ "$OS" = "linux" ]; then
                print_info "Устанавливаю PyTorch с поддержкой ROCm 6.0..."
                pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
            else
                print_warn "ROCm поддерживается только на Linux. Устанавливаю CPU версию."
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            fi
            ;;
        intel)
            print_info "Устанавливаю PyTorch с поддержкой Intel oneAPI..."
            pip install torch torchvision
            pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ || {
                print_warn "Intel Extension не установлен (требуется драйвер oneAPI)"
                print_info "Будет использоваться CPU/встроенная графика"
            }
            ;;
        mps)
            print_info "Устанавливаю PyTorch с поддержкой Metal (MPS)..."
            pip install torch torchvision
            ;;
        *)
            print_info "Устанавливаю CPU версию PyTorch..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
    
    print_step "PyTorch установлен"
}

# ============================================================================
# Установка остальных зависимостей
# ============================================================================

install_requirements() {
    print_info "Установка остальных зависимостей..."
    
    # Устанавливаем requirements.txt, пропуская torch (уже установлен)
    pip install ultralytics PyQt5 Pillow opencv-python-headless numpy pyyaml pandas matplotlib tqdm -q
    
    print_step "Зависимости установлены"
}

# ============================================================================
# Проверка установки
# ============================================================================

verify_installation() {
    print_info "Проверка установки..."
    
    # Проверяем PyTorch
    PYTORCH_CHECK=$(python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1) || {
        print_error "PyTorch не работает!"
        return 1
    }
    print_step "$PYTORCH_CHECK"
    
    # Проверяем CUDA/MPS/XPU
    DEVICE_CHECK=$(python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS: Apple Metal GPU')
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f'XPU: Intel GPU')
else:
    print('Device: CPU')
" 2>&1) || DEVICE_CHECK="Device: CPU"
    print_step "$DEVICE_CHECK"
    
    # Проверяем ultralytics
    YOLO_CHECK=$(python -c "from ultralytics import YOLO; print('YOLO: OK')" 2>&1) || {
        print_error "Ultralytics не работает!"
        return 1
    }
    print_step "$YOLO_CHECK"
    
    # Проверяем PyQt5
    QT_CHECK=$(python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5: OK')" 2>&1) || {
        print_error "PyQt5 не работает!"
        return 1
    }
    print_step "$QT_CHECK"
    
    return 0
}

# ============================================================================
# Финальные инструкции
# ============================================================================

print_success() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  ${GREEN}✓ Установка завершена успешно!${NC}                          ${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Для запуска приложения:"
    echo -e "  ${CYAN}source .venv/bin/activate${NC}"
    echo -e "  ${CYAN}python main.py${NC}"
    echo ""
    echo -e "Или используйте скрипт:"
    echo -e "  ${CYAN}./run.sh${NC}"
    echo ""
}

# ============================================================================
# Главная функция
# ============================================================================

main() {
    print_header
    
    # Определяем систему и оборудование
    detect_os
    detect_gpu
    check_python
    
    echo ""
    
    # Проверяем системные зависимости
    install_system_deps
    
    echo ""
    
    # Создаём venv
    create_venv
    
    # Устанавливаем PyTorch
    install_pytorch
    
    # Устанавливаем остальное
    install_requirements
    
    echo ""
    
    # Проверяем установку
    if verify_installation; then
        print_success
    else
        echo ""
        print_error "Установка завершена с ошибками."
        print_info "Попробуйте переустановить: rm -rf .venv && ./setup.sh"
        exit 1
    fi
}

# Запуск
main "$@"
