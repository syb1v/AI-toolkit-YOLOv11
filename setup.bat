@echo off
chcp 65001 >nul 2>&1
:: ============================================================================
:: AI Toolkit YOLO v2 - Автоматическая установка для Windows
:: Поддержка: NVIDIA CUDA, Intel, CPU
:: ============================================================================

setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  AI Toolkit YOLO v2 - Автоматическая настройка            ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: ============================================================================
:: Определение GPU
:: ============================================================================

set GPU_TYPE=cpu
set GPU_NAME=CPU

:: Проверяем NVIDIA
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader^,nounits 2^>nul') do (
        set GPU_TYPE=nvidia
        set GPU_NAME=%%i
    )
    if "!GPU_TYPE!"=="nvidia" (
        echo [√] Обнаружен GPU: !GPU_NAME!
        goto :gpu_detected
    )
)

:: Проверяем Intel GPU через WMI
for /f "tokens=*" %%i in ('wmic path win32_videocontroller get name 2^>nul ^| findstr /i "Intel"') do (
    set GPU_TYPE=intel
    set GPU_NAME=%%i
    echo [i] Обнаружена Intel графика: !GPU_NAME!
    goto :gpu_detected
)

echo [i] GPU не обнаружен, будет использоваться CPU

:gpu_detected

:: ============================================================================
:: Проверка Python
:: ============================================================================

echo.
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python не найден!
    echo.
    echo Установите Python 3.10+ с https://python.org
    echo При установке отметьте "Add Python to PATH"
    pause
    exit /b 1
)

:: Проверяем версию
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [√] Python найден: %PYVER%

:: ============================================================================
:: Создание виртуального окружения
:: ============================================================================

echo.
if exist .venv (
    echo [!] Виртуальное окружение уже существует.
    set /p RECREATE="Пересоздать? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo [i] Удаление старого окружения...
        rmdir /s /q .venv
    ) else (
        echo [i] Используется существующее окружение
        goto :install_deps
    )
)

echo [i] Создание виртуального окружения...
python -m venv .venv
if %errorlevel% neq 0 (
    echo [X] Ошибка создания venv!
    pause
    exit /b 1
)
echo [√] Venv создан

:install_deps

:: ============================================================================
:: Активация venv
:: ============================================================================

call .venv\Scripts\activate.bat

:: Обновляем pip
echo [i] Обновление pip...
python -m pip install --upgrade pip -q

:: ============================================================================
:: Установка PyTorch
:: ============================================================================

echo.
echo [i] Установка PyTorch для %GPU_TYPE%...

if "%GPU_TYPE%"=="nvidia" (
    echo [i] Устанавливаю PyTorch с поддержкой CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "%GPU_TYPE%"=="intel" (
    echo [i] Устанавливаю PyTorch для CPU/Intel...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo [i] Устанавливаю CPU версию PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

if %errorlevel% neq 0 (
    echo [X] Ошибка установки PyTorch!
    pause
    exit /b 1
)
echo [√] PyTorch установлен

:: ============================================================================
:: Установка остальных зависимостей
:: ============================================================================

echo.
echo [i] Установка остальных зависимостей...
pip install ultralytics PyQt5 Pillow opencv-python-headless numpy pyyaml pandas matplotlib tqdm -q
if %errorlevel% neq 0 (
    echo [X] Ошибка установки зависимостей!
    pause
    exit /b 1
)
echo [√] Зависимости установлены

:: ============================================================================
:: Проверка установки
:: ============================================================================

echo.
echo [i] Проверка установки...

:: Проверяем PyTorch
python -c "import torch; print(f'[√] PyTorch {torch.__version__}')"
if %errorlevel% neq 0 (
    echo [X] PyTorch не работает!
    pause
    exit /b 1
)

:: Проверяем устройство
python -c "import torch; print('[√] CUDA:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

:: Проверяем YOLO
python -c "from ultralytics import YOLO; print('[√] YOLO: OK')"
if %errorlevel% neq 0 (
    echo [X] Ultralytics не работает!
    pause
    exit /b 1
)

:: Проверяем PyQt5
python -c "from PyQt5.QtWidgets import QApplication; print('[√] PyQt5: OK')"
if %errorlevel% neq 0 (
    echo [X] PyQt5 не работает!
    pause
    exit /b 1
)

:: ============================================================================
:: Готово
:: ============================================================================

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  √ Установка завершена успешно!                           ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Для запуска приложения:
echo   .venv\Scripts\activate
echo   python main.py
echo.
echo Или дважды кликните на run.bat
echo.
pause
