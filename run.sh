#!/bin/bash
# ============================================================================
# AI Toolkit YOLO v2 - Запуск приложения
# ============================================================================

cd "$(dirname "$0")"

# Проверяем наличие venv
if [ ! -d ".venv" ]; then
    echo "Виртуальное окружение не найдено!"
    echo "Запустите ./setup.sh для установки."
    exit 1
fi

# Активируем venv
source .venv/bin/activate

# Устанавливаем переменные окружения для Qt
export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

# Запускаем приложение
python main.py "$@"
