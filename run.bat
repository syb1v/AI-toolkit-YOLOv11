@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

if not exist .venv (
    echo Виртуальное окружение не найдено!
    echo Запустите setup.bat для установки.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python main.py %*
