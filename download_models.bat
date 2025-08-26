@echo off
setlocal

REM Быстрое скачивание чекпойнтов моделей CAS
REM 1) Переходим в корень проекта
cd /d "%~dp0"

REM 2) Проверяем Python
where python >nul 2>nul
if errorlevel 1 (
  echo [ОШИБКА] Python не найден. Установите Python 3.9+ и добавьте в PATH.
  pause
  exit /b 1
)

REM 3) Создаём виртуальное окружение, если его нет
if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Создаю виртуальное окружение .venv ...
  python -m venv .venv
)

REM 4) Запуск скрипта скачивания
echo [RUN] Запуск скачивания моделей...
.venv\Scripts\python.exe download_models.py

echo.
echo [EXIT] Скачивание завершено.
pause
