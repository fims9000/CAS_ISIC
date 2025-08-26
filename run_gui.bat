@echo off
setlocal

REM Быстрый запуск CAS GUI
REM 1) Переходим в корень проекта
cd /d "%~dp0"

REM 2) Создаём виртуальное окружение, если его нет
if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Создаю виртуальное окружение .venv ...
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ОШИБКА] Python не найден. Установите Python 3.9+ и добавьте в PATH.
    pause
    exit /b 1
  )
  python -m venv .venv
)

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  set "PY=python"
)

REM 3) Обновляем pip и ставим зависимости
"%PY%" -m pip --disable-pip-version-check install --upgrade pip >nul
if exist requirements.txt (
  echo [SETUP] Устанавливаю зависимости (requirements.txt)...
  "%PY%" -m pip --disable-pip-version-check install -r requirements.txt
)

REM 4) Запуск GUI
echo [RUN] Запуск приложения...
"%PY%" -X faulthandler -m GUI.main

echo.
echo [EXIT] Работа приложения завершена.
pause


