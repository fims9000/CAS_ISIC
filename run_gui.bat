@echo off
setlocal enabledelayedexpansion

rem Перейти в директорию текущего скрипта
cd /d "%~dp0"

title CAS GUI Launcher
chcp 65001 >nul
set PYTHONUTF8=1

rem Создать виртуальное окружение, если отсутствует
if not exist ".venv" (
  echo [i] Создаю виртуальное окружение .venv ...
  where py >nul 2>&1
  if %errorlevel%==0 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
)

rem Активировать окружение
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

rem Проверить pip
python -m pip --version >nul 2>&1
if not %errorlevel%==0 (
  echo [!] Не удалось найти pip в активированном окружении.
  exit /b 1
)

echo [i] Обновляю pip ...
python -m pip install --upgrade pip

echo [i] Устанавливаю обязательные зависимости для GUI/Core/Models/XAI ...
python -m pip install --prefer-binary numpy Pillow opencv-python scikit-image matplotlib scikit-learn
python -m pip install --prefer-binary torch torchvision

rem Гарантированная установка PyQt5 (если ещё не стоит)
python -c "import PyQt5" >nul 2>&1
if not %errorlevel%==0 (
  echo [i] Устанавливаю PyQt5 ...
  python -m pip install PyQt5==5.15.9
)

echo [i] Устанавливаю XAI-зависимости (необязательные) ...
python -m pip install --prefer-binary captum shap lime || echo [!] Не удалось установить некоторые XAI-библиотеки, продолжу...
python -m pip install --prefer-binary "pytorch-grad-cam>=1.4.8" ^
  || python -m pip install "git+https://github.com/jacobgil/pytorch-grad-cam@master" ^
  || echo [!] pytorch-grad-cam недоступен для вашей версии Python. XAI Grad-CAM будет отключён.

rem Скачать модели при необходимости
if exist "download_models.bat" (
  echo [i] Проверяю/скачиваю модели ...
  call "download_models.bat"
) else if exist "download_models.py" (
  echo [i] Проверяю/скачиваю модели ...
  python "download_models.py"
)

echo [i] Запускаю GUI ...
python -u "GUI\main.py" %*
set "EC=%errorlevel%"
if not "%EC%"=="0" (
  echo [x] Приложение завершилось с ошибкой %EC%.
  pause
)

endlocal


