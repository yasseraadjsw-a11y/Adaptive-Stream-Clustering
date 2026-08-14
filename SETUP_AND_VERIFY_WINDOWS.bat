@echo off
setlocal
python -m pip install --upgrade pip || exit /b 1
python -m pip install -r requirements.txt || exit /b 1
python -m compileall -q src experiments scripts tests main.py || exit /b 1
python main.py verify || exit /b 1
endlocal
