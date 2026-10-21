@echo off
rem Dainamo Trainer launcher for Windows: sets up the Python env and web UI on first run (and again
rem when requirements.txt or the web UI's source changes, as after a git pull), then starts the server.
rem Options pass through, e.g.  launch.bat --host 0.0.0.0   (see: launch.bat --help)
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo Creating Python environment...
  where uv >nul 2>nul
  if not errorlevel 1 (
    uv venv --python 3.12 .venv || goto :fail
  ) else (
    where py >nul 2>nul
    if not errorlevel 1 (
      py -3.12 -m venv .venv || goto :fail
    ) else (
      python -m venv .venv || goto :fail
    )
  )
)

fc /b requirements.txt .venv\requirements.installed >nul 2>nul
if errorlevel 1 (
  echo Installing Python packages, PyTorch with CUDA among them. This takes a while the first time...
  where uv >nul 2>nul
  if not errorlevel 1 (
    uv pip install --python "%PY%" -r requirements.txt || goto :fail
  ) else (
    "%PY%" -m pip install -r requirements.txt || goto :fail
  )
  copy /y requirements.txt .venv\requirements.installed >nul
)

"%PY%" -m dainamo.webui check
if not errorlevel 1 goto :ui_ready
where npm >nul 2>nul
if errorlevel 1 goto :no_npm
if exist web\dist\index.html (
  echo The web UI changed since it was built. Rebuilding it...
) else (
  echo Building web UI...
)
pushd web
call npm install || (popd & goto :fail)
call npm run build || (popd & goto :fail)
popd
"%PY%" -m dainamo.webui built || goto :fail
goto :ui_ready
:no_npm
if not exist web\dist\index.html (
  echo The web UI isn't built and npm wasn't found. Install Node.js 18 or newer from https://nodejs.org and run this again.
  goto :fail
)
echo Note: the web UI changed since it was built, but npm wasn't found to rebuild it, so this is the older UI.
:ui_ready

where ffmpeg >nul 2>nul
if errorlevel 1 echo Note: ffmpeg wasn't found. Video datasets and video samples need it: winget install ffmpeg

"%PY%" -m dainamo %*
if errorlevel 1 pause
exit /b

:fail
echo.
echo Setup didn't finish; see the messages above.
pause
exit /b 1
