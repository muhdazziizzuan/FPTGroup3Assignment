@echo off
echo Starting AgriPest AI Application...
echo.

REM Set Ollama path
set "OLLAMA_PATH=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"

REM Check if Ollama is installed
if not exist "%OLLAMA_PATH%" (
    echo ERROR: Ollama is not installed at expected location
    echo Expected location: %OLLAMA_PATH%
    echo Please install Ollama from https://ollama.ai/
    pause
    exit /b 1
)

REM Check if required model exists
echo Checking for required Ollama model...
"%OLLAMA_PATH%" list | findstr "phi4-mini" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Model phi4-mini not found. Pulling model...
    "%OLLAMA_PATH%" pull phi4-mini
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to pull model
        pause
        exit /b 1
    )
)

echo Starting services...
echo.

REM Start Ollama service in background
echo [1/3] Starting Ollama service...
start /b "%OLLAMA_PATH%" serve

REM Wait a moment for Ollama to start
timeout /t 3 /nobreak >nul
echo Waiting for Ollama to start...

REM Start backend server in background
echo [2/3] Starting Backend Server...
cd /d "%~dp0backend"
start /b npm start
cd /d "%~dp0"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul
echo Waiting for backend to start...

REM Start frontend server in current terminal
echo [3/3] Starting Frontend Server...
echo Frontend Server Starting...
npm start

echo.
echo ========================================
echo   AgriPest AI Application Started!
echo ========================================
echo.
echo Services running in separate windows:
echo   - Ollama Service (http://localhost:11434)
echo   - Backend Server (http://localhost:8000)
echo   - Frontend Server (http://localhost:3000)
echo.
echo Wait for all services to fully start, then open:
echo   http://localhost:3000
echo.
echo Press any key to exit this window...
pause >nul