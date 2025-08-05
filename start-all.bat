@echo off
echo Starting AgriPest AI Application...
echo.

REM Check if Ollama is installed
where ollama >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Ollama is not installed or not in PATH
    echo Please install Ollama from https://ollama.ai/
    pause
    exit /b 1
)

REM Check if required model exists
echo Checking for required Ollama model...
ollama list | findstr "llama3.2:1b" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Model llama3.2:1b not found. Pulling model...
    ollama pull llama3.2:1b
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to pull model
        pause
        exit /b 1
    )
)

echo Starting services...
echo.

REM Start Ollama service in new window
echo [1/3] Starting Ollama service...
start "Ollama Service" cmd /k "echo Ollama Service Running && ollama serve"

REM Wait a moment for Ollama to start
timeout /t 3 /nobreak >nul

REM Start backend server in new window
echo [2/3] Starting Backend Server...
start "Backend Server" cmd /k "cd /d %~dp0backend && echo Backend Server Starting... && npm start"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server in new window
echo [3/3] Starting Frontend Server...
start "Frontend Server" cmd /k "cd /d %~dp0 && echo Frontend Server Starting... && npm start"

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