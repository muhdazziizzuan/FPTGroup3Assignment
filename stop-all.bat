@echo off
echo Stopping AgriPest AI Application services...
echo.

REM Kill Node.js processes (frontend and backend)
echo Stopping Node.js servers...
taskkill /f /im node.exe 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   - Node.js servers stopped
) else (
    echo   - No Node.js servers were running
)

REM Kill Ollama process
echo Stopping Ollama service...
taskkill /f /im ollama.exe 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   - Ollama service stopped
) else (
    echo   - Ollama service was not running
)

echo.
echo All AgriPest AI services have been stopped.
echo.
pause