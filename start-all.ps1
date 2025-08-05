# AgriPest AI Application Startup Script
# This script starts Ollama, Backend, and Frontend services

Write-Host "Starting AgriPest AI Application..." -ForegroundColor Green
Write-Host ""

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check if Ollama is installed
if (-not (Test-Command "ollama")) {
    Write-Host "ERROR: Ollama is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Ollama from https://ollama.ai/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Node.js is installed
if (-not (Test-Command "npm")) {
    Write-Host "ERROR: Node.js/npm is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Node.js from https://nodejs.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if required model exists
Write-Host "Checking for required Ollama model..." -ForegroundColor Cyan
$modelCheck = ollama list | Select-String "llama3.2:1b"
if (-not $modelCheck) {
    Write-Host "Model llama3.2:1b not found. Pulling model..." -ForegroundColor Yellow
    ollama pull llama3.2:1b
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to pull model" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Model downloaded successfully!" -ForegroundColor Green
}

Write-Host "Starting services..." -ForegroundColor Green
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Ollama service in new PowerShell window
Write-Host "[1/3] Starting Ollama service..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Ollama Service Running' -ForegroundColor Green; ollama serve" -WindowStyle Normal

# Wait for Ollama to start
Write-Host "Waiting for Ollama to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start backend server in new PowerShell window
Write-Host "[2/3] Starting Backend Server..." -ForegroundColor Cyan
$backendPath = Join-Path $scriptDir "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backendPath'; Write-Host 'Backend Server Starting...' -ForegroundColor Green; npm start" -WindowStyle Normal

# Wait for backend to start
Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Start frontend server in new PowerShell window
Write-Host "[3/3] Starting Frontend Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$scriptDir'; Write-Host 'Frontend Server Starting...' -ForegroundColor Green; npm start" -WindowStyle Normal

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   AgriPest AI Application Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Services running in separate windows:" -ForegroundColor White
Write-Host "  - Ollama Service (http://localhost:11434)" -ForegroundColor Cyan
Write-Host "  - Backend Server (http://localhost:8000)" -ForegroundColor Cyan
Write-Host "  - Frontend Server (http://localhost:3000)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Wait for all services to fully start, then open:" -ForegroundColor Yellow
Write-Host "  http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "To stop all services, close the individual PowerShell windows." -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to exit this window"