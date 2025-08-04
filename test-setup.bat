@echo off
echo ========================================
echo  Pest Management AI - Testing Setup
echo ========================================
echo.

echo Step 1: Checking Docker...
docker --version
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Step 2: Building and starting services...
docker-compose down
docker-compose up -d --build

echo Step 3: Waiting for services to start...
timeout /t 10 /nobreak

echo Step 4: Checking service status...
docker-compose ps

echo.
echo Step 5: Setting up Ollama models...
echo This may take a few minutes to download models...
echo.

echo Waiting for Ollama to be ready...
:wait_ollama
curl -f http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting for Ollama service...
    timeout /t 5 /nobreak >nul
    goto wait_ollama
)

echo Ollama is ready! Pulling models...
echo.
echo Pulling llama3.2 (this may take 5-10 minutes)...
curl -X POST http://localhost:11434/api/pull -d "{\"name\": \"llama3.2\"}"

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Your services are now running:
echo - Frontend: http://localhost:3000
echo - Backend:  http://localhost:8000
echo - Ollama:   http://localhost:11434
echo.
echo Open http://localhost:3000 in your browser to test!
echo.
pause