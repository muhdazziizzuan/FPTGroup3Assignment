@echo off
setlocal enabledelayedexpansion

REM Organic Farm Pest Management AI - Frontend Deployment Script (Windows)
REM This script helps deploy the frontend application using Docker on Windows

echo 🌱 Organic Farm Pest Management AI - Frontend Deployment
echo =================================================

REM Function to check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)
echo ✅ Docker is available

REM Function to check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Docker Compose is not installed. Using docker commands instead.
    set USE_COMPOSE=false
) else (
    echo ✅ Docker Compose is available
    set USE_COMPOSE=true
)

REM Get command argument
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=deploy

REM Main logic
if "%COMMAND%"=="deploy" goto :deploy
if "%COMMAND%"=="logs" goto :logs
if "%COMMAND%"=="stop" goto :stop
if "%COMMAND%"=="restart" goto :restart
if "%COMMAND%"=="help" goto :help

echo ❌ Unknown command: %COMMAND%
echo Use 'deploy.bat help' for available commands
pause
exit /b 1

:deploy
if "%USE_COMPOSE%"=="true" (
    echo 🚀 Building and starting with Docker Compose...
    docker-compose down --remove-orphans
    docker-compose up --build -d
    echo ✅ Application is running at http://localhost:3000
) else (
    echo 🚀 Building Docker image...
    docker build -t organic-farm-frontend .
    
    echo 🛑 Stopping existing container if running...
    docker stop organic-farm-frontend-container >nul 2>&1
    docker rm organic-farm-frontend-container >nul 2>&1
    
    echo 🚀 Starting new container...
    docker run -d --name organic-farm-frontend-container -p 3000:80 organic-farm-frontend
    
    echo ✅ Application is running at http://localhost:3000
)
echo.
echo Press any key to exit...
pause >nul
exit /b 0

:logs
if "%USE_COMPOSE%"=="true" (
    echo 📋 Showing application logs (Press Ctrl+C to exit):
    docker-compose logs -f frontend
) else (
    echo 📋 Showing application logs (Press Ctrl+C to exit):
    docker logs -f organic-farm-frontend-container
)
exit /b 0

:stop
if "%USE_COMPOSE%"=="true" (
    echo 🛑 Stopping application with Docker Compose...
    docker-compose down
) else (
    echo 🛑 Stopping Docker container...
    docker stop organic-farm-frontend-container >nul 2>&1
    docker rm organic-farm-frontend-container >nul 2>&1
)
echo ✅ Application stopped
echo.
echo Press any key to exit...
pause >nul
exit /b 0

:restart
if "%USE_COMPOSE%"=="true" (
    echo 🛑 Stopping application...
    docker-compose down
    timeout /t 2 /nobreak >nul
    echo 🚀 Starting application...
    docker-compose up --build -d
    echo ✅ Application restarted at http://localhost:3000
) else (
    echo 🛑 Stopping container...
    docker stop organic-farm-frontend-container >nul 2>&1
    docker rm organic-farm-frontend-container >nul 2>&1
    timeout /t 2 /nobreak >nul
    echo 🚀 Building and starting...
    docker build -t organic-farm-frontend .
    docker run -d --name organic-farm-frontend-container -p 3000:80 organic-farm-frontend
    echo ✅ Application restarted at http://localhost:3000
)
echo.
echo Press any key to exit...
pause >nul
exit /b 0

:help
echo Usage: deploy.bat [command]
echo Commands:
echo   deploy   - Build and start the application (default)
echo   logs     - Show application logs
echo   stop     - Stop the application
echo   restart  - Restart the application
echo   help     - Show this help message
echo.
echo Press any key to exit...
pause >nul
exit /b 0