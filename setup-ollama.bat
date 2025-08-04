@echo off
echo Setting up Ollama models for Pest Management AI...

echo Waiting for Ollama service to start...
:wait_loop
curl -f http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting for Ollama to be ready...
    timeout /t 5 /nobreak >nul
    goto wait_loop
)

echo Ollama is ready! Pulling required models...

echo Pulling llama3.2 model (recommended for general chat)...
curl -X POST http://localhost:11434/api/pull -d "{\"name\": \"llama3.2\"}"

echo.
echo Pulling llava model (for image analysis)...
curl -X POST http://localhost:11434/api/pull -d "{\"name\": \"llava\"}"

echo.
echo Setup complete! Available models:
curl http://localhost:11434/api/tags

echo.
echo You can now use the following models:
echo - llama3.2: General purpose chat model
echo - llava: Vision model for image analysis
echo.
echo The backend service will use llama3.2 by default.
echo To use a different model, send a request to /api/chat with the 'model' parameter.

pause