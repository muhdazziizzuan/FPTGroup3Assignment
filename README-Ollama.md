# Ollama Integration for Pest Management AI

This project provides an AI-powered pest management system for organic farms using Ollama for local LLM processing.

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available

### Essential Commands to Run the Webapp

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Setup AI models (choose one):**
   ```bash
   # Windows
   setup-ollama.bat
   
   # Linux/Mac
   ./setup-ollama.sh
   
   # Manual setup
   docker-compose exec ollama ollama pull llama3.2:1b
   ```

3. **Verify everything is running:**
   ```bash
   # Check service status
   docker-compose ps
   
   # Test the API
   curl http://localhost:8000/health
   ```

4. **Access the webapp:**
   - Open browser: http://localhost:3000
   - Backend API: http://localhost:8000

### Essential Management Commands

```bash
# Stop the webapp
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f

# Update and rebuild
docker-compose up --build -d

# Check resource usage
docker stats
```

### Model Management Commands

```bash
# List available models
docker-compose exec ollama ollama list

# Pull additional models
docker-compose exec ollama ollama pull gemma2:2b

# Remove unused models
docker-compose exec ollama ollama rm model-name

# Test model directly
docker-compose exec ollama ollama run llama3.2:1b "Hello"
```

### Troubleshooting Commands

```bash
# If services won't start
docker-compose down
docker-compose up -d

# If models are missing
docker-compose exec ollama ollama pull llama3.2:1b

# Check specific service logs
docker-compose logs backend
docker-compose logs ollama
docker-compose logs frontend

# Reset everything (WARNING: removes all data)
docker-compose down -v
docker-compose up -d
```

## Architecture

```
Frontend (React) → Backend (Node.js) → Ollama (LLM Service)
     ↓                    ↓                    ↓
   Port 3000           Port 8000           Port 11434
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Setup Ollama models:**
   ```bash
   # On Windows
   setup-ollama.bat
   
   # On Linux/Mac
   chmod +x setup-ollama.sh
   ./setup-ollama.sh
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Ollama API: http://localhost:11434

### Option 2: Manual Setup

1. **Install Ollama locally:**
   - Visit https://ollama.ai and download for your OS
   - Or use Docker: `docker run -d -p 11434:11434 ollama/ollama`

2. **Pull required models:**
   ```bash
   ollama pull llama3.2
   ollama pull llava  # For image analysis
   ```

3. **Start backend service:**
   ```bash
   cd backend
   npm install
   npm start
   ```

4. **Start frontend:**
   ```bash
   npm install
   npm start
   ```

## Available Models

### llama3.2 (Default)
- **Purpose:** General conversation and pest management advice
- **Size:** ~2GB
- **Best for:** Text-based queries, treatment recommendations

### llava
- **Purpose:** Image analysis and visual pest identification
- **Size:** ~4GB
- **Best for:** Analyzing uploaded pest images

## API Endpoints

### POST /api/chat
Send messages to the AI assistant.

**Request:**
```javascript
// Text only
{
  "message": "How do I treat aphids organically?",
  "model": "llama3.2"  // optional
}

// With images (multipart/form-data)
FormData with:
- message: "What pest is this?"
- images: [File objects]
- model: "llava"  // recommended for images
```

**Response:**
```javascript
{
  "response": "AI generated response...",
  "model": "llama3.2",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### GET /api/models
List available models.

### POST /api/models/pull
Pull a new model.

### GET /ollama/status
Check Ollama connection status.

## Configuration

### Environment Variables

**Backend (.env):**
```env
PORT=8000
OLLAMA_URL=http://localhost:11434
NODE_ENV=production
```

**Docker Compose:**
```yaml
environment:
  - OLLAMA_URL=http://ollama:11434
  - OLLAMA_KEEP_ALIVE=24h
```

## GPU Support

To enable GPU acceleration for faster inference:

1. **Uncomment GPU configuration in docker-compose.yml:**
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

2. **Ensure NVIDIA Docker runtime is installed:**
   ```bash
   # Install nvidia-docker2
   sudo apt-get install nvidia-docker2
   sudo systemctl restart docker
   ```

## Troubleshooting

### Common Issues

1. **"Ollama service is not available"**
   - Check if Ollama container is running: `docker-compose ps`
   - Verify Ollama is accessible: `curl http://localhost:11434/api/tags`

2. **"Model not found"**
   - Pull the required model: `ollama pull llama3.2`
   - Check available models: `curl http://localhost:11434/api/tags`

3. **Slow responses**
   - Enable GPU support (see above)
   - Use smaller models like `llama3.2:8b` instead of larger variants
   - Increase `OLLAMA_KEEP_ALIVE` to keep models in memory

4. **Out of memory errors**
   - Reduce model size or use quantized versions
   - Increase Docker memory limits
   - Close other applications using GPU/RAM

### Logs

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f ollama
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Model Recommendations

### For Development/Testing
- **llama3.2:8b** - Good balance of speed and quality
- **phi3:mini** - Very fast, smaller responses

### For Production
- **llama3.2:70b** - Best quality (requires powerful hardware)
- **llama3.2:13b** - Good quality, moderate resource usage

### For Image Analysis
- **llava:7b** - Standard vision model
- **llava:13b** - Better accuracy for complex images

## Performance Optimization

1. **Keep models warm:**
   ```yaml
   environment:
     - OLLAMA_KEEP_ALIVE=24h
   ```

2. **Use appropriate model sizes:**
   - Development: 7B-8B parameter models
   - Production: 13B+ parameter models

3. **Enable GPU acceleration** (see GPU Support section)

4. **Adjust request timeouts:**
   ```javascript
   // In backend/server.js
   timeout: 60000 // Increase for larger models
   ```

## Security Considerations

- Ollama runs locally, so no data is sent to external services
- File uploads are limited to 10MB and 5 files maximum
- Uploaded files are automatically cleaned up after processing
- Consider adding authentication for production deployments

## Development

### Adding New Models

1. **Pull the model:**
   ```bash
   docker-compose exec ollama ollama pull model-name
   ```

2. **Update the backend to support it:**
   ```javascript
   // In backend/server.js, modify the default model
   const { message, model = 'new-model-name' } = req.body;
   ```

### Custom Prompts

Modify the system prompt in `backend/server.js`:

```javascript
const systemPrompt = `Your custom system prompt here...`;
```

## Support

For issues related to:
- **Ollama:** https://github.com/ollama/ollama
- **Models:** https://ollama.ai/library
- **This integration:** Check the project's GitHub issues