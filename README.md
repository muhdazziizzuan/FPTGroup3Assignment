# Organic Farm Pest Management AI - Frontend

A modern React-based frontend application for the Organic Farm Pest Management AI System. This application provides farmers with an intuitive interface to interact with AI-powered pest identification and organic treatment recommendation services.

## Features

- **Chat Interface**: Natural language conversations with AI assistant for pest management advice
- **Image Upload**: Drag-and-drop or camera capture for pest image analysis
- **AI-Powered Analysis**: Real-time pest identification with confidence scores
- **Organic Treatment Recommendations**: Comprehensive organic solutions for identified pests
- **Responsive Design**: Mobile-friendly interface optimized for field use
- **Offline Capability**: Designed to work with offline-capable backend systems
- **Docker Ready**: Containerized for easy deployment

## Technology Stack

- **Frontend Framework**: React 18
- **Styling**: CSS3 with modern design patterns
- **Icons**: Lucide React
- **File Upload**: React Dropzone
- **HTTP Client**: Axios
- **Build Tool**: Create React App
- **Web Server**: Nginx (production)
- **Containerization**: Docker

## Quick Start

### Prerequisites

- Node.js 18+ (for development)
- Docker (for containerized deployment)
- Docker Compose (optional, for orchestration)

### Development Setup

1. **Clone and navigate to the project**:
   ```bash
   cd FPTGroup3Assignment
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm start
   ```

4. **Open your browser**:
   Navigate to `http://localhost:3000`

### Docker Deployment

#### Option 1: Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Access the application
# Open http://localhost:3000 in your browser

# Stop the container
docker-compose down
```

#### Option 2: Manual Docker Commands

```bash
# Build the Docker image
docker build -t organic-farm-frontend .

# Run the container
docker run -p 3000:80 organic-farm-frontend

# Access the application at http://localhost:3000
```

## Project Structure

```
src/
├── components/
│   ├── Header.js              # Application header with branding
│   ├── Header.css
│   ├── ChatInterface.js       # AI chat assistant interface
│   ├── ChatInterface.css
│   ├── ImageUpload.js         # Image upload and camera capture
│   ├── ImageUpload.css
│   ├── PestResults.js         # AI analysis results display
│   └── PestResults.css
├── App.js                     # Main application component
├── App.css
├── index.js                   # React entry point
└── index.css                  # Global styles

public/
├── index.html                 # HTML template
├── manifest.json              # PWA manifest
└── favicon.ico                # Application icon

Docker files:
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Container orchestration
└── nginx.conf                 # Production web server config
```

## Configuration

### Backend Integration

The frontend is configured to proxy API requests to the backend service. Update the following configurations based on your backend setup:

1. **Development Proxy** (package.json):
   ```json
   "proxy": "http://localhost:8000"
   ```

2. **Production Nginx** (nginx.conf):
   ```nginx
   location /api/ {
       proxy_pass http://backend:8000/;
   }
   ```

### Environment Variables

Create a `.env` file for environment-specific configurations:

```env
# API Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_MAX_FILE_SIZE=10485760

# Feature Flags
REACT_APP_ENABLE_CAMERA=true
REACT_APP_ENABLE_OFFLINE_MODE=true
```

## API Integration

The frontend expects the following backend endpoints:

### Image Analysis
```
POST /api/analyze-image
Content-Type: multipart/form-data

Body: FormData with 'image' field

Response:
{
  "pestName": "string",
  "confidence": "number (0-1)",
  "description": "string",
  "organicTreatments": ["string"],
  "severity": "string (Low|Medium|High)",
  "actionRequired": "string"
}
```

### Chat Interface
```
POST /api/chat
Content-Type: application/json

Body:
{
  "message": "string",
  "conversation_id": "string (optional)"
}

Response:
{
  "response": "string",
  "conversation_id": "string"
}
```

## Deployment Considerations

### Production Optimizations

1. **Build Optimization**: The Docker build uses multi-stage builds for smaller image size
2. **Nginx Configuration**: Optimized for serving static files and API proxying
3. **Caching**: Static assets are cached with appropriate headers
4. **Compression**: Gzip compression enabled for better performance

### Security

1. **File Upload Limits**: 10MB maximum file size
2. **CORS**: Configure backend CORS settings for your domain
3. **Content Security Policy**: Add CSP headers in production
4. **HTTPS**: Use HTTPS in production environments

### Scaling

1. **Load Balancing**: Use multiple container instances behind a load balancer
2. **CDN**: Serve static assets through a CDN
3. **API Gateway**: Use an API gateway for backend service management

## Development

### Available Scripts

- `npm start`: Start development server
- `npm build`: Build for production
- `npm test`: Run test suite
- `npm run eject`: Eject from Create React App (irreversible)

### Code Style

- Use functional components with hooks
- Follow React best practices
- Maintain responsive design principles
- Use semantic HTML elements
- Implement proper error handling

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Change port in docker-compose.yml if 3000 is occupied
2. **API Connection**: Verify backend service is running and accessible
3. **File Upload Issues**: Check file size limits and supported formats
4. **Build Failures**: Clear node_modules and reinstall dependencies

### Logs

```bash
# View container logs
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f frontend
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Test your changes thoroughly
3. Update documentation for new features
4. Ensure responsive design compatibility
5. Maintain accessibility standards

## License

This project is part of the Organic Farm Pest Management AI System educational project.

## Support

For technical support or questions about the frontend implementation, please refer to the project documentation or contact the development team.