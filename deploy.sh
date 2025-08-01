#!/bin/bash

# Organic Farm Pest Management AI - Frontend Deployment Script
# This script helps deploy the frontend application using Docker

set -e

echo "🌱 Organic Farm Pest Management AI - Frontend Deployment"
echo "================================================="

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    echo "✅ Docker is available"
}

# Function to check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        echo "⚠️  Docker Compose is not installed. Using docker commands instead."
        return 1
    fi
    echo "✅ Docker Compose is available"
    return 0
}

# Function to build and run with Docker Compose
deploy_with_compose() {
    echo "🚀 Building and starting with Docker Compose..."
    docker-compose down --remove-orphans
    docker-compose up --build -d
    echo "✅ Application is running at http://localhost:3000"
}

# Function to build and run with Docker commands
deploy_with_docker() {
    echo "🚀 Building Docker image..."
    docker build -t organic-farm-frontend .
    
    echo "🛑 Stopping existing container if running..."
    docker stop organic-farm-frontend-container 2>/dev/null || true
    docker rm organic-farm-frontend-container 2>/dev/null || true
    
    echo "🚀 Starting new container..."
    docker run -d \
        --name organic-farm-frontend-container \
        -p 3000:80 \
        organic-farm-frontend
    
    echo "✅ Application is running at http://localhost:3000"
}

# Function to show logs
show_logs() {
    if check_docker_compose; then
        echo "📋 Showing application logs (Press Ctrl+C to exit):"
        docker-compose logs -f frontend
    else
        echo "📋 Showing application logs (Press Ctrl+C to exit):"
        docker logs -f organic-farm-frontend-container
    fi
}

# Function to stop the application
stop_application() {
    if check_docker_compose; then
        echo "🛑 Stopping application with Docker Compose..."
        docker-compose down
    else
        echo "🛑 Stopping Docker container..."
        docker stop organic-farm-frontend-container 2>/dev/null || true
        docker rm organic-farm-frontend-container 2>/dev/null || true
    fi
    echo "✅ Application stopped"
}

# Main deployment logic
main() {
    check_docker
    
    case "${1:-deploy}" in
        "deploy")
            if check_docker_compose; then
                deploy_with_compose
            else
                deploy_with_docker
            fi
            ;;
        "logs")
            show_logs
            ;;
        "stop")
            stop_application
            ;;
        "restart")
            stop_application
            sleep 2
            if check_docker_compose; then
                deploy_with_compose
            else
                deploy_with_docker
            fi
            ;;
        "help")
            echo "Usage: $0 [command]"
            echo "Commands:"
            echo "  deploy   - Build and start the application (default)"
            echo "  logs     - Show application logs"
            echo "  stop     - Stop the application"
            echo "  restart  - Restart the application"
            echo "  help     - Show this help message"
            ;;
        *)
            echo "❌ Unknown command: $1"
            echo "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"