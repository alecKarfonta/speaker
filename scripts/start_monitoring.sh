#!/bin/bash

# Start TTS API with monitoring stack
echo "Starting TTS API with monitoring stack..."

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Check if the main compose file exists
if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml not found"
    exit 1
fi

# Start the complete stack with monitoring
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 20

# Check service status
echo "Checking service status..."
docker-compose ps

echo ""
echo "Complete stack started successfully!"
echo ""
echo "Services:"
echo "  - TTS API: http://localhost:8010"
echo "  - API Docs: http://localhost:8010/docs"
echo "  - Frontend: http://localhost:3010"
echo "  - Prometheus: http://localhost:9199"
echo "  - Grafana: http://localhost:3333 (admin/admin123)"
echo ""
echo "Metrics endpoints:"
echo "  - Prometheus metrics: http://localhost:8010/prometheus"
echo "  - API metrics: http://localhost:8010/metrics"
echo "  - Health check: http://localhost:8010/health"
echo ""
echo "To stop the stack:"
echo "  docker-compose down"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To test monitoring:"
echo "  python scripts/test_monitoring.py" 