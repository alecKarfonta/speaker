# Speaker TTS API

A high-quality Text-to-Speech API with XTTS v2 voice cloning capabilities, featuring real-time speech generation, multi-language support, and emotion control.

## 🚀 Features

- **Voice Cloning**: Upload audio samples to create custom voices
- **Multi-language Support**: Generate speech in 29+ languages
- **Emotion Control**: Add emotion tags to control speech style
- **Real-time Generation**: Fast audio generation with configurable parameters
- **RESTful API**: Clean, well-documented REST API with OpenAPI/Swagger
- **Monitoring & Metrics**: Built-in performance monitoring and health checks
- **Rate Limiting**: Configurable rate limiting to prevent abuse
- **Audit Logging**: Comprehensive audit trail for sensitive operations
- **Performance Monitoring**: Intelligent word-based performance tracking with Prometheus metrics

## 📋 Stage 1 Improvements

### ✅ API Enhancements
- [x] OpenAPI documentation and interactive docs (Swagger/Redoc)
- [x] API versioning for future-proofing
- [x] More granular error handling and status codes
- [x] Rate limiting (100 requests/minute per IP)
- [x] Endpoint for voice sample management (list, delete)
- [x] Endpoint for supported languages and voices metadata

### ✅ Model & Performance
- [x] Enhanced health monitoring with system metrics
- [x] Performance tracking and response time monitoring
- [x] GPU/CPU health reporting (when available)

### ✅ Security
- [x] Audit logging for sensitive actions
- [x] Secure file uploads with validation
- [x] CORS configuration and security headers
- [x] Comprehensive error handling and logging

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Start with monitoring stack
./scripts/start_monitoring.sh

# Or start basic services only
docker-compose up -d
```

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API:
```bash
python app/main.py
```

The API will be available at `http://localhost:8010`

## 📚 API Documentation

- **Interactive Docs**: http://localhost:8010/docs
- **OpenAPI Spec**: http://localhost:8010/openapi.json

## 🔧 Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# API Configuration
API_TITLE=Speaker TTS API
API_VERSION=1.0.0
HOST=0.0.0.0
PORT=8010
DEBUG=false

# Environment
ENVIRONMENT=production  # development, staging, production

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# CORS
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com

# Security
API_KEY_REQUIRED=false
API_KEYS=key1,key2,key3

# Logging
LOG_LEVEL=INFO
AUDIT_LOG_FILE=logs/audit.log

# Performance
MAX_TEXT_LENGTH=2000
MAX_CONCURRENT_REQUESTS=50
```

### Configuration Files

The API uses multiple configuration files:

- `app/config.yaml`: TTS model configuration
- `app/voicebox_config.json`: Voice processing settings
- Environment variables: Runtime configuration

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_stage1_improvements.py

# Run with coverage
pytest --cov=app
```

### Test Coverage
The test suite covers:
- API endpoints and responses
- Error handling and validation
- Rate limiting functionality
- Monitoring and metrics
- Configuration validation

## 📈 Monitoring & Observability

### Health Checks
The API provides comprehensive health monitoring:

- **System Health**: CPU, memory, disk usage
- **GPU Health**: GPU memory and utilization (when available)
- **Service Health**: Model status, available voices
- **Performance Metrics**: Response times, request rates

### Metrics Endpoint
```http
GET /metrics
```

Returns:
- Total requests and success/failure rates
- Average response times
- Requests per minute
- Voice and language usage statistics
- Error counts by type

### Audit Logging
All sensitive operations are logged to `logs/audit.log`:
- Voice uploads and deletions
- TTS generation requests
- Error occurrences
- Client IP and user agent information

## 🚀 Deployment

### Docker
```bash
# Build image
docker build -t speaker-tts .

# Run container
docker run -p 8010:8010 speaker-tts
```

### Docker Compose
```bash
# Start with monitoring stack
./scripts/start_monitoring.sh

# Or start basic services only
docker-compose up -d
```

### Kubernetes
```bash
# Apply configurations
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app=speaker-tts
```

## 🔒 Security

### Current Security Features
- Rate limiting to prevent abuse
- File upload validation
- CORS configuration
- Audit logging
- Error handling without information leakage

### Recommended Production Security
- Enable API key authentication
- Configure proper CORS origins
- Use HTTPS/TLS
- Set up proper firewall rules
- Monitor audit logs
- Regular security updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: http://localhost:8010/docs
- **Health Check**: http://localhost:8010/health
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions

## 🔄 Changelog

### Version 1.0.0 (Stage 1)
- ✅ OpenAPI documentation and interactive docs
- ✅ API versioning and enhanced error handling
- ✅ Rate limiting and CORS configuration
- ✅ Comprehensive monitoring and metrics
- ✅ Audit logging for security
- ✅ Enhanced voice management endpoints
- ✅ Multi-language support documentation
- ✅ Performance tracking and health monitoring
- ✅ Configuration management system
- ✅ Comprehensive test suite

---

**Next Steps**: See `plans/next-steps.md` for upcoming improvements and roadmap.

## Performance Monitoring

The TTS API includes comprehensive performance monitoring with intelligent word-based tracking:

### Metrics Available

- **Request Performance**: Response times by endpoint, voice, and language
- **Word-based Analysis**: Performance categorized by word count ranges (1-10, 11-25, 26-50, 51-100, 101-200, 200+)
- **Character-based Analysis**: Performance by character count ranges
- **System Resources**: CPU, memory, and GPU usage
- **Error Tracking**: Error rates and types
- **Voice Usage**: Most popular voices and languages

### Monitoring Stack

The monitoring stack is now integrated into the main Docker Compose setup:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Beautiful dashboards for visualization
- **Real-time Monitoring**: Live performance tracking

### Accessing Metrics

- **Prometheus Metrics**: http://localhost:8010/prometheus
- **API Metrics**: http://localhost:8010/metrics
- **Prometheus UI**: http://localhost:9090
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Frontend**: http://localhost:3010

### Key Metrics

The system tracks these intelligent metrics:

1. **Word Count Performance**: 
   - Tracks response time based on actual word count
   - Handles emotion tags intelligently (e.g., "(happy) Hello world" counts as 2 words)
   - Categorizes into meaningful ranges for analysis

2. **Character Count Performance**:
   - Tracks performance by character count ranges
   - Useful for understanding text length impact

3. **Voice-specific Performance**:
   - Each voice's performance is tracked separately
   - Helps identify which voices are faster/slower

4. **System Health**:
   - Real-time CPU and memory usage
   - GPU utilization (if available)
   - Model load times

### Prometheus Metrics

The API exposes these Prometheus metrics:

- `tts_requests_total`: Total request count with labels
- `tts_request_duration_seconds`: Request duration histograms
- `tts_word_count_duration_seconds`: Word-based performance
- `tts_character_count_duration_seconds`: Character-based performance
- `tts_active_voices`: Number of active voices
- `tts_system_memory_usage_bytes`: Memory usage
- `tts_system_cpu_usage_percent`: CPU usage
- `tts_errors_total`: Error counts by type

### Grafana Dashboard

The included Grafana dashboard provides:

- Request rate visualization
- Response time heatmaps by word count
- Voice performance comparison
- Error rate tracking
- System resource monitoring
- Real-time performance distribution
