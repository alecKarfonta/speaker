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

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd speaker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   python -m app.main
   ```

4. **Access the API**
   - API: http://localhost:8010
   - Documentation: http://localhost:8010/docs
   - Health Check: http://localhost:8010/health

## 📚 API Documentation

### Base URL
```
http://localhost:8010
```

### Authentication
Currently, the API is open access with rate limiting. API key authentication can be enabled via configuration.

### Rate Limits
- 100 requests per minute per IP address
- Configurable via environment variables

### Endpoints

#### 🏠 Root Endpoint
```http
GET /
```
Returns API information and status.

**Response:**
```json
{
  "message": "Welcome to the Speaker TTS API",
  "version": "1.0.0",
  "model": "xtts_v2",
  "status": "operational",
  "documentation": "/docs",
  "health_check": "/health"
}
```

#### 🎯 TTS Generation
```http
POST /tts
```

**Request Body:**
```json
{
  "text": "Hello, this is a test of the TTS system.",
  "voice_name": "demo_1",
  "language": "en",
  "temperature": 0.8,
  "top_p": 0.9,
  "emotion": "(happy)",
  "speed": 1.0
}
```

**Parameters:**
- `text` (string, required): Text to convert (1-2000 characters)
- `voice_name` (string, required): Name of the voice to use
- `language` (string, optional): Two-letter language code (default: "en")
- `temperature` (float, optional): Randomness control (0.1-1.0, default: 0.8)
- `top_p` (float, optional): Top-p sampling (0.1-1.0, default: 0.9)
- `emotion` (string, optional): Emotion tag like "(happy)", "(sad)", etc.
- `speed` (float, optional): Speech speed multiplier (0.5-2.0, default: 1.0)

**Response:** Audio data in WAV format

#### 🎤 Voice Management

**List Voices**
```http
GET /voices
```

**Upload Voice**
```http
POST /voices
Content-Type: multipart/form-data

voice_name: my_voice
file: [audio_file.wav]
```

**Delete Voice**
```http
DELETE /voices/{voice_name}
```

#### 🌍 Languages
```http
GET /languages
```
Returns supported languages and their codes.

#### 💓 Health Check
```http
GET /health
```
Returns service health status and system metrics.

#### 📊 Metrics
```http
GET /metrics
```
Returns API usage metrics and performance statistics.

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
# Start services
docker-compose up -d

# View logs
docker-compose logs -f
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
