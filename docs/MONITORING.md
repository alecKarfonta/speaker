# TTS API Performance Monitoring

This document describes the comprehensive performance monitoring system implemented for the TTS API, featuring intelligent word-based performance tracking and Prometheus metrics integration.

## Overview

The monitoring system provides real-time insights into TTS API performance, with special focus on:

- **Word-based performance analysis**: Tracks response times based on actual word count
- **Character-based analysis**: Performance by character count ranges
- **Voice-specific metrics**: Individual voice performance tracking
- **System health monitoring**: CPU, memory, and GPU utilization
- **Error tracking**: Comprehensive error rate monitoring

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TTS API       │    │   Prometheus    │    │   Grafana       │
│   (FastAPI)     │───▶│   (Metrics)     │───▶│   (Dashboard)   │
│                 │    │                 │    │                 │
│ - /prometheus   │    │ - Scraping      │    │ - Visualization │
│ - /metrics      │    │ - Storage       │    │ - Alerts        │
│ - /health       │    │ - Querying      │    │ - Reports       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Features

### 1. Intelligent Word Counting

The system intelligently counts words by:
- Removing emotion tags like `(happy)`, `(sad)`, etc.
- Handling various text formats and punctuation
- Categorizing into meaningful ranges:
  - 1-10 words
  - 11-25 words
  - 26-50 words
  - 51-100 words
  - 101-200 words
  - 200+ words

### 2. Performance Metrics

#### Request Metrics
- `tts_requests_total`: Total request count with labels
- `tts_request_duration_seconds`: Request duration histograms
- `tts_errors_total`: Error counts by type

#### Word-based Metrics
- `tts_word_count_duration_seconds`: Performance by word count range
- `tts_character_count_duration_seconds`: Performance by character count

#### System Metrics
- `tts_active_voices`: Number of active voices
- `tts_system_memory_usage_bytes`: Memory usage
- `tts_system_cpu_usage_percent`: CPU usage
- `tts_model_load_time_seconds`: Model loading time

### 3. Prometheus Integration

The API exposes metrics at `/prometheus` endpoint in Prometheus format:

```bash
# Example metrics
tts_requests_total{endpoint="/tts",method="POST",status_code="200",voice_name="biden",language="en"} 42
tts_request_duration_seconds_bucket{endpoint="/tts",voice_name="biden",language="en",le="1.0"} 35
tts_word_count_duration_seconds_bucket{word_count_range="11-25",voice_name="biden",le="2.5"} 28
```

## Implementation Details

### Monitoring Module (`app/monitoring.py`)

The core monitoring functionality is implemented in the `MetricsCollector` class:

```python
class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        # Word-based performance tracking
        self.word_count_performance = defaultdict(list)
        self.character_count_performance = defaultdict(list)
    
    def _count_words(self, text: str) -> int:
        """Count words intelligently, handling emotion tags"""
        text = re.sub(r'\([^)]*\)', '', text)  # Remove emotion tags
        words = [word for word in text.split() if word.strip()]
        return len(words)
    
    def record_request(self, endpoint: str, method: str, status_code: int, 
                      response_time: float, voice_name: Optional[str] = None,
                      language: Optional[str] = None, text_length: Optional[int] = None,
                      text: Optional[str] = None):
        """Record request with word-based performance tracking"""
```

### API Integration (`app/main.py`)

The API integrates monitoring through:

1. **Middleware**: Automatically tracks all requests
2. **TTS Endpoint**: Enhanced to pass actual text for word counting
3. **Prometheus Endpoint**: Exposes metrics for scraping

```python
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response_time = time.time() - start_time
    
    # Record metrics with enhanced tracking
    metrics_collector.record_request(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        response_time=response_time,
        voice_name=voice_name,
        language=language,
        text_length=text_length,
        text=text  # Pass actual text for word counting
    )
    
    # Update system metrics for Prometheus
    metrics_collector.update_system_metrics()
    
    return response
```

## Deployment Options

### 1. Local Development with Docker Compose

```bash
# Start with monitoring stack
./scripts/start_monitoring.sh

# Or manually
docker-compose -f docker-compose.monitoring.yml up -d
```

This starts:
- TTS API on port 8010
- Prometheus on port 9090
- Grafana on port 3000
- Frontend on port 3001

### 2. Kubernetes Deployment

```bash
# Deploy monitoring stack
kubectl apply -f k8s/monitoring/
```

Includes:
- Prometheus deployment with persistent storage
- Grafana with pre-configured dashboard
- Service monitors for automatic discovery

## Monitoring Dashboards

### Grafana Dashboard

The included Grafana dashboard provides:

1. **Request Rate**: Real-time request rate visualization
2. **Word Count Performance**: Heatmap showing performance by word count
3. **Voice Performance**: Comparison of different voices
4. **Error Rate**: Error tracking and alerting
5. **System Resources**: CPU, memory, and GPU monitoring
6. **Response Time Distribution**: Histogram of response times

### Dashboard Panels

- **Request Rate**: `rate(tts_requests_total[5m])`
- **Word Count Performance**: `rate(tts_word_count_duration_seconds_bucket[5m])`
- **P95 Response Time**: `histogram_quantile(0.95, rate(tts_request_duration_seconds_bucket[5m]))`
- **Error Rate**: `rate(tts_errors_total[5m])`
- **System CPU**: `tts_system_cpu_usage_percent`
- **System Memory**: `tts_system_memory_usage_bytes / 1024 / 1024 / 1024`

## Testing and Validation

### Test Script

Use the provided test script to generate metrics:

```bash
python scripts/test_monitoring.py
```

This script:
- Makes various TTS requests with different word counts
- Tests different voices and languages
- Generates performance data for analysis
- Shows real-time metrics

### Manual Testing

```bash
# Test TTS endpoint
curl -X POST "http://localhost:8010/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "voice_name": "biden", "language": "en"}'

# Check metrics
curl "http://localhost:8010/metrics"
curl "http://localhost:8010/prometheus"
```

## Performance Insights

### Word Count Analysis

The system reveals patterns like:
- Short texts (1-10 words): ~0.5-1.5 seconds
- Medium texts (11-25 words): ~1.5-3.0 seconds
- Long texts (26-50 words): ~3.0-5.0 seconds
- Very long texts (50+ words): ~5.0+ seconds

### Voice Performance

Different voices may show varying performance:
- Some voices are consistently faster
- Performance may vary by language
- Model loading affects initial response times

### System Optimization

Monitor these metrics for optimization:
- High CPU usage: Consider scaling or optimization
- Memory pressure: Check for memory leaks
- Slow response times: Investigate bottlenecks
- Error rates: Identify problematic requests

## Alerting

### Prometheus Alerts

Configure alerts for:
- High error rates: `rate(tts_errors_total[5m]) > 0.1`
- Slow response times: `histogram_quantile(0.95, rate(tts_request_duration_seconds_bucket[5m])) > 10`
- High CPU usage: `tts_system_cpu_usage_percent > 80`
- High memory usage: `tts_system_memory_usage_bytes / 1024 / 1024 / 1024 > 8`

### Grafana Alerts

Set up Grafana alerts for:
- Dashboard-based thresholds
- Anomaly detection
- Business metrics

## Troubleshooting

### Common Issues

1. **Metrics not appearing**: Check if `/prometheus` endpoint is accessible
2. **High response times**: Monitor word count performance
3. **Memory issues**: Check system memory metrics
4. **Prometheus connection**: Verify network connectivity

### Debug Commands

```bash
# Check API health
curl "http://localhost:8010/health"

# View raw metrics
curl "http://localhost:8010/prometheus"

# Check Prometheus targets
curl "http://localhost:9090/api/v1/targets"

# Test Grafana connection
curl "http://localhost:3000/api/health"
```

## Future Enhancements

### Planned Features

1. **Advanced Analytics**: Machine learning-based performance prediction
2. **Custom Dashboards**: User-configurable monitoring views
3. **Alert Integration**: Slack, email, and webhook notifications
4. **Performance Optimization**: Automatic scaling based on metrics
5. **Cost Analysis**: Resource usage cost tracking

### Metrics Expansion

- **Quality Metrics**: Audio quality assessment
- **User Experience**: End-to-end request tracking
- **Business Metrics**: Usage patterns and trends
- **Security Metrics**: Authentication and authorization tracking

## Conclusion

The monitoring system provides comprehensive insights into TTS API performance, enabling:

- **Performance optimization** based on word count patterns
- **Capacity planning** using historical data
- **Issue detection** through real-time monitoring
- **Quality assurance** with detailed metrics
- **Business intelligence** through usage analytics

This intelligent monitoring approach ensures the TTS API operates efficiently and provides the best possible user experience. 