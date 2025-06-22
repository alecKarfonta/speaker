"""
Monitoring and metrics module for the TTS API.
Provides functionality for tracking API usage, performance metrics, and health monitoring.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and tracks API metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.start_time = time.time()
        
        # Request tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_times = deque(maxlen=max_history)
        self.request_history = deque(maxlen=max_history)
        
        # Voice usage tracking
        self.voice_usage = defaultdict(int)
        self.language_usage = defaultdict(int)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        
        # Performance tracking
        self.model_load_time = None
        self.last_request_time = None
        
        # Thread safety
        self._lock = threading.Lock()
    
    def record_request(self, endpoint: str, method: str, status_code: int, 
                      response_time: float, voice_name: Optional[str] = None,
                      language: Optional[str] = None, text_length: Optional[int] = None):
        """Record a request and its metrics"""
        with self._lock:
            self.total_requests += 1
            self.last_request_time = datetime.utcnow()
            
            if 200 <= status_code < 400:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                self.error_counts[f"{status_code}"] += 1
            
            # Record timing
            self.request_times.append(response_time)
            self.request_history.append({
                'timestamp': datetime.utcnow(),
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time': response_time,
                'voice_name': voice_name,
                'language': language,
                'text_length': text_length
            })
            
            # Track voice and language usage
            if voice_name:
                self.voice_usage[voice_name] += 1
            if language:
                self.language_usage[language] += 1
    
    def record_model_load_time(self, load_time: float):
        """Record model loading time"""
        with self._lock:
            self.model_load_time = load_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            avg_response_time = (
                sum(self.request_times) / len(self.request_times) 
                if self.request_times else 0
            )
            
            # Calculate requests per minute
            if self.request_history:
                one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
                recent_requests = sum(
                    1 for req in self.request_history 
                    if req['timestamp'] > one_minute_ago
                )
            else:
                recent_requests = 0
            
            return {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'average_response_time': avg_response_time,
                'requests_per_minute': recent_requests,
                'active_voices': len(self.voice_usage),
                'model_load_time': self.model_load_time,
                'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
                'uptime': time.time() - self.start_time,
                'top_voices': dict(sorted(self.voice_usage.items(), key=lambda x: x[1], reverse=True)[:5]),
                'top_languages': dict(sorted(self.language_usage.items(), key=lambda x: x[1], reverse=True)[:5]),
                'error_counts': dict(self.error_counts)
            }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.request_times.clear()
            self.request_history.clear()
            self.voice_usage.clear()
            self.language_usage.clear()
            self.error_counts.clear()
            self.start_time = time.time()

class HealthMonitor:
    """Monitors system health and resources"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent
                },
                'disk_usage': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'percent': (disk.used / disk.total) * 100
                },
                'network_io': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'process_memory': {
                    'rss_mb': process_memory.rss / (1024**2),
                    'vms_mb': process_memory.vms / (1024**2)
                },
                'uptime': time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                'error': str(e),
                'uptime': time.time() - self.start_time
            }
    
    def get_gpu_health(self) -> Optional[Dict[str, Any]]:
        """Get GPU health metrics if available"""
        try:
            # Try to import torch for GPU info
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = {}
                
                for i in range(gpu_count):
                    gpu_info[f'gpu_{i}'] = {
                        'name': torch.cuda.get_device_name(i),
                        'memory_allocated_mb': torch.cuda.memory_allocated(i) / (1024**2),
                        'memory_reserved_mb': torch.cuda.memory_reserved(i) / (1024**2),
                        'memory_total_mb': torch.cuda.get_device_properties(i).total_memory / (1024**2)
                    }
                
                return gpu_info
            else:
                return None
        except ImportError:
            logger.debug("PyTorch not available for GPU monitoring")
            return None
        except Exception as e:
            logger.error(f"Error getting GPU health: {e}")
            return None

class AuditLogger:
    """Logs sensitive actions for audit purposes"""
    
    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler for audit logs
        import os
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/audit.log")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def log_voice_upload(self, voice_name: str, file_name: str, file_size: int, 
                        client_ip: str, user_agent: Optional[str] = None):
        """Log voice upload action"""
        self.logger.info(
            f"VOICE_UPLOAD: voice_name={voice_name}, file={file_name}, "
            f"size={file_size}, client_ip={client_ip}, user_agent={user_agent}"
        )
    
    def log_voice_delete(self, voice_name: str, client_ip: str, 
                        user_agent: Optional[str] = None):
        """Log voice deletion action"""
        self.logger.info(
            f"VOICE_DELETE: voice_name={voice_name}, client_ip={client_ip}, "
            f"user_agent={user_agent}"
        )
    
    def log_tts_generation(self, voice_name: str, text_length: int, language: str,
                          client_ip: str, user_agent: Optional[str] = None):
        """Log TTS generation action"""
        self.logger.info(
            f"TTS_GENERATION: voice={voice_name}, text_length={text_length}, "
            f"language={language}, client_ip={client_ip}, user_agent={user_agent}"
        )
    
    def log_error(self, error_type: str, error_message: str, client_ip: str,
                  user_agent: Optional[str] = None):
        """Log error occurrences"""
        self.logger.error(
            f"ERROR: type={error_type}, message={error_message}, "
            f"client_ip={client_ip}, user_agent={user_agent}"
        )

# Global instances
metrics_collector = MetricsCollector()
health_monitor = HealthMonitor()
audit_logger = AuditLogger()

def get_metrics() -> Dict[str, Any]:
    """Get current metrics"""
    return metrics_collector.get_metrics()

def get_health() -> Dict[str, Any]:
    """Get current health status"""
    system_health = health_monitor.get_system_health()
    gpu_health = health_monitor.get_gpu_health()
    
    health_data = {
        'system': system_health,
        'gpu': gpu_health
    }
    
    # Determine overall health status
    if system_health.get('error'):
        health_data['status'] = 'unhealthy'
    elif system_health.get('memory_usage', {}).get('percent', 0) > 90:
        health_data['status'] = 'degraded'
    elif system_health.get('cpu_usage_percent', 0) > 90:
        health_data['status'] = 'degraded'
    else:
        health_data['status'] = 'healthy'
    
    return health_data 