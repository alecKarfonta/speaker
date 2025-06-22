#!/usr/bin/env python3
"""
Test script to demonstrate the monitoring features of the TTS API.
This script makes various TTS requests to generate metrics for monitoring.
"""

import requests
import time
import random
import json
from typing import List, Dict, Optional

# API configuration
API_BASE_URL = "http://localhost:8010"

# Test texts with different word counts
TEST_TEXTS = [
    # Short texts (1-10 words)
    "Hello world!",
    "How are you today?",
    "This is a test.",
    "The quick brown fox.",
    "Welcome to our API.",
    
    # Medium texts (11-25 words)
    "This is a medium length text that contains more words to test the performance monitoring system.",
    "The weather today is quite nice and I think we should go for a walk in the park.",
    "Artificial intelligence has made significant progress in recent years.",
    "The restaurant serves delicious food and has excellent service.",
    "Learning a new programming language can be challenging but rewarding.",
    
    # Longer texts (26-50 words)
    "This is a longer text that contains many more words to thoroughly test the word-based performance monitoring system that we have implemented in our TTS API. It should provide good data for analysis.",
    "The development team worked tirelessly to implement comprehensive monitoring features including word count analysis, character count tracking, and voice-specific performance metrics.",
    "Natural language processing has evolved significantly over the past decade with the introduction of transformer models and advanced neural network architectures.",
    "The conference featured presentations from leading experts in machine learning, artificial intelligence, and data science.",
    "Environmental conservation efforts require collaboration between governments, businesses, and individuals to achieve meaningful results.",
    
    # Very long texts (50+ words)
    "This is a very long text designed to test the performance monitoring system with a substantial amount of content. It contains many words and should demonstrate how the system handles longer requests. The monitoring system should track response times based on word count and provide insights into performance characteristics.",
    "The comprehensive monitoring system we have implemented includes intelligent word counting that handles emotion tags, tracks performance by voice and language, and provides detailed metrics for analysis. This enables us to optimize performance and identify bottlenecks in the TTS generation process.",
]

# Test voices
VOICES = ["biden", "trump", "major", "dsp"]

# Test languages
LANGUAGES = ["en", "es", "fr", "de"]

# Test emotions
EMOTIONS = ["happy", "sad", "excited", "calm", None]

def make_tts_request(text: str, voice: str, language: str = "en", emotion: Optional[str] = None) -> Dict:
    """Make a TTS request and return response info."""
    payload = {
        "text": text,
        "voice_name": voice,
        "language": language,
        "temperature": 0.7,
        "top_p": 0.85,
        "speed": 1.0
    }
    
    if emotion:
        payload["emotion"] = emotion
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_BASE_URL}/tts", json=payload)
        end_time = time.time()
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time": end_time - start_time,
            "text_length": len(text),
            "word_count": len(text.split()),
            "voice": voice,
            "language": language,
            "emotion": emotion,
            "audio_size": len(response.content) if response.status_code == 200 else 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response_time": time.time() - start_time,
            "text_length": len(text),
            "word_count": len(text.split()),
            "voice": voice,
            "language": language,
            "emotion": emotion
        }

def get_metrics() -> Dict:
    """Get current API metrics."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_prometheus_metrics() -> str:
    """Get Prometheus metrics."""
    try:
        response = requests.get(f"{API_BASE_URL}/prometheus")
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def run_performance_test(num_requests: int = 20):
    """Run a performance test with various request types."""
    print(f"🚀 Starting performance test with {num_requests} requests...")
    print("=" * 60)
    
    results = []
    
    for i in range(num_requests):
        # Randomly select test parameters
        text = random.choice(TEST_TEXTS)
        voice = random.choice(VOICES)
        language = random.choice(LANGUAGES)
        emotion = random.choice(EMOTIONS)
        
        print(f"Request {i+1}/{num_requests}: {len(text.split())} words, {voice} voice, {language} language")
        
        result = make_tts_request(text, voice, language, emotion)
        results.append(result)
        
        if result["success"]:
            print(f"  ✅ Success: {result['response_time']:.2f}s")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    return results

def analyze_results(results: List[Dict]):
    """Analyze and display test results."""
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful_requests)}")
    print(f"Failed: {len(failed_requests)}")
    print(f"Success rate: {len(successful_requests)/len(results)*100:.1f}%")
    
    if successful_requests:
        response_times = [r["response_time"] for r in successful_requests]
        print(f"\nResponse Time Statistics:")
        print(f"  Average: {sum(response_times)/len(response_times):.2f}s")
        print(f"  Min: {min(response_times):.2f}s")
        print(f"  Max: {max(response_times):.2f}s")
        
        # Word count analysis
        word_counts = [r["word_count"] for r in successful_requests]
        print(f"\nWord Count Statistics:")
        print(f"  Average: {sum(word_counts)/len(word_counts):.1f} words")
        print(f"  Min: {min(word_counts)} words")
        print(f"  Max: {max(word_counts)} words")
        
        # Voice performance
        voice_performance = {}
        for r in successful_requests:
            voice = r["voice"]
            if voice not in voice_performance:
                voice_performance[voice] = []
            voice_performance[voice].append(r["response_time"])
        
        print(f"\nVoice Performance:")
        for voice, times in voice_performance.items():
            avg_time = sum(times) / len(times)
            print(f"  {voice}: {avg_time:.2f}s average ({len(times)} requests)")

def main():
    """Main function to run the monitoring test."""
    print("🎯 TTS API Monitoring Test")
    print("=" * 60)
    
    # Check if API is running
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ API is healthy: {health_data['status']}")
            print(f"   Model: {health_data['model']}")
            print(f"   Available voices: {health_data['available_voices']}")
        else:
            print("❌ API health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Run performance test
    results = run_performance_test(num_requests=15)
    
    # Analyze results
    analyze_results(results)
    
    # Show metrics
    print("\n" + "=" * 60)
    print("📈 CURRENT METRICS")
    print("=" * 60)
    
    metrics = get_metrics()
    if "error" not in metrics:
        print(f"Total requests: {metrics.get('total_requests', 'N/A')}")
        print(f"Average response time: {metrics.get('average_response_time', 'N/A')}")
        print(f"Requests per minute: {metrics.get('requests_per_minute', 'N/A')}")
        
        if 'word_performance' in metrics:
            print(f"\nWord Performance:")
            for word_range, perf in metrics['word_performance'].items():
                print(f"  {word_range} words: {perf['avg_time']:.2f}s avg ({perf['count']} requests)")
    
    print("\n" + "=" * 60)
    print("🔗 MONITORING LINKS")
    print("=" * 60)
    print(f"API Metrics: {API_BASE_URL}/metrics")
    print(f"Prometheus: {API_BASE_URL}/prometheus")
    print(f"Health Check: {API_BASE_URL}/health")
    print(f"API Documentation: {API_BASE_URL}/docs")
    print(f"Frontend: http://localhost:3010")
    print(f"Prometheus UI: http://localhost:9199")
    print(f"Grafana Dashboard: http://localhost:3333 (admin/admin123)")
    
    print("\n🎉 Test completed! Check the monitoring dashboards for detailed analysis.")

if __name__ == "__main__":
    main() 