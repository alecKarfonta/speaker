#!/bin/bash

# Speaker TTS API Startup Script
# This script activates the virtual environment and starts the API server

echo "🚀 Starting Speaker TTS API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import fastapi, psutil, pydantic" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data/voices

# Validate configuration
echo "⚙️ Validating configuration..."
python -c "from app.config import validate_configuration; print('✅ Configuration valid')"

# Start the API server
echo "🌐 Starting API server on http://localhost:8010"
echo "📚 API Documentation: http://localhost:8010/docs"
echo "💓 Health Check: http://localhost:8010/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python -m app.main 