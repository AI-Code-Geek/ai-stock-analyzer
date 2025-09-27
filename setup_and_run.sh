#!/bin/bash

echo ""
echo "🤖 AI Stock Timeframe Analyzer & Scanner"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

echo "✅ Python is installed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "⚠️ Some packages failed to install, but continuing..."
fi

# Copy environment file
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp ".env.example" ".env"
    echo "✅ Created .env file"
fi

# Create directories
mkdir -p data logs models

echo ""
echo "🚀 Starting AI Stock Analyzer..."
echo "📱 The application will open in your web browser"
echo "🌐 URL: http://localhost:8501"
echo ""
echo "🤖 Note: This app uses Llama 3.2 for enhanced AI analysis"
echo "💡 Install OLLAMA and run: ollama pull llama3.2"
echo ""
echo "⌨️ Press Ctrl+C to stop the application"
echo "=========================================="

# Run the application
python3 run_app.py