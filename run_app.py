#!/usr/bin/env python3
"""
Enhanced AI Stock Analyzer - Startup Script
Run this script to start the application with all features
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import yfinance
        import pandas
        import numpy
        import plotly
        import ta
        import requests
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_ollama():
    """Check if OLLAMA is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ OLLAMA is running and accessible")
            return True
        else:
            print("⚠️ OLLAMA is not responding properly")
            return False
    except Exception:
        print("⚠️ OLLAMA is not running or not accessible at localhost:11434")
        print("Please install and start OLLAMA:")
        print("1. Download from: https://ollama.ai")
        print("2. Install a model: ollama pull llama3.2")
        print("3. Start service: ollama serve")
        return False

def setup_environment():
    """Set up the environment"""
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Create necessary directories
    directories = ["data", "logs", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Copy .env.example to .env if it doesn't exist
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from .env.example")

def main():
    """Main function to start the application"""
    print("🤖 AI Stock Timeframe Analyzer & Scanner")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check OLLAMA (warning only, not blocking)
    ollama_available = check_ollama()
    if not ollama_available:
        print("\n⚠️ Warning: OLLAMA not available - AI analysis will be limited")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    print("\n🚀 Starting AI Stock Analyzer...")
    print("📱 The application will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("\n⌨️ Press Ctrl+C to stop the application")
    print("=" * 50)
    
    # Start the Streamlit application
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/ui/streamlit_app.py",
            "--server.address=0.0.0.0",
            "--server.port=8501",
            "--server.headless=false",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)