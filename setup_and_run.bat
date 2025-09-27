@echo off
echo.
echo 🤖 AI Stock Timeframe Analyzer & Scanner
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python is installed

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📥 Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ⚠️ Some packages failed to install, but continuing...
)

REM Copy environment file
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo ✅ Created .env file
    )
)

REM Create directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "models" mkdir models

echo.
echo 🚀 Starting AI Stock Analyzer...
echo 📱 The application will open in your web browser
echo 🌐 URL: http://localhost:8501
echo.
echo 🤖 Note: This app uses Llama 3.2 for enhanced AI analysis
echo 💡 Install OLLAMA and run: ollama pull llama3.2
echo.
echo ⌨️ Press Ctrl+C to stop the application
echo ==========================================

REM Run the application
python run_app.py

pause