# 🤖 AI Stock Analyzer

A sophisticated AI-powered stock analysis platform that leverages local OLLAMA models for intelligent market analysis across daily and weekly timeframes.

## 🎯 Features

### 🔍 **Advanced Stock Scanning**
- **Multi-Type Scanners**: S&P 500, penny stocks, custom symbols, and mixed analysis
- **Parallel Processing**: High-performance concurrent scanning with live progress
- **Advanced Filtering**: AI score, price range, trends, RSI, volume, volatility filters
- **Quick Presets**: Pre-configured filters for growth, value, and momentum strategies

### 🤖 **AI-Powered Analysis (Llama 3.2)**
- **Intelligent Scoring**: 0-100 AI scoring system using local OLLAMA Llama 3.2 model
- **Enhanced Analysis**: Improved reasoning and context understanding with Llama 3.2
- **Multi-Timeframe Analysis**: Daily and weekly trend alignment detection
- **Component Scoring**: Detailed breakdown of trend, volume, RSI, price action, volatility, and momentum
- **Signal Generation**: Automated buy/sell/hold signals with confidence levels
- **Pattern Recognition**: Candlestick patterns, chart patterns, and trend analysis

### 📊 **Enhanced Technical Analysis**
- **Complete Indicator Suite**: EMAs (20/50/200), RSI, MACD, volume analysis, ATR, VWAP
- **Support & Resistance**: Automated level identification with pivot point analysis
- **Demand & Supply Zones**: Volume-based zone detection for institutional levels
- **Volatility Analysis**: Risk assessment and position sizing recommendations

### 🎨 **Interactive Visualization**
- **Advanced Charts**: Multi-panel candlestick charts with overlays and zones
- **Real-time Updates**: Live scanning progress with immediate result updates
- **Custom Styling**: Beautiful CSS styling with trend-based color coding
- **Responsive Design**: Mobile-friendly interface with adaptive layouts

### 📈 **Comprehensive Results Display**
- **Smart Categorization**: Strong buys, value plays, momentum opportunities
- **Interactive Tables**: Sortable, filterable results with detailed metrics
- **Live Analytics**: Real-time statistics and distribution charts
- **Export Options**: CSV, JSON, and summary reports with metadata

### 🔧 **Professional Features**
- **Data Persistence**: SQLite database for caching and historical analysis
- **Error Handling**: Robust error management with circuit breakers and retries
- **Performance Monitoring**: Detailed logging and performance metrics
- **Docker Support**: Full containerization with OLLAMA integration

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OLLAMA installed and running locally
- At least 8GB RAM (16GB recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-stock-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Install and start OLLAMA**
   ```bash
   # Install OLLAMA (see https://ollama.ai)
   ollama pull llama3.2  # Latest Llama 3.2 model
   ollama serve
   ```

### Running the Application

#### Easy Startup (Recommended)
```bash
# Windows
setup_and_run.bat

# Linux/macOS  
chmod +x setup_and_run.sh
./setup_and_run.sh

# Or using Python directly
python run_app.py
```

#### Manual Startup
```bash
python main.py web
# or
streamlit run src/ui/streamlit_app.py
```

Open your browser to `http://localhost:8501`

#### Command Line Interface
```bash
# Check dependencies
python main.py check

# Run S&P 500 scan
python main.py scan sp500 --min-score 70 --export

# Run penny stock scan
python main.py scan penny --min-score 60 --max-price 5

# Custom symbol scan
python main.py scan custom --symbols "AAPL,MSFT,GOOGL" --min-score 75
```

### Docker Deployment

```bash
# Start with Docker Compose (includes OLLAMA)
docker-compose up -d

# Or build and run manually
docker build -t ai-stock-analyzer .
docker run -p 8501:8501 ai-stock-analyzer
```

## 🧠 Why Llama 3.2?

The application now uses **Llama 3.2**, Meta's latest and most capable open-source language model, providing:

- **🎯 Enhanced Accuracy**: Better understanding of financial context and technical analysis
- **⚡ Improved Speed**: Faster response times with optimized inference
- **🧠 Better Reasoning**: More sophisticated analysis of market patterns and trends
- **📊 Structured Output**: Cleaner, more actionable trading insights
- **🔒 Privacy**: All AI processing happens locally - no data sent to external services

### Model Comparison
| Feature | Llama 2 | **Llama 3.2** |
|---------|---------|---------------|
| Parameters | 7B-70B | 1B-90B |
| Context Length | 4K tokens | 128K tokens |
| Financial Analysis | Good | **Excellent** |
| Speed | Moderate | **Fast** |
| Accuracy | High | **Very High** |

## 📊 How It Works

### AI Scoring Algorithm (0-100 Scale)

The AI scoring system evaluates stocks using weighted factors:

- **Trend Alignment (30%)**: Weekly-daily trend synchronization
- **Volume Confirmation (20%)**: Above-average volume analysis
- **RSI Positioning (15%)**: Optimal RSI placement for trend direction
- **Price Action (15%)**: Price vs moving average relationships
- **Volatility Analysis (10%)**: Optimal volatility range assessment
- **Momentum Quality (10%)**: Recent price momentum evaluation

### Signal Classifications

- **90-100**: Very Strong - High conviction signals
- **75-89**: Strong - Good trading opportunities
- **60-74**: Moderate - Proceed with caution
- **45-59**: Weak - Consider avoiding
- **0-44**: Very Weak - High risk signals

## 🖥️ User Interface

### Dashboard
- Real-time scan results overview
- Top performers visualization
- Score distribution analysis
- Key metrics summary

### Stock Scanner
- Multiple scanner types (S&P 500, Penny Stocks, Custom)
- Advanced filtering options
- Real-time progress tracking
- Export capabilities

### Individual Analysis
- Detailed stock charts with technical indicators
- AI narrative analysis
- Component score breakdowns
- Interactive visualizations

### Settings
- OLLAMA configuration
- Scanning parameters
- Technical analysis settings
- Performance tuning

## 🛠️ Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=sqlite:///./data/stocks.db

# OLLAMA Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2:7b

# Performance
MAX_WORKERS=10
YAHOO_RATE_LIMIT=0.5
CACHE_TTL_MINUTES=60

# Logging
LOG_LEVEL=INFO
```

### Technical Analysis Settings

Configure in `config/settings.py`:
- EMA periods: [20, 50, 200]
- RSI period: 14
- MACD parameters: 12, 26, 9
- Volume analysis windows
- Scoring weights

## 📈 Technical Indicators

### Trend Analysis
- **EMAs**: 20, 50, 200 period exponential moving averages
- **Trend Alignment**: Cross-timeframe trend synchronization
- **Support/Resistance**: Automated level identification

### Momentum Indicators
- **RSI**: 14-period Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Price Action**: Price vs moving average positioning

### Volume Analysis
- **Volume Ratio**: Current vs average volume
- **VWAP**: Volume Weighted Average Price
- **Volume Confirmation**: Pattern validation

### Volatility Measures
- **ATR**: Average True Range
- **Volatility**: Annualized price volatility
- **Risk Assessment**: Position sizing recommendations

## 🔍 Stock Scanners

### S&P 500 Scanner
- Scans all S&P 500 constituents
- High-quality large-cap stocks
- Suitable for conservative strategies

### Penny Stock Scanner
- Filters stocks under $5
- Higher risk/reward potential
- Increased volatility considerations

### Custom Scanner
- User-defined symbol lists
- Flexible for specific strategies
- Portfolio analysis capabilities

## 📊 Data Sources

- **Yahoo Finance**: Primary data source for OHLC and volume
- **Wikipedia**: S&P 500 constituent list
- **Local Database**: Caching and historical storage
- **OLLAMA**: Local AI model for analysis

## 🚨 Error Handling

### Robust Error Management
- Comprehensive exception handling
- Graceful degradation on failures
- Retry mechanisms with exponential backoff
- Circuit breaker patterns

### Data Validation
- OHLC data integrity checks
- Symbol validation
- Technical indicator bounds checking
- Anomaly detection

### Logging
- Structured logging with multiple levels
- Separate log files for different components
- Performance monitoring
- Error tracking and reporting

## 🔧 Development

### Project Structure
```
ai-stock-analyzer/
├── src/
│   ├── data/              # Data acquisition & processing
│   ├── analysis/          # Technical analysis modules
│   ├── ai/               # OLLAMA integration & AI logic
│   ├── scanner/          # Stock scanning engines
│   ├── ui/               # Streamlit interface
│   └── utils/            # Helper functions
├── config/               # Configuration files
├── data/                 # Cache & databases
├── logs/                 # Application logs
├── tests/                # Unit & integration tests
└── docs/                 # Documentation
```

### Adding New Features

1. **Technical Indicators**: Add to `src/analysis/technical_indicators.py`
2. **AI Models**: Extend `src/ai/ollama_client.py`
3. **Scanners**: Modify `src/scanner/stock_scanner.py`
4. **UI Components**: Update `src/ui/streamlit_app.py`

### Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Integration tests
pytest tests/integration/
```

## 🔒 Security & Compliance

### Risk Disclaimers
- **Educational Purpose**: This tool is for educational and research purposes only
- **Not Financial Advice**: Results should not be considered as investment advice
- **Risk Warning**: All investments carry risk of loss
- **Backtesting Required**: Thoroughly test strategies before live trading

### Data Privacy
- **Local Processing**: All AI analysis runs locally via OLLAMA
- **No External APIs**: No data sent to external AI services
- **Configurable Logging**: Control what data is logged

## 🎯 Performance Targets

- **Single Stock Analysis**: < 2 seconds
- **S&P 500 Scan**: < 5 minutes
- **UI Response Time**: < 1 second for most operations
- **AI Analysis**: < 10 seconds per stock

## 🔮 Future Enhancements

### Short-term (3-6 months)
- Additional timeframes (4-hour, monthly)
- More technical indicators (Bollinger Bands, Fibonacci)
- Portfolio analysis capabilities
- Mobile responsive design

### Long-term (6-12 months)
- Custom ML model training
- Options analysis integration
- Fundamental analysis incorporation
- Social sentiment analysis
- Paper trading simulation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OLLAMA Team**: For the excellent local LLM platform
- **Yahoo Finance**: For providing free market data
- **Streamlit**: For the amazing web app framework
- **TA-Lib**: For technical analysis calculations

## 📞 Support

- **Documentation**: Check the `/docs` directory
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Email**: contact@example.com

---

**⚠️ Disclaimer**: This software is for educational purposes only. Past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.