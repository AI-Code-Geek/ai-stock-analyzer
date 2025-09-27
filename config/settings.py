import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/stocks.db")
        self.CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "60"))
        self.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))
        self.OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.YAHOO_RATE_LIMIT = float(os.getenv("YAHOO_RATE_LIMIT", "0.5"))
        
        # Technical Analysis Settings
        self.EMA_PERIODS = [20, 50, 200]
        self.RSI_PERIOD = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # AI Scoring Weights
        self.SCORING_WEIGHTS = {
            "trend_alignment": 0.30,
            "volume_confirmation": 0.20,
            "rsi_positioning": 0.15,
            "price_action": 0.15,
            "volatility_analysis": 0.10,
            "momentum_quality": 0.10
        }
        
        # Score Classification (fixed gaps for continuous coverage)
        self.SCORE_RANGES = {
            "very_strong": (90, 100),
            "strong": (75, 89.99),
            "moderate": (60, 74.99),
            "weak": (45, 59.99),
            "very_weak": (0, 44.99)
        }
        
        # Scanner Settings
        self.PENNY_STOCK_THRESHOLD = 5.0
        self.MIN_VOLUME = 100000
        self.MIN_MARKET_CAP = 1000000
        
    def get_database_config(self) -> Dict[str, Any]:
        return {
            "url": self.DATABASE_URL,
            "echo": self.LOG_LEVEL == "DEBUG"
        }
    
    def get_ollama_config(self) -> Dict[str, Any]:
        return {
            "host": self.OLLAMA_HOST,
            "model": self.OLLAMA_MODEL
        }

settings = Settings()