import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from dataclasses import dataclass
from enum import Enum

from src.data.yahoo_finance import YahooFinanceClient
from src.data.database import StockDatabase
from src.analysis.technical_indicators import TechnicalIndicators, TrendAnalyzer, PatternRecognition
from src.ai.ollama_client import AIAnalyzer
from config.settings import settings

logger = logging.getLogger(__name__)

class ScannerType(Enum):
    SP500 = "sp500"
    PENNY_STOCKS = "penny_stocks"
    CUSTOM = "custom"

@dataclass
class ScanFilter:
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: Optional[int] = None
    min_ai_score: Optional[float] = None
    trend_filter: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    signal_classification: Optional[str] = None

@dataclass
class ScanResult:
    symbol: str
    current_price: float
    ai_score: float
    signal_classification: str
    trend_alignment: str
    volume_ratio: float
    rsi: float
    daily_trend: str
    weekly_trend: str
    confidence: float
    ai_narrative: str
    scan_timestamp: float

class StockScanner:
    def __init__(self):
        self.yahoo_client = YahooFinanceClient()
        self.database = StockDatabase()
        self.technical_indicators = TechnicalIndicators()
        self.trend_analyzer = TrendAnalyzer()
        self.pattern_recognition = PatternRecognition()
        self.ai_analyzer = AIAnalyzer()
        self.max_workers = settings.MAX_WORKERS
    
    def scan_stocks(self, 
                   scanner_type: ScannerType, 
                   symbols: Optional[List[str]] = None,
                   filters: Optional[ScanFilter] = None,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ScanResult]:
        """
        Main scanning method that processes multiple stocks in parallel
        """
        try:
            # Get symbol list based on scanner type
            if scanner_type == ScannerType.SP500:
                symbol_list = self.yahoo_client.get_sp500_symbols()
            elif scanner_type == ScannerType.PENNY_STOCKS:
                symbol_list = self.yahoo_client.scan_penny_stocks()
            elif scanner_type == ScannerType.CUSTOM and symbols:
                symbol_list = symbols
            else:
                logger.error("Invalid scanner configuration")
                return []
            
            logger.info(f"Starting scan of {len(symbol_list)} symbols using {scanner_type.value}")
            
            results = []
            completed = 0
            
            # Process stocks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(self._analyze_single_stock, symbol): symbol 
                    for symbol in symbol_list
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(completed, len(symbol_list))
                    
                    try:
                        result = future.result()
                        if result and self._passes_filters(result, filters):
                            results.append(result)
                            
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
            
            # Sort results by AI score (descending)
            results.sort(key=lambda x: x.ai_score, reverse=True)
            
            logger.info(f"Scan completed. Found {len(results)} stocks matching criteria")
            return results
            
        except Exception as e:
            logger.error(f"Error during stock scan: {e}")
            return []
    
    def _analyze_single_stock(self, symbol: str) -> Optional[ScanResult]:
        """
        Analyze a single stock and return ScanResult
        """
        try:
            # Get daily data
            daily_data = self.yahoo_client.get_stock_data(symbol, period="1y", interval="1d")
            if daily_data is None or len(daily_data) < 50:
                logger.debug(f"Insufficient daily data for {symbol}")
                return None
            
            # Get weekly data
            weekly_data = self.yahoo_client.get_weekly_data(symbol, period="2y")
            if weekly_data is None or len(weekly_data) < 20:
                logger.debug(f"Insufficient weekly data for {symbol}")
                return None
            
            # Calculate technical indicators
            daily_with_indicators = self.technical_indicators.calculate_all_indicators(daily_data)
            weekly_with_indicators = self.technical_indicators.calculate_all_indicators(weekly_data)
            
            # Analyze trends
            trend_analysis = self.trend_analyzer.analyze_timeframe_alignment(
                daily_with_indicators, weekly_with_indicators
            )
            
            # Detect patterns
            patterns = self.pattern_recognition.detect_candlestick_patterns(daily_with_indicators)
            chart_patterns = self.pattern_recognition.detect_chart_patterns(daily_with_indicators)
            patterns.update(chart_patterns)
            
            # Prepare data for AI analysis
            latest_daily = daily_with_indicators.iloc[-1]
            analysis_data = {
                'current_price': latest_daily['close'],
                'daily_trend': trend_analysis['daily_trend'],
                'weekly_trend': trend_analysis['weekly_trend'],
                'alignment_score': trend_analysis['alignment_score'],
                'rsi': latest_daily.get('rsi', 50),
                'volume_ratio': latest_daily.get('volume_ratio', 1),
                'volatility': latest_daily.get('volatility', 0.2),
                'price_positioning': trend_analysis['daily_trend'].get('price_positioning', {}),
                'macd_signal': self._determine_macd_signal(latest_daily),
                'patterns': patterns
            }
            
            # Get AI analysis (without news for bulk scanning to keep it fast)
            ai_results = self.ai_analyzer.analyze_stock(symbol, analysis_data, None)
            
            # Store results in database
            self.database.store_analysis_result(symbol, {
                **ai_results,
                'timeframe': 'daily',
                **analysis_data
            })
            
            # Create scan result
            return ScanResult(
                symbol=symbol,
                current_price=analysis_data['current_price'],
                ai_score=ai_results['ai_score'],
                signal_classification=ai_results['signal_classification'],
                trend_alignment=trend_analysis['timeframe_sync'],
                volume_ratio=analysis_data['volume_ratio'],
                rsi=analysis_data['rsi'],
                daily_trend=analysis_data['daily_trend']['overall_trend'],
                weekly_trend=analysis_data['weekly_trend']['overall_trend'],
                confidence=ai_results['confidence'],
                ai_narrative=ai_results['ai_narrative'],
                scan_timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _determine_macd_signal(self, data: pd.Series) -> str:
        """
        Determine MACD signal from latest data
        """
        try:
            macd = data.get('macd', 0)
            macd_signal = data.get('macd_signal', 0)
            macd_histogram = data.get('macd_histogram', 0)
            
            if macd > macd_signal and macd_histogram > 0:
                return 'bullish'
            elif macd < macd_signal and macd_histogram < 0:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _passes_filters(self, result: ScanResult, filters: Optional[ScanFilter]) -> bool:
        """
        Check if scan result passes the specified filters
        """
        if not filters:
            return True
        
        try:
            # Price filters
            if filters.min_price and result.current_price < filters.min_price:
                return False
            if filters.max_price and result.current_price > filters.max_price:
                return False
            
            # AI score filter
            if filters.min_ai_score and result.ai_score < filters.min_ai_score:
                return False
            
            # Trend filter
            if filters.trend_filter:
                if filters.trend_filter == 'bullish' and result.daily_trend != 'bullish':
                    return False
                elif filters.trend_filter == 'bearish' and result.daily_trend != 'bearish':
                    return False
                elif filters.trend_filter == 'neutral' and result.daily_trend != 'neutral':
                    return False
            
            # RSI filters
            if filters.rsi_min and result.rsi < filters.rsi_min:
                return False
            if filters.rsi_max and result.rsi > filters.rsi_max:
                return False
            
            # Signal classification filter
            if filters.signal_classification and result.signal_classification != filters.signal_classification:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return False

class ScannerManager:
    """
    High-level interface for managing different types of scans
    """
    def __init__(self):
        self.scanner = StockScanner()
        self.active_scans = {}
    
    def start_sp500_scan(self, filters: Optional[ScanFilter] = None, 
                        progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ScanResult]:
        """
        Start S&P 500 scan
        """
        return self.scanner.scan_stocks(ScannerType.SP500, filters=filters, progress_callback=progress_callback)
    
    def start_penny_stock_scan(self, filters: Optional[ScanFilter] = None,
                              progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ScanResult]:
        """
        Start penny stock scan
        """
        return self.scanner.scan_stocks(ScannerType.PENNY_STOCKS, filters=filters, progress_callback=progress_callback)
    
    def start_custom_scan(self, symbols: List[str], filters: Optional[ScanFilter] = None,
                         progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ScanResult]:
        """
        Start custom symbol scan
        """
        return self.scanner.scan_stocks(ScannerType.CUSTOM, symbols=symbols, filters=filters, progress_callback=progress_callback)
    
    def create_default_filters(self) -> ScanFilter:
        """
        Create default filter set
        """
        return ScanFilter(
            min_ai_score=60.0,
            min_volume=100000,
            rsi_min=30,
            rsi_max=70
        )
    
    def create_bullish_filters(self) -> ScanFilter:
        """
        Create filters for bullish signals
        """
        return ScanFilter(
            min_ai_score=70.0,
            trend_filter='bullish',
            rsi_min=40,
            rsi_max=70
        )
    
    def create_oversold_filters(self) -> ScanFilter:
        """
        Create filters for oversold conditions
        """
        return ScanFilter(
            min_ai_score=50.0,
            rsi_min=20,
            rsi_max=35
        )

class ScanResultExporter:
    """
    Export scan results to various formats
    """
    @staticmethod
    def to_dataframe(results: List[ScanResult]) -> pd.DataFrame:
        """
        Convert scan results to pandas DataFrame
        """
        data = []
        for result in results:
            data.append({
                'Symbol': result.symbol,
                'Price': result.current_price,
                'AI Score': result.ai_score,
                'Signal': result.signal_classification,
                'Trend Alignment': result.trend_alignment,
                'Volume Ratio': result.volume_ratio,
                'RSI': result.rsi,
                'Daily Trend': result.daily_trend,
                'Weekly Trend': result.weekly_trend,
                'Confidence': result.confidence,
                'Timestamp': pd.to_datetime(result.scan_timestamp, unit='s')
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def to_csv(results: List[ScanResult], filename: str) -> bool:
        """
        Export results to CSV file
        """
        try:
            df = ScanResultExporter.to_dataframe(results)
            df.to_csv(filename, index=False)
            logger.info(f"Exported {len(results)} results to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    @staticmethod
    def to_json(results: List[ScanResult], filename: str) -> bool:
        """
        Export results to JSON file
        """
        try:
            data = []
            for result in results:
                data.append({
                    'symbol': result.symbol,
                    'current_price': result.current_price,
                    'ai_score': result.ai_score,
                    'signal_classification': result.signal_classification,
                    'trend_alignment': result.trend_alignment,
                    'volume_ratio': result.volume_ratio,
                    'rsi': result.rsi,
                    'daily_trend': result.daily_trend,
                    'weekly_trend': result.weekly_trend,
                    'confidence': result.confidence,
                    'ai_narrative': result.ai_narrative,
                    'scan_timestamp': result.scan_timestamp
                })
            
            import json
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(results)} results to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False