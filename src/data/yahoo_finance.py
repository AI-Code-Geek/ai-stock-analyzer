import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

class YahooFinanceClient:
    def __init__(self):
        self.rate_limit = settings.YAHOO_RATE_LIMIT
        self.last_request_time = 0
    
    def _rate_limit_request(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_stock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
        try:
            self._rate_limit_request()
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Remove timezone info and ensure datetime index
            if hasattr(data.index, 'tz'):
                data.index = data.index.tz_localize(None)
            
            logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            self._rate_limit_request()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                logger.warning(f"No info found for {symbol}")
                return None
            
            return info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return None
    
    def get_weekly_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        try:
            daily_data = self.get_stock_data(symbol, period, "1d")
            if daily_data is None:
                return None
            
            # Resample to weekly data (Friday close)
            weekly_data = daily_data.resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return weekly_data
            
        except Exception as e:
            logger.error(f"Error creating weekly data for {symbol}: {e}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        try:
            self._rate_limit_request()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if symbol exists and has basic data
            return bool(info and 'symbol' in info and info.get('regularMarketPrice'))
            
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def get_sp500_symbols(self) -> List[str]:
        try:
            # Get S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            
            symbols = sp500_table['Symbol'].tolist()
            # Clean symbols (remove dots, etc.)
            symbols = [symbol.replace('.', '-') for symbol in symbols]
            
            logger.info(f"Retrieved {len(symbols)} S&P 500 symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {e}")
            # Fallback to a smaller list of major stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
                'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
                'ABBV', 'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS',
                'MRK', 'ABT', 'VZ', 'ADBE', 'NFLX', 'CMCSA', 'CRM', 'ACN', 'NKE'
            ]
    
    def scan_penny_stocks(self, max_price: float = 5.0, min_volume: int = 100000) -> List[str]:
        try:
            # This is a simplified approach - in production, you'd want a more comprehensive screener
            potential_symbols = []
            
            # Sample of penny stock symbols to check
            penny_candidates = [
                'SIRI', 'NOK', 'GEVO', 'ZNGA', 'PLUG', 'FCEL', 'IDEX', 'TOPS',
                'SHIP', 'EXPR', 'AMC', 'SNDL', 'NAKD', 'CTRM', 'DLPN', 'INPX'
            ]
            
            for symbol in penny_candidates:
                try:
                    info = self.get_stock_info(symbol)
                    if info and info.get('regularMarketPrice', 0) <= max_price:
                        if info.get('averageVolume', 0) >= min_volume:
                            potential_symbols.append(symbol)
                except Exception as e:
                    logger.debug(f"Error checking penny stock {symbol}: {e}")
                    continue
            
            logger.info(f"Found {len(potential_symbols)} penny stocks under ${max_price}")
            return potential_symbols
            
        except Exception as e:
            logger.error(f"Error scanning penny stocks: {e}")
            return []
    
    def get_multiple_stocks_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        results = {}
        
        def fetch_stock_data(symbol):
            try:
                data = self.get_stock_data(symbol, period)
                return symbol, data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, None
        
        # Use ThreadPoolExecutor for parallel requests with rate limiting
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            future_to_symbol = {
                executor.submit(fetch_stock_data, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    results[symbol] = data
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results

class DataValidator:
    @staticmethod
    def validate_ohlc_data(data: pd.DataFrame) -> bool:
        if data.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for negative values
        if (data[['open', 'high', 'low', 'close']] < 0).any().any():
            return False
        
        # Check high >= low, open, close
        if not ((data['high'] >= data['low']) & 
                (data['high'] >= data['open']) & 
                (data['high'] >= data['close'])).all():
            return False
        
        # Check low <= open, close
        if not ((data['low'] <= data['open']) & 
                (data['low'] <= data['close'])).all():
            return False
        
        return True
    
    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        # Remove rows with NaN values
        data = data.dropna()
        
        # Remove duplicate dates
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    @staticmethod
    def detect_anomalies(data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        # Detect price anomalies using z-score
        price_changes = data['close'].pct_change()
        z_scores = np.abs((price_changes - price_changes.mean()) / price_changes.std())
        
        # Mark anomalous days
        data['anomaly'] = z_scores > threshold
        
        return data