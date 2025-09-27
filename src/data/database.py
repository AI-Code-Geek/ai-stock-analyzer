import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StockDatabase:
    def __init__(self, db_path: str = "data/stocks.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    timeframe TEXT DEFAULT 'daily',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date, timeframe)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_info (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    price REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    analysis_date DATE NOT NULL,
                    timeframe TEXT NOT NULL,
                    ai_score REAL,
                    trend_alignment REAL,
                    volume_confirmation REAL,
                    rsi_positioning REAL,
                    price_action REAL,
                    volatility_analysis REAL,
                    momentum_quality REAL,
                    signal_classification TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, analysis_date, timeframe)
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_data(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_symbol ON analysis_results(symbol)")
    
    def store_stock_data(self, symbol: str, data: pd.DataFrame, timeframe: str = 'daily'):
        try:
            data_to_store = data.copy()
            data_to_store['symbol'] = symbol
            data_to_store['timeframe'] = timeframe
            data_to_store.reset_index(inplace=True)
            
            with sqlite3.connect(self.db_path) as conn:
                data_to_store.to_sql('stock_data', conn, if_exists='append', index=False,
                                   method='ignore')
            logger.info(f"Stored {len(data_to_store)} records for {symbol} ({timeframe})")
        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")
    
    def get_stock_data(self, symbol: str, days: int = 365, timeframe: str = 'daily') -> Optional[pd.DataFrame]:
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT date, open, high, low, close, volume
                    FROM stock_data
                    WHERE symbol = ? AND timeframe = ? AND date >= ?
                    ORDER BY date
                """
                df = pd.read_sql_query(query, conn, params=(symbol, timeframe, cutoff_date.date()))
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    return df
                
            return None
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return None
    
    def store_stock_info(self, symbol: str, info: Dict[str, Any]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO stock_info 
                    (symbol, company_name, sector, industry, market_cap, price, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    symbol,
                    info.get('longName', ''),
                    info.get('sector', ''),
                    info.get('industry', ''),
                    info.get('marketCap', 0),
                    info.get('currentPrice', 0)
                ))
        except Exception as e:
            logger.error(f"Error storing info for {symbol}: {e}")
    
    def store_analysis_result(self, symbol: str, analysis: Dict[str, Any]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO analysis_results
                    (symbol, analysis_date, timeframe, ai_score, trend_alignment, 
                     volume_confirmation, rsi_positioning, price_action, 
                     volatility_analysis, momentum_quality, signal_classification)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    datetime.now().date(),
                    analysis.get('timeframe', 'daily'),
                    analysis.get('ai_score', 0),
                    analysis.get('trend_alignment', 0),
                    analysis.get('volume_confirmation', 0),
                    analysis.get('rsi_positioning', 0),
                    analysis.get('price_action', 0),
                    analysis.get('volatility_analysis', 0),
                    analysis.get('momentum_quality', 0),
                    analysis.get('signal_classification', 'Unknown')
                ))
        except Exception as e:
            logger.error(f"Error storing analysis for {symbol}: {e}")
    
    def get_cached_symbols(self, timeframe: str = 'daily', min_age_hours: int = 24) -> List[str]:
        try:
            cutoff_time = datetime.now() - timedelta(hours=min_age_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT symbol 
                    FROM stock_data 
                    WHERE timeframe = ? AND created_at >= ?
                """, (timeframe, cutoff_time))
                
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting cached symbols: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 730):
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM stock_data 
                    WHERE date < ?
                """, (cutoff_date.date(),))
                
                deleted_count = cursor.rowcount
                logger.info(f"Cleaned up {deleted_count} old records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")