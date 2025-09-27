import functools
import traceback
import logging
from typing import Any, Callable, Optional, Type, Union
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class StockAnalysisError(Exception):
    """Base exception for stock analysis errors"""
    pass

class DataFetchError(StockAnalysisError):
    """Error fetching stock data"""
    pass

class TechnicalAnalysisError(StockAnalysisError):
    """Error in technical analysis calculations"""
    pass

class AIAnalysisError(StockAnalysisError):
    """Error in AI analysis"""
    pass

class ScannerError(StockAnalysisError):
    """Error in stock scanning operations"""
    pass

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function on failure with exponential backoff
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        break
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator

def handle_errors(
    error_type: Type[Exception] = Exception,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
):
    """
    Decorator to handle errors gracefully
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {type(e).__name__}: {str(e)}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator

def validate_data_frame(df: pd.DataFrame, required_columns: list, min_rows: int = 1) -> bool:
    """
    Validate DataFrame has required structure
    """
    try:
        if df is None or df.empty:
            logger.error("DataFrame is None or empty")
            return False
        
        if len(df) < min_rows:
            logger.error(f"DataFrame has insufficient rows: {len(df)} < {min_rows}")
            return False
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"DataFrame missing required columns: {missing_columns}")
            return False
        
        # Check for NaN values in required columns
        nan_columns = [col for col in required_columns if df[col].isna().any()]
        if nan_columns:
            logger.warning(f"DataFrame has NaN values in columns: {nan_columns}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating DataFrame: {e}")
        return False

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    symbol = symbol.strip().upper()
    
    # Basic validation - alphanumeric characters and hyphens/dots
    if not symbol.replace('-', '').replace('.', '').isalnum():
        return False
    
    # Length check
    if len(symbol) < 1 or len(symbol) > 10:
        return False
    
    return True

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def safe_percentage(value: float, total: float, default: float = 0.0) -> float:
    """
    Safely calculate percentage
    """
    return safe_divide(value * 100, total, default)

class DataValidator:
    """
    Comprehensive data validation utilities
    """
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> tuple[bool, list]:
        """
        Validate OHLC price data
        """
        errors = []
        
        if not validate_data_frame(data, ['open', 'high', 'low', 'close', 'volume']):
            errors.append("Invalid DataFrame structure")
            return False, errors
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] < 0).any():
                errors.append(f"Negative values found in {col}")
        
        # Check OHLC relationships
        if not (data['high'] >= data['low']).all():
            errors.append("High prices less than low prices")
        
        if not (data['high'] >= data['open']).all():
            errors.append("High prices less than open prices")
        
        if not (data['high'] >= data['close']).all():
            errors.append("High prices less than close prices")
        
        if not (data['low'] <= data['open']).all():
            errors.append("Low prices greater than open prices")
        
        if not (data['low'] <= data['close']).all():
            errors.append("Low prices greater than close prices")
        
        # Check for unrealistic price movements
        price_changes = data['close'].pct_change().abs()
        extreme_changes = price_changes > 0.5  # 50% daily change
        if extreme_changes.any():
            errors.append(f"Extreme price changes detected: {extreme_changes.sum()} days")
        
        # Check volume
        if (data['volume'] < 0).any():
            errors.append("Negative volume values")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_technical_indicators(data: pd.DataFrame) -> tuple[bool, list]:
        """
        Validate technical indicators
        """
        errors = []
        
        # RSI should be between 0 and 100
        if 'rsi' in data.columns:
            if (data['rsi'] < 0).any() or (data['rsi'] > 100).any():
                errors.append("RSI values outside 0-100 range")
        
        # Volume ratio should be positive
        if 'volume_ratio' in data.columns:
            if (data['volume_ratio'] < 0).any():
                errors.append("Negative volume ratio values")
        
        # Moving averages should be positive
        ma_columns = [col for col in data.columns if 'ema_' in col or 'sma_' in col]
        for col in ma_columns:
            if (data[col] < 0).any():
                errors.append(f"Negative values in {col}")
        
        return len(errors) == 0, errors

class ErrorReporter:
    """
    Centralized error reporting and metrics
    """
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
    
    def report_error(self, error_type: str, error_message: str, context: dict = None):
        """
        Report an error for tracking
        """
        timestamp = datetime.now()
        
        # Update counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to history
        error_record = {
            'timestamp': timestamp,
            'type': error_type,
            'message': error_message,
            'context': context or {}
        }
        self.error_history.append(error_record)
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Log the error
        logger.error(f"{error_type}: {error_message}", extra={'context': context})
    
    def get_error_summary(self) -> dict:
        """
        Get summary of errors
        """
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def clear_errors(self):
        """
        Clear error history
        """
        self.error_counts.clear()
        self.error_history.clear()

# Global error reporter instance
error_reporter = ErrorReporter()

def log_performance(operation_name: str):
    """
    Decorator to log performance metrics
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = datetime.now() - start_time
                logger.info(f"Performance - {operation_name}: {duration.total_seconds():.2f}s")
                return result
            except Exception as e:
                duration = datetime.now() - start_time
                logger.error(f"Performance - {operation_name} FAILED: {duration.total_seconds():.2f}s - {e}")
                raise
        return wrapper
    return decorator

def circuit_breaker(failure_threshold: int = 5, recovery_time: int = 60):
    """
    Circuit breaker pattern to prevent cascading failures
    """
    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = None
        is_open = False
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, is_open
            
            current_time = datetime.now()
            
            # Check if circuit breaker should be reset
            if is_open and last_failure_time:
                if (current_time - last_failure_time).seconds >= recovery_time:
                    logger.info(f"Circuit breaker reset for {func.__name__}")
                    failure_count = 0
                    is_open = False
            
            # If circuit is open, raise exception
            if is_open:
                raise StockAnalysisError(f"Circuit breaker open for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                # Reset failure count on success
                failure_count = 0
                return result
            except Exception as e:
                failure_count += 1
                last_failure_time = current_time
                
                if failure_count >= failure_threshold:
                    is_open = True
                    logger.error(f"Circuit breaker opened for {func.__name__} after {failure_count} failures")
                
                raise
        
        return wrapper
    return decorator