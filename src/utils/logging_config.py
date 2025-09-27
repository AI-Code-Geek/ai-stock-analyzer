import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

from config.settings import settings

def setup_logging():
    """
    Configure logging for the application
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log level from settings
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general logs
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "ai_stock_analyzer.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error-only file handler
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Scanner-specific handler
    scanner_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "scanner.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    scanner_handler.setLevel(logging.INFO)
    scanner_handler.setFormatter(detailed_formatter)
    
    # Add scanner handler to scanner loggers
    scanner_logger = logging.getLogger('src.scanner')
    scanner_logger.addHandler(scanner_handler)
    
    # AI analysis handler
    ai_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "ai_analysis.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    ai_handler.setLevel(logging.INFO)
    ai_handler.setFormatter(detailed_formatter)
    
    # Add AI handler to AI loggers
    ai_logger = logging.getLogger('src.ai')
    ai_logger.addHandler(ai_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("AI Stock Analyzer - Logging Initialized")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    logger.info(f"Log Directory: {log_dir.absolute()}")
    logger.info("=" * 50)

class PerformanceLogger:
    """
    Context manager for logging performance metrics
    """
    def __init__(self, operation_name: str, logger: logging.Logger = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = datetime.now() - self.start_time
            if exc_type is not None:
                self.logger.error(f"Operation failed: {self.operation_name} - Duration: {duration} - Error: {exc_val}")
            else:
                self.logger.info(f"Operation completed: {self.operation_name} - Duration: {duration}")

class ErrorHandler:
    """
    Centralized error handling and logging
    """
    @staticmethod
    def log_exception(logger: logging.Logger, operation: str, exception: Exception):
        """
        Log exception with context
        """
        logger.error(f"Error in {operation}: {type(exception).__name__}: {str(exception)}", exc_info=True)
    
    @staticmethod
    def log_warning(logger: logging.Logger, operation: str, message: str):
        """
        Log warning with context
        """
        logger.warning(f"Warning in {operation}: {message}")
    
    @staticmethod
    def log_critical(logger: logging.Logger, operation: str, message: str):
        """
        Log critical error
        """
        logger.critical(f"Critical error in {operation}: {message}")

class ScanLogger:
    """
    Specialized logger for scan operations
    """
    def __init__(self, scan_type: str):
        self.scan_type = scan_type
        self.logger = logging.getLogger(f'src.scanner.{scan_type}')
        self.start_time = None
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
    
    def start_scan(self, total_symbols: int):
        """
        Log scan start
        """
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.scan_type} scan - Total symbols: {total_symbols}")
    
    def log_symbol_processed(self, symbol: str, success: bool, error_msg: str = None):
        """
        Log individual symbol processing
        """
        self.processed_count += 1
        if success:
            self.success_count += 1
            self.logger.debug(f"Successfully processed {symbol}")
        else:
            self.error_count += 1
            self.logger.warning(f"Failed to process {symbol}: {error_msg}")
    
    def end_scan(self):
        """
        Log scan completion
        """
        if self.start_time:
            duration = datetime.now() - self.start_time
            success_rate = (self.success_count / self.processed_count * 100) if self.processed_count > 0 else 0
            
            self.logger.info(f"Scan completed - Duration: {duration}")
            self.logger.info(f"Processed: {self.processed_count}, Success: {self.success_count}, Errors: {self.error_count}")
            self.logger.info(f"Success rate: {success_rate:.1f}%")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name
    """
    return logging.getLogger(name)