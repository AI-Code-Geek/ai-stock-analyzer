"""
AI Stock Analyzer - Main Application Entry Point
Daily & Weekly Timeframe Analysis with OLLAMA Integration
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set up logging first
from src.utils.logging_config import setup_logging
setup_logging()

import logging
import argparse
from typing import Optional

from src.scanner.stock_scanner import ScannerManager, ScanFilter, ScannerType
from src.utils.error_handlers import ErrorReporter, error_reporter
from config.settings import settings

logger = logging.getLogger(__name__)

def run_streamlit_app():
    """
    Run the Streamlit web application
    """
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set up Streamlit arguments
        sys.argv = [
            "streamlit",
            "run",
            "src/ui/streamlit_app.py",
            "--server.address=0.0.0.0",
            "--server.port=8501"
        ]
        
        logger.info("Starting Streamlit application...")
        stcli.main()
        
    except Exception as e:
        logger.error(f"Failed to start Streamlit app: {e}")
        raise

def run_cli_scanner(args):
    """
    Run stock scanner from command line
    """
    try:
        logger.info("Starting CLI stock scanner...")
        
        scanner_manager = ScannerManager()
        
        # Create filters
        filters = ScanFilter(
            min_ai_score=args.min_score,
            min_price=args.min_price,
            max_price=args.max_price,
            trend_filter=args.trend,
            rsi_min=args.rsi_min,
            rsi_max=args.rsi_max
        )
        
        # Determine scanner type
        if args.scanner_type == "sp500":
            scanner_type = ScannerType.SP500
            symbols = None
        elif args.scanner_type == "penny":
            scanner_type = ScannerType.PENNY_STOCKS
            symbols = None
        else:  # custom
            scanner_type = ScannerType.CUSTOM
            symbols = args.symbols.split(",") if args.symbols else []
        
        # Progress callback
        def progress_callback(completed, total):
            percentage = (completed / total) * 100
            print(f"\rProgress: {completed}/{total} ({percentage:.1f}%)", end="", flush=True)
        
        # Run scan
        print(f"Starting {args.scanner_type} scan...")
        results = scanner_manager.scanner.scan_stocks(
            scanner_type, 
            symbols=symbols, 
            filters=filters, 
            progress_callback=progress_callback
        )
        
        print(f"\n\nScan completed! Found {len(results)} stocks matching criteria.")
        
        # Display top results
        if results:
            print("\nTop 10 Results:")
            print("-" * 80)
            print(f"{'Symbol':<8} {'Price':<10} {'AI Score':<10} {'Signal':<12} {'Trend':<10}")
            print("-" * 80)
            
            for result in results[:10]:
                print(f"{result.symbol:<8} ${result.current_price:<9.2f} {result.ai_score:<10.1f} "
                      f"{result.signal_classification:<12} {result.daily_trend:<10}")
        
        # Export results if requested
        if args.export:
            from src.scanner.stock_scanner import ScanResultExporter
            filename = f"scan_results_{args.scanner_type}.csv"
            if ScanResultExporter.to_csv(results, filename):
                print(f"\nResults exported to {filename}")
        
    except Exception as e:
        logger.error(f"CLI scanner failed: {e}")
        raise

def check_dependencies():
    """
    Check if all required dependencies are available
    """
    try:
        logger.info("Checking dependencies...")
        
        # Check OLLAMA connection
        from src.ai.ollama_client import OllamaClient
        ollama_client = OllamaClient()
        
        if ollama_client.check_connection():
            logger.info("✅ OLLAMA connection successful")
            models = ollama_client.list_models()
            logger.info(f"Available models: {models}")
        else:
            logger.warning("⚠️ OLLAMA not available - AI analysis will be limited")
        
        # Check Yahoo Finance
        from src.data.yahoo_finance import YahooFinanceClient
        yahoo_client = YahooFinanceClient()
        
        test_data = yahoo_client.get_stock_data("AAPL", period="5d")
        if test_data is not None:
            logger.info("✅ Yahoo Finance connection successful")
        else:
            logger.warning("⚠️ Yahoo Finance connection issues")
        
        # Check database
        from src.data.database import StockDatabase
        db = StockDatabase()
        logger.info("✅ Database initialized")
        
        logger.info("Dependency check completed")
        return True
        
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False

def main():
    """
    Main application entry point
    """
    parser = argparse.ArgumentParser(
        description="AI Stock Analyzer - Daily & Weekly Timeframe Analysis"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Web app command
    web_parser = subparsers.add_parser("web", help="Start web application")
    
    # CLI scanner command
    scan_parser = subparsers.add_parser("scan", help="Run stock scanner")
    scan_parser.add_argument("scanner_type", choices=["sp500", "penny", "custom"], 
                           help="Type of scanner to run")
    scan_parser.add_argument("--symbols", help="Comma-separated symbols for custom scan")
    scan_parser.add_argument("--min-score", type=float, default=60, 
                           help="Minimum AI score")
    scan_parser.add_argument("--min-price", type=float, default=0, 
                           help="Minimum stock price")
    scan_parser.add_argument("--max-price", type=float, default=1000, 
                           help="Maximum stock price")
    scan_parser.add_argument("--trend", choices=["bullish", "bearish", "neutral"],
                           help="Trend filter")
    scan_parser.add_argument("--rsi-min", type=int, default=0, help="Minimum RSI")
    scan_parser.add_argument("--rsi-max", type=int, default=100, help="Maximum RSI")
    scan_parser.add_argument("--export", action="store_true", 
                           help="Export results to CSV")
    
    # Check dependencies command
    check_parser = subparsers.add_parser("check", help="Check system dependencies")
    
    args = parser.parse_args()
    
    try:
        logger.info("AI Stock Analyzer starting...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        if args.command == "web":
            run_streamlit_app()
        
        elif args.command == "scan":
            run_cli_scanner(args)
        
        elif args.command == "check":
            if check_dependencies():
                print("✅ All dependencies check passed")
                return 0
            else:
                print("❌ Dependency check failed")
                return 1
        
        else:
            # Default to web app if no command specified
            print("No command specified. Starting web application...")
            run_streamlit_app()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        error_reporter.report_error("application_failure", str(e))
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)