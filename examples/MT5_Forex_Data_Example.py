#!/usr/bin/env python3
"""
MT5 Forex Data Example for FinRL-Meta

This example demonstrates how to use the MT5 data processor to:
1. Download forex data from MetaTrader 5 terminal
2. Process and clean the data
3. Add technical indicators
4. Prepare data for reinforcement learning training

Requirements:
- MetaTrader 5 terminal installed and running
- MetaTrader5 Python package installed
- FinRL-Meta framework
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add the parent directory to the path to import FinRL-Meta modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta.data_processor import DataProcessor, DataSource


def main():
    """Main function demonstrating MT5 data processor usage"""
    
    print("=== MT5 Forex Data Processor Example ===\n")
    
    # 1. Initialize MT5 Data Processor
    print("1. Initializing MT5 Data Processor...")
    
    # Set date range (last 30 days)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    try:
        processor = DataProcessor(
            data_source=DataSource.mt5,
            start_date=start_date,
            end_date=end_date,
            time_interval="1h"  # 1-hour candles
        )
        print("✓ MT5 Data Processor initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize MT5 processor: {str(e)}")
        print("Make sure MetaTrader 5 terminal is running and accessible")
        return
    
    # 2. Download Forex Data
    print("\n2. Downloading forex data...")
    
    # Common forex pairs
    forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    
    try:
        processor.download_data(ticker_list=forex_pairs)
        print("✓ Forex data downloaded successfully")
        
    except Exception as e:
        print(f"✗ Failed to download data: {str(e)}")
        return
    
    # 3. Clean and Process Data
    print("\n3. Cleaning and processing data...")
    
    try:
        processor.clean_data()
        print("✓ Data cleaned successfully")
        
        # Fill missing values
        processor.fillna()
        print("✓ Missing values handled")
        
    except Exception as e:
        print(f"✗ Failed to clean data: {str(e)}")
        return
    
    # 4. Add Technical Indicators
    print("\n4. Adding technical indicators...")
    
    # List of technical indicators to add
    tech_indicators = [
        "macd", "macd_signal", "macd_hist",
        "rsi", "cci", "dx",
        "boll_ub", "boll_lb"
    ]
    
    try:
        processor.add_technical_indicator(
            tech_indicator_list=tech_indicators,
            select_stockstats_talib=0  # Use stockstats (default)
        )
        print("✓ Technical indicators added successfully")
        
    except Exception as e:
        print(f"✗ Failed to add technical indicators: {str(e)}")
        return
    
    # 5. Display Data Summary
    print("\n5. Data Summary:")
    print(f"   Total records: {len(processor.dataframe)}")
    print(f"   Symbols: {processor.dataframe['tic'].unique()}")
    print(f"   Date range: {processor.dataframe['time'].min()} to {processor.dataframe['time'].max()}")
    print(f"   Columns: {list(processor.dataframe.columns)}")
    
    # 6. Show Sample Data
    print("\n6. Sample Data (first 5 rows):")
    print(processor.dataframe.head())
    
    # 7. Save Final Processed Data
    print("\n7. Saving processed data...")
    
    try:
        # Save using the processor's save_data method
        processor.processor.save_data("./data/mt5_forex_processed.csv")
        print("✓ Processed data saved successfully")
        
    except Exception as e:
        print(f"✗ Failed to save data: {str(e)}")
        return
    
    # 8. Additional MT5 Features
    print("\n8. Exploring additional MT5 features...")
    
    try:
        # Get available symbols
        mt5_processor = processor.processor
        available_symbols = mt5_processor.get_available_symbols()
        print(f"   Available symbols: {len(available_symbols)}")
        
        # Get symbol info for EURUSD
        eurusd_info = mt5_processor.get_symbol_info("EURUSD")
        if eurusd_info:
            print(f"   EURUSD info: {eurusd_info['currency_base']}/{eurusd_info['currency_profit']}")
            print(f"   Digits: {eurusd_info['digits']}")
            print(f"   Spread: {eurusd_info['spread']}")
        
        # Get current price for EURUSD
        current_price = mt5_processor.get_current_price("EURUSD")
        if current_price:
            print(f"   Current EURUSD - Bid: {current_price['bid']}, Ask: {current_price['ask']}")
            
    except Exception as e:
        print(f"   Note: Some MT5 features not available: {str(e)}")
    
    print("\n=== Example completed successfully! ===")
    print("\nNext steps:")
    print("1. Use the processed data with FinRL-Meta trading environments")
    print("2. Train reinforcement learning agents on forex data")
    print("3. Backtest trading strategies")
    print("4. Implement live trading systems")


def check_mt5_connection():
    """Check if MT5 terminal is accessible"""
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            terminal_info = mt5.terminal_info()
            print(f"✓ MT5 Terminal: {getattr(terminal_info, 'name', 'MetaTrader 5')}")
            
            # Check what attributes are available and print them safely
            if hasattr(terminal_info, 'build'):
                print(f"  Build: {terminal_info.build}")
            if hasattr(terminal_info, 'connected'):
                print(f"  Connected: {terminal_info.connected}")
            if hasattr(terminal_info, 'trade_allowed'):
                print(f"  Trade allowed: {terminal_info.trade_allowed}")
            
            mt5.shutdown()
            return True
        else:
            print("✗ MT5 Terminal not accessible")
            return False
    except ImportError:
        print("✗ MetaTrader5 package not installed")
        print("  Install with: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f"✗ MT5 connection error: {str(e)}")
        return False


if __name__ == "__main__":
    print("Checking MT5 connection...")
    if check_mt5_connection():
        main()
    else:
        print("\nPlease ensure MetaTrader 5 terminal is running and accessible")
        print("Then run this example again.")
