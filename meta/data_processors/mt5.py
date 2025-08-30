import datetime as dt
import os
from typing import List

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from meta.data_processors._base import _Base


class MT5(_Base):
    """
    MetaTrader 5 data processor for FinRL-Meta
    
    Features:
    - Connects to MT5 terminal for live/offline data
    - Supports multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
    - Handles forex, stocks, indices, and commodities
    - Automatic timezone handling
    - Volume data support
    """
    
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        
        # MT5 timeframe mapping
        self.timeframe_mapping = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "1w": mt5.TIMEFRAME_W1,
            "1M": mt5.TIMEFRAME_MN1,
        }
        
        # Initialize MT5 connection
        self.initialize_mt5()
    
    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5 terminal"""
        if not mt5.initialize():
            print("MT5 initialization failed!")
            print("Error:", mt5.last_error())
            raise ConnectionError("Failed to connect to MetaTrader 5 terminal")
        
        print("MetaTrader 5 terminal connected successfully")
        
        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info is not None:
            print(f"Terminal: {getattr(terminal_info, 'name', 'MetaTrader 5')}")
            if hasattr(terminal_info, 'build'):
                print(f"Build: {terminal_info.build}")
            if hasattr(terminal_info, 'connected'):
                print(f"Connected: {terminal_info.connected}")
            if hasattr(terminal_info, 'trade_allowed'):
                print(f"Trade allowed: {terminal_info.trade_allowed}")
    
    def download_data(
        self, 
        ticker_list: List[str], 
        save_path: str = "./data/mt5_dataset.csv"
    ):
        """
        Download data from MT5 for specified tickers
        
        Args:
            ticker_list: List of ticker symbols (e.g., ["EURUSD", "GBPUSD"])
            save_path: Path to save the downloaded data
        """
        if not ticker_list:
            raise ValueError("ticker_list cannot be empty")
        
        # Convert time interval to MT5 timeframe
        if self.time_interval not in self.timeframe_mapping:
            raise ValueError(f"Unsupported time_interval: {self.time_interval}. "
                           f"Supported: {list(self.timeframe_mapping.keys())}")
        
        mt5_timeframe = self.timeframe_mapping[self.time_interval]
        
        # Convert dates to datetime
        start_dt = dt.datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = dt.datetime.strptime(self.end_date, "%Y-%m-%d")
        
        # Add time if not specified
        if self.time_interval in ["1m", "5m", "15m", "30m", "1h", "4h"]:
            start_dt = start_dt.replace(hour=0, minute=0, second=0)
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        
        final_df = pd.DataFrame()
        
        for ticker in ticker_list:
            print(f"Downloading data for {ticker}...")
            
            try:
                # Get historical data from MT5
                rates = mt5.copy_rates_range(
                    ticker, 
                    mt5_timeframe, 
                    start_dt, 
                    end_dt
                )
                
                if rates is None or len(rates) == 0:
                    print(f"No data found for {ticker}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['tic'] = ticker
                
                # Rename columns to match FinRL-Meta format
                df.rename(columns={
                    'time': 'time',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'tick_volume': 'volume',
                    'spread': 'spread',
                    'real_volume': 'real_volume'
                }, inplace=True)
                
                # Convert timestamp to datetime
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Add date features
                df['day'] = df['time'].dt.dayofweek
                
                # Select required columns
                df = df[['tic', 'time', 'open', 'high', 'low', 'close', 'volume', 'day']]
                
                final_df = pd.concat([final_df, df], axis=0, ignore_index=True)
                
                print(f"Downloaded {len(df)} records for {ticker}")
                
            except Exception as e:
                print(f"Error downloading data for {ticker}: {str(e)}")
                continue
        
        if final_df.empty:
            raise ValueError("No data was downloaded from MT5")
        
        self.dataframe = final_df
        self.save_data(save_path)
        
        print(f"Download complete! Dataset saved to {save_path}")
        print(f"Shape of DataFrame: {self.dataframe.shape}")
    
    def get_available_symbols(self):
        """Get list of available symbols from MT5"""
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        
        symbol_names = [symbol.name for symbol in symbols]
        return symbol_names
    
    def get_symbol_info(self, symbol: str):
        """Get detailed information about a specific symbol"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            'name': info.name,
            'currency_base': info.currency_base,
            'currency_profit': info.currency_profit,
            'digits': info.digits,
            'spread': info.spread,
            'spread_float': info.spread_float,
            'ticks_bookdepth': info.ticks_bookdepth,
            'trade_calc_mode': info.trade_calc_mode,
            'trade_mode': info.trade_mode,
            'start_time': info.start_time,
            'expiration_time': info.expiration_time,
            'trade_stops_level': info.trade_stops_level,
            'trade_freeze_level': info.trade_freeze_level
        }
    
    def get_current_price(self, symbol: str):
        """Get current price for a symbol"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': pd.to_datetime(tick.time, unit='s')
        }
    
    def __del__(self):
        """Cleanup MT5 connection"""
        try:
            mt5.shutdown()
        except:
            pass
