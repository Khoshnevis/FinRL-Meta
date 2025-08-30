#!/usr/bin/env python3
"""
Unit tests for MT5 data processor

This file tests the MT5 data processor functionality
"""

import unittest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta.data_processors._base import DataSource
from meta.data_processors.mt5 import MT5


class TestMT5Processor(unittest.TestCase):
    """Test cases for MT5 data processor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-31"
        self.time_interval = "1h"
        
    @patch('meta.data_processors.mt5.mt5')
    def test_mt5_initialization(self, mock_mt5):
        """Test MT5 processor initialization"""
        # Mock successful MT5 initialization
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = Mock(
            name="Test Terminal",
            build=12345,
            connected=True,
            trade_allowed=True
        )
        
        processor = MT5(
            data_source="mt5",
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.time_interval
        )
        
        self.assertEqual(processor.time_interval, self.time_interval)
        self.assertEqual(processor.start_date, self.start_date)
        self.assertEqual(processor.end_date, self.end_date)
        
    @patch('meta.data_processors.mt5.mt5')
    def test_mt5_connection_failure(self, mock_mt5):
        """Test MT5 connection failure handling"""
        # Mock failed MT5 initialization
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = "Connection failed"
        
        with self.assertRaises(ConnectionError):
            MT5(
                data_source="mt5",
                start_date=self.start_date,
                end_date=self.end_date,
                time_interval=self.time_interval
            )
    
    def test_timeframe_mapping(self):
        """Test timeframe mapping to MT5 constants"""
        with patch('meta.data_processors.mt5.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.terminal_info.return_value = Mock()
            
            processor = MT5(
                data_source="mt5",
                start_date=self.start_date,
                end_date=self.end_date,
                time_interval="1h"
            )
            
            # Test that timeframe mapping exists
            self.assertIn("1h", processor.timeframe_mapping)
            self.assertIn("1d", processor.timeframe_mapping)
            self.assertIn("1m", processor.timeframe_mapping)
    
    def test_invalid_timeframe(self):
        """Test handling of invalid timeframe"""
        with patch('meta.data_processors.mt5.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.terminal_info.return_value = Mock()
            
            processor = MT5(
                data_source="mt5",
                start_date=self.start_date,
                end_date=self.end_date,
                time_interval="1h"
            )
            
            with self.assertRaises(ValueError):
                processor.download_data(["EURUSD"], save_path="./test.csv")
    
    @patch('meta.data_processors.mt5.mt5')
    def test_download_data_success(self, mock_mt5):
        """Test successful data download"""
        # Mock successful MT5 initialization
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = Mock()
        
        # Mock historical data with proper structure
        import numpy as np
        mock_rates = np.array([
            (1640995200, 1.1300, 1.1310, 1.1290, 1.1305, 1000, 0, 0),  # timestamp, open, high, low, close, volume, spread, real_volume
            (1640998800, 1.1305, 1.1320, 1.1300, 1.1315, 1200, 0, 0),
        ], dtype=[
            ('time', np.int64),
            ('open', np.float64),
            ('high', np.float64),
            ('low', np.float64),
            ('close', np.float64),
            ('tick_volume', np.int64),
            ('spread', np.int64),
            ('real_volume', np.int64)
        ])
        
        mock_mt5.copy_rates_range.return_value = mock_rates
        
        processor = MT5(
            data_source="mt5",
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval="1h"
        )
        
        # Test data download
        processor.download_data(["EURUSD"], save_path="./test.csv")
        
        # Verify data was processed
        self.assertIsNotNone(processor.dataframe)
        self.assertGreater(len(processor.dataframe), 0)
        self.assertIn("tic", processor.dataframe.columns)
        self.assertIn("time", processor.dataframe.columns)
    
    def test_get_available_symbols(self):
        """Test getting available symbols"""
        with patch('meta.data_processors.mt5.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.terminal_info.return_value = Mock()
            
            # Mock symbols with proper name attribute
            mock_symbols = []
            for name in ["EURUSD", "GBPUSD", "USDJPY"]:
                mock_symbol = Mock()
                mock_symbol.name = name
                mock_symbols.append(mock_symbol)
            
            mock_mt5.symbols_get.return_value = mock_symbols
            
            processor = MT5(
                data_source="mt5",
                start_date=self.start_date,
                end_date=self.end_date,
                time_interval=self.time_interval
            )
            
            symbols = processor.get_available_symbols()
            self.assertEqual(len(symbols), 3)
            self.assertIn("EURUSD", symbols)
    
    def test_get_symbol_info(self):
        """Test getting symbol information"""
        with patch('meta.data_processors.mt5.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.terminal_info.return_value = Mock()
            
            # Mock symbol info with proper attributes
            mock_info = Mock()
            mock_info.name = "EURUSD"
            mock_info.currency_base = "EUR"
            mock_info.currency_profit = "USD"
            mock_info.digits = 5
            mock_info.spread = 10
            mock_mt5.symbol_info.return_value = mock_info
            
            processor = MT5(
                data_source="mt5",
                start_date=self.start_date,
                end_date=self.end_date,
                time_interval=self.time_interval
            )
            
            info = processor.get_symbol_info("EURUSD")
            self.assertIsNotNone(info)
            self.assertEqual(info["name"], "EURUSD")
            self.assertEqual(info["currency_base"], "EUR")
    
    def test_get_current_price(self):
        """Test getting current price"""
        with patch('meta.data_processors.mt5.mt5') as mock_mt5:
            mock_mt5.initialize.return_value = True
            mock_mt5.terminal_info.return_value = Mock()
            
            # Mock tick data with proper attributes
            mock_tick = Mock()
            mock_tick.bid = 1.1300
            mock_tick.ask = 1.1305
            mock_tick.last = 1.1302
            mock_tick.volume = 1000
            mock_tick.time = 1640995200
            mock_mt5.symbol_info_tick.return_value = mock_tick
            
            processor = MT5(
                data_source="mt5",
                start_date=self.start_date,
                end_date=self.end_date,
                time_interval=self.time_interval
            )
            
            price = processor.get_current_price("EURUSD")
            self.assertIsNotNone(price)
            self.assertEqual(price["bid"], 1.1300)
            self.assertEqual(price["ask"], 1.1305)


if __name__ == "__main__":
    # Run tests
    unittest.main()
