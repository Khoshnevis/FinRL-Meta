#!/usr/bin/env python3
"""
Test script to check MT5 data availability
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta

def test_mt5_data():
    """Test what data is available in MT5"""
    print("🔍 Testing MT5 Data Availability")
    print("=" * 50)
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    print("✅ MT5 initialized successfully")
    
    # Test different symbols
    symbols_to_test = ["EURUSD", "GBPUSD", "AUDUSD", "AUDCAD", "GBPJPY"]
    
    for symbol in symbols_to_test:
        print(f"\n📊 Testing {symbol}:")
        
        # Test different timeframes
        timeframes = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "1h": mt5.TIMEFRAME_H1,
            "1d": mt5.TIMEFRAME_D1
        }
        
        for tf_name, tf in timeframes.items():
            try:
                # Try to get recent data
                rates = mt5.copy_rates_from(symbol, tf, datetime.now(), 100)
                if rates is not None and len(rates) > 0:
                    print(f"  ✅ {tf_name}: {len(rates)} records available")
                    # Show date range
                    first_date = datetime.fromtimestamp(rates[0]['time'])
                    last_date = datetime.fromtimestamp(rates[-1]['time'])
                    print(f"     Date range: {first_date.strftime('%Y-%m-%d %H:%M')} to {last_date.strftime('%Y-%m-%d %H:%M')}")
                else:
                    print(f"  ❌ {tf_name}: No data available")
            except Exception as e:
                print(f"  ❌ {tf_name}: Error - {e}")
    
    # Test historical data range
    print(f"\n📅 Testing Historical Data Range:")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days
    
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    for symbol in ["EURUSD", "AUDUSD"]:
        try:
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_date, end_date)
            if rates is not None and len(rates) > 0:
                print(f"  ✅ {symbol}: {len(rates)} records in range")
            else:
                print(f"  ❌ {symbol}: No data in range")
        except Exception as e:
            print(f"  ❌ {symbol}: Error - {e}")
    
    mt5.shutdown()
    print("\n✅ MT5 test completed")

if __name__ == "__main__":
    test_mt5_data()
