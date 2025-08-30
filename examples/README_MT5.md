# MT5 Data Processor for FinRL-Meta

## Overview

The MT5 (MetaTrader 5) data processor allows you to integrate MetaTrader 5 terminal data directly into FinRL-Meta for forex trading and reinforcement learning applications.

## Features

- **Direct MT5 Integration**: Connect to MetaTrader 5 terminal for live/offline data
- **Multi-Timeframe Support**: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
- **Multi-Asset Support**: Forex, stocks, indices, commodities
- **Real-time Data**: Access current prices and market information
- **Historical Data**: Download historical OHLCV data
- **Automatic Cleanup**: Proper connection management and cleanup

## Installation

### Prerequisites

1. **MetaTrader 5 Terminal**: Must be installed and running
2. **Python Package**: Install the MetaTrader5 Python wrapper

```bash
pip install MetaTrader5
```

### Verify Installation

```python
import MetaTrader5 as mt5
if mt5.initialize():
    print("MT5 connection successful!")
    mt5.shutdown()
else:
    print("MT5 connection failed!")
```

## Usage

### Basic Usage

```python
from meta.data_processor import DataProcessor, DataSource

# Initialize MT5 data processor
processor = DataProcessor(
    data_source=DataSource.mt5,
    start_date="2023-01-01",
    end_date="2023-12-31",
    time_interval="1h"  # 1-hour candles
)

# Download forex data
forex_pairs = ["EURUSD", "GBPUSD", "USDJPY"]
processor.download_data(ticker_list=forex_pairs)

# Clean and process data
processor.clean_data()
processor.fillna()

# Add technical indicators
processor.add_technical_indicator([
    "macd", "rsi", "boll_ub", "boll_lb"
])

# Save processed data
processor.save_data("./data/mt5_forex_data.csv")
```

### Advanced Features

#### Get Available Symbols

```python
mt5_processor = processor.processor
symbols = mt5_processor.get_available_symbols()
print(f"Available symbols: {symbols[:10]}...")  # First 10 symbols
```

#### Get Symbol Information

```python
symbol_info = mt5_processor.get_symbol_info("EURUSD")
if symbol_info:
    print(f"Currency pair: {symbol_info['currency_base']}/{symbol_info['currency_profit']}")
    print(f"Digits: {symbol_info['digits']}")
    print(f"Spread: {symbol_info['spread']}")
```

#### Get Current Prices

```python
current_price = mt5_processor.get_current_price("EURUSD")
if current_price:
    print(f"Bid: {current_price['bid']}")
    print(f"Ask: {current_price['ask']}")
    print(f"Last: {current_price['last']}")
```

## Supported Timeframes

| FinRL-Meta | MT5 Constant | Description |
|-------------|--------------|-------------|
| `1m`       | `TIMEFRAME_M1` | 1 minute |
| `5m`       | `TIMEFRAME_M5` | 5 minutes |
| `15m`      | `TIMEFRAME_M15` | 15 minutes |
| `30m`      | `TIMEFRAME_M30` | 30 minutes |
| `1h`       | `TIMEFRAME_H1` | 1 hour |
| `4h`       | `TIMEFRAME_H4` | 4 hours |
| `1d`       | `TIMEFRAME_D1` | 1 day |
| `1w`       | `TIMEFRAME_W1` | 1 week |
| `1M`       | `TIMEFRAME_MN1` | 1 month |

## Data Format

The processor outputs data in the standard FinRL-Meta format:

```python
columns = [
    "tic",              # Ticker symbol (e.g., "EURUSD")
    "time",             # Timestamp (datetime)
    "open",             # Opening price
    "high",             # High price
    "low",              # Low price
    "close",            # Closing price
    "volume",           # Trading volume
    "day"               # Day of week (0-6, Monday=0)
]
```

## Error Handling

The processor includes comprehensive error handling:

- **Connection Errors**: Automatic detection of MT5 terminal availability
- **Data Validation**: Checks for empty or invalid data
- **Symbol Validation**: Verifies symbol availability before downloading
- **Graceful Degradation**: Continues processing other symbols if one fails

## Common Issues and Solutions

### 1. MT5 Terminal Not Accessible

**Error**: "Failed to connect to MetaTrader 5 terminal"

**Solutions**:
- Ensure MT5 terminal is running
- Check if MT5 is running as administrator
- Verify MT5 version compatibility
- Restart MT5 terminal

### 2. No Data Downloaded

**Error**: "No data was downloaded from MT5"

**Solutions**:
- Check symbol names (case-sensitive)
- Verify date range availability
- Ensure MT5 has historical data for the period
- Check internet connection for live data

### 3. Import Error

**Error**: "No module named 'MetaTrader5'"

**Solutions**:
```bash
pip install MetaTrader5
# or
conda install -c conda-forge metatrader5
```

## Integration with Trading Environments

### Forex Trading Environment

```python
from meta.env_fx_trading.env_fx import tgym

# Load MT5 data
processor = DataProcessor(
    data_source=DataSource.mt5,
    start_date="2023-01-01",
    end_date="2023-12-31",
    time_interval="1h"
)

processor.download_data(["EURUSD", "GBPUSD"])
processor.clean_data()

# Use with forex trading environment
env = tgym(
    df=processor.dataframe,
    env_config_file="./config/forex_config.json"
)
```

### Portfolio Optimization

```python
from meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

# Load MT5 data for multiple assets
processor.download_data(["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
processor.clean_data()
processor.add_technical_indicator(["macd", "rsi"])

# Use with portfolio optimization environment
env = PortfolioOptimizationEnv(
    df=processor.dataframe,
    **env_config
)
```

## Performance Considerations

- **Data Volume**: Large historical datasets may take time to download
- **Memory Usage**: Monitor memory usage with large datasets
- **Connection Stability**: Ensure stable MT5 connection for live data
- **Batch Processing**: Consider downloading data in smaller time chunks

## Testing

Run the unit tests to verify functionality:

```bash
cd unit_tests
python test_mt5.py
```

## Examples

See the comprehensive example in `examples/MT5_Forex_Data_Example.py` for a complete workflow demonstration.

## Contributing

To contribute to the MT5 data processor:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This MT5 data processor is part of FinRL-Meta and follows the same license terms.

## Support

For issues and questions:

1. Check the common issues section above
2. Review the example code
3. Open an issue on the FinRL-Meta repository
4. Check MT5 documentation for terminal-specific issues
