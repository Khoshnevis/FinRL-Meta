# MT5 GPU-Fixed Training and Backtesting for FinRL-Meta

This directory contains comprehensive scripts to train and backtest MT5 models while automatically handling GPU compatibility issues.

## 🚀 Quick Start

### 1. GPU Troubleshooting
First, diagnose any GPU issues:
```bash
python GPU_Troubleshooter.py
```

### 2. Simple 1-Minute Training
Train on 3 months of 1-minute data:
```bash
python MT5_1min_Training.py
```

### 3. Full Training Pipeline
Complete training with custom parameters:
```bash
python MT5_GPU_Fixed_Training.py --timeframe 1m --training_months 3 --backtest_months 1
```

### 4. Backtesting After Training
Test trained models on new market data:
```bash
python MT5_Backtest_After_Training.py --model_path ./trained_models/your_model.zip
```

## 📁 Scripts Overview

### `GPU_Troubleshooter.py`
- **Purpose**: Diagnose and fix GPU compatibility issues
- **Features**:
  - Automatic GPU detection
  - CUDA compatibility testing
  - Known problematic GPU identification (RTX 5090, 4090 Ti, etc.)
  - PyTorch and Stable-Baselines3 verification
  - MT5 connection testing
  - Comprehensive troubleshooting solutions

### `MT5_1min_Training.py`
- **Purpose**: Simple training on 1-minute data
- **Features**:
  - Automatic GPU/CPU fallback
  - 3 months of 1-minute candle data
  - Basic PPO training
  - Quick backtest validation
  - Minimal configuration required

### `MT5_GPU_Fixed_Training.py`
- **Purpose**: Full-featured training pipeline
- **Features**:
  - Advanced GPU management
  - Customizable training parameters
  - Multiple timeframe support
  - Comprehensive backtesting
  - Results visualization
  - Command-line configuration

### `MT5_Backtest_After_Training.py`
- **Purpose**: Backtesting trained models
- **Features**:
  - Load pre-trained models
  - Test on new market data
  - Comprehensive performance metrics
  - Advanced visualizations
  - Multiple episode testing

## 🔧 GPU Issue Solutions

### Common Problems and Solutions

#### 1. RTX 5090/4090 Ti Compatibility Issues
**Problem**: These GPUs use new architecture (sm_120) not yet fully supported by PyTorch.

**Solutions**:
```bash
# Option 1: Use CPU (slower but stable)
python MT5_1min_Training.py  # Automatically falls back to CPU

# Option 2: Try PyTorch nightly builds
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Option 3: Wait for official PyTorch support
# Monitor: https://pytorch.org/get-started/locally/
```

#### 2. CUDA Installation Issues
**Problem**: PyTorch installed without CUDA support.

**Solutions**:
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Driver Compatibility Issues
**Problem**: NVIDIA drivers incompatible with CUDA version.

**Solutions**:
- Update NVIDIA drivers to latest version
- Check driver compatibility matrix on NVIDIA website
- Ensure driver version supports your CUDA version

#### 4. Memory Issues
**Problem**: GPU memory allocation failures.

**Solutions**:
- Reduce batch size in training configuration
- Use gradient accumulation
- Monitor GPU memory with `nvidia-smi`
- Set `CUDA_VISIBLE_DEVICES` to specific GPU

## 📊 Training Commands

### Basic 1-Minute Training (3 months)
```bash
# Simple training with automatic GPU handling
python MT5_1min_Training.py
```

### Advanced Training with Custom Parameters
```bash
# Train on 1-minute data for 3 months, backtest on 1 month
python MT5_GPU_Fixed_Training.py \
    --timeframe 1m \
    --training_months 3 \
    --backtest_months 1 \
    --symbols EURUSD GBPUSD USDJPY \
    --timesteps 500000

# Train on 5-minute data for 6 months
python MT5_GPU_Fixed_Training.py \
    --timeframe 5m \
    --training_months 6 \
    --backtest_months 2 \
    --timesteps 1000000

# Train on 1-hour data for 12 months
python MT5_GPU_Fixed_Training.py \
    --timeframe 1h \
    --training_months 12 \
    --backtest_months 3 \
    --timesteps 2000000
```

### Training Parameters
- `--timeframe`: Trading timeframe (1m, 5m, 15m, 30m, 1h, 4h)
- `--training_months`: Number of months for training data
- `--backtest_months`: Number of months for backtest data
- `--symbols`: Trading symbols (space-separated)
- `--timesteps`: Training timesteps (higher = longer training)

## 🧪 Backtesting Commands

### Backtest Trained Model
```bash
# Basic backtest on 1 month of new data
python MT5_Backtest_After_Training.py \
    --model_path ./trained_models/forex_ppo_1m_20241201_143022.zip

# Extended backtest with custom parameters
python MT5_Backtest_After_Training.py \
    --model_path ./trained_models/forex_ppo_1m_20241201_143022.zip \
    --timeframe 1m \
    --backtest_months 2 \
    --symbols EURUSD GBPUSD USDJPY \
    --episodes 20
```

### Backtest Parameters
- `--model_path`: Path to trained model (.zip file)
- `--timeframe`: Trading timeframe for backtest
- `--backtest_months`: Number of months for backtest data
- `--symbols`: Trading symbols to test
- `--episodes`: Number of backtest episodes

## 📈 Results and Outputs

### Training Outputs
- **Models**: Saved to `./trained_models/`
- **Logs**: Tensorboard logs in `./tensorboard_log/`
- **Console**: Real-time training progress

### Backtesting Outputs
- **Results**: JSON files in `./results/`
- **Plots**: Comprehensive visualizations in `./backtest_plots/`
- **Metrics**: Performance summaries and statistics

### Key Performance Metrics
- **Returns**: Total and average returns
- **Risk**: Maximum drawdown, volatility
- **Efficiency**: Sharpe ratio, Calmar ratio
- **Trading**: Win rate, profit factor, trade count

## 🖥️ System Requirements

### Minimum Requirements
- Python 3.8+
- MetaTrader 5 terminal
- 8GB RAM
- Stable internet connection

### Recommended Requirements
- Python 3.9+
- NVIDIA GPU with 8GB+ VRAM
- 16GB+ RAM
- SSD storage
- High-speed internet

### Required Python Packages
```bash
pip install -r requirements.txt
# Or install individually:
pip install torch stable-baselines3 MetaTrader5 pandas numpy matplotlib seaborn
```

## 🔍 Troubleshooting

### Common Issues

#### 1. MT5 Connection Failed
```bash
# Check if MT5 terminal is running
# Ensure MT5 is running as administrator
# Verify MT5 version compatibility
```

#### 2. No Data Downloaded
```bash
# Check symbol names (case-sensitive)
# Verify date range availability
# Ensure MT5 has historical data
# Check internet connection
```

#### 3. Training Crashes
```bash
# Run GPU troubleshooter first
python GPU_Troubleshooter.py

# Try with CPU fallback
python MT5_1min_Training.py

# Check memory usage
nvidia-smi
```

#### 4. Import Errors
```bash
# Install missing packages
pip install MetaTrader5 stable-baselines3 torch

# Check Python path
python -c "import sys; print(sys.path)"
```

### Getting Help

1. **Run GPU Troubleshooter**: `python GPU_Troubleshooter.py`
2. **Check Console Output**: Look for specific error messages
3. **Verify Dependencies**: Ensure all packages are installed
4. **Check MT5 Connection**: Verify MetaTrader 5 is accessible
5. **Review Logs**: Check tensorboard logs for training issues

## 📚 Advanced Usage

### Custom Training Configurations
```python
# Modify training parameters in the scripts
config = {
    'learning_rate': 1e-4,        # Lower for stability
    'n_steps': 4096,              # Higher for better gradients
    'batch_size': 128,            # Higher for better convergence
    'n_epochs': 20,               # More epochs per update
    'gamma': 0.99,                # Discount factor
    'ent_coef': 0.005,           # Lower for less exploration
}
```

### Custom Reward Functions
```python
# Modify reward function in environment
config = {
    'reward_function': 'sharpe',   # Options: sharpe, calmar, sortino, simple
    'lookback_window': 30,        # Technical indicator lookback
}
```

### Multi-Asset Trading
```python
# Add more trading symbols
symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD']
```

## 🎯 Best Practices

### Training
1. **Start Simple**: Use 1-minute data with 3 months for initial testing
2. **Monitor Progress**: Use tensorboard to track training metrics
3. **Validate Early**: Run quick backtests during training
4. **Save Checkpoints**: Models are automatically saved during training

### Backtesting
1. **Use Fresh Data**: Test on data from after training period
2. **Multiple Episodes**: Run multiple backtest episodes for robustness
3. **Risk Metrics**: Focus on drawdown and Sharpe ratio
4. **Out-of-Sample**: Avoid overfitting by testing on unseen data

### GPU Optimization
1. **Monitor Memory**: Use `nvidia-smi` to track GPU usage
2. **Batch Sizes**: Adjust batch size based on GPU memory
3. **Mixed Precision**: Consider using mixed precision training
4. **Fallback Strategy**: Always have CPU fallback for compatibility

## 📞 Support

For additional help:
- Check the GPU Troubleshooter output
- Review console error messages
- Verify all dependencies are installed
- Ensure MT5 terminal is accessible
- Check system requirements

## 🔄 Updates and Maintenance

### Regular Updates
- Update PyTorch to latest stable version
- Update NVIDIA drivers regularly
- Monitor for new GPU compatibility fixes
- Check FinRL-Meta updates

### Version Compatibility
- PyTorch: 2.0.0+
- Stable-Baselines3: 2.7.0+
- MetaTrader5: 5.0.45+
- Python: 3.8+

---

**Note**: These scripts automatically handle most GPU compatibility issues and provide fallback to CPU when needed. For optimal performance, ensure your GPU setup is properly configured and up-to-date.
