# MT5 Training and Backtesting Guide for FinRL-Meta

This guide explains how to use your MT5 data processor for training reinforcement learning agents on historical data and backtesting on different timeframes.

## 🚀 Quick Start

### 1. Run the Quick Pipeline
```bash
cd examples
python MT5_Quick_Training_Backtest.py
```

This will:
- Download 1 year of hourly MT5 data for training
- Train a PPO agent for 30,000 timesteps
- Backtest on 15m, 30m, and 1h timeframes
- Generate performance comparison plots

### 2. Run the Complete Pipeline
```bash
python MT5_Complete_Training_Backtesting.py
```

This provides a comprehensive workflow with:
- Extended training period (1.5 years)
- More sophisticated hyperparameters
- Detailed performance metrics
- Advanced visualization

## 📊 Training vs Backtesting Strategy

### Training Data (Longer Period, Coarser Timeframe)
- **Period**: 1-2 years of historical data
- **Timeframe**: 1h (hourly candles)
- **Rationale**: 
  - Longer period provides diverse market conditions
  - Coarser timeframe reduces noise and computational cost
  - Hourly data captures daily patterns without excessive detail

### Backtesting Data (Shorter Period, Finer Timeframes)
- **Period**: 3-6 months of recent data
- **Timeframes**: 15m, 30m, 1h, 4h
- **Rationale**:
  - Recent data reflects current market conditions
  - Multiple timeframes test strategy robustness
  - Finer timeframes provide detailed execution analysis

## 🔧 Configuration

### Training Configuration
```json
{
    "training": {
        "data_source": "mt5",
        "timeframe": "1h",
        "forex_pairs": ["EURUSD", "GBPUSD", "USDJPY"],
        "training_period_days": 540,
        "model": {
            "algorithm": "PPO",
            "total_timesteps": 100000
        }
    }
}
```

### Environment Configuration
```json
{
    "environment": {
        "initial_balance": 100000,
        "max_position_size": 0.1,
        "transaction_fee": 0.0001,
        "reward_function": "sharpe"
    }
}
```

## 📈 Training Workflow

### Step 1: Data Preparation
```python
from meta.data_processor import DataProcessor, DataSource

# Download training data
processor = DataProcessor(
    data_source=DataSource.mt5,
    start_date="2022-01-01",
    end_date="2023-06-30",
    time_interval="1h"
)

forex_pairs = ["EURUSD", "GBPUSD", "USDJPY"]
processor.download_data(ticker_list=forex_pairs)
processor.clean_data()
processor.fillna()
processor.add_technical_indicator(["macd", "rsi", "boll_ub", "boll_lb"])
```

### Step 2: Environment Creation
```python
from examples.Best_Practice_Forex_Env import BestPracticeForexEnv

env = BestPracticeForexEnv(
    df=processor.dataframe,
    config={
        "initial_balance": 100000,
        "max_position_size": 0.1,
        "transaction_fee": 0.0001,
        "reward_function": "sharpe"
    }
)
```

### Step 3: Agent Training
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

vec_env = DummyVecEnv([lambda: env])

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
)

model.learn(total_timesteps=100000)
model.save("./trained_models/forex_ppo_trained")
```

## 🧪 Backtesting Workflow

### Single Timeframe Backtesting
```python
# Download backtest data
backtest_processor = DataProcessor(
    data_source=DataSource.mt5,
    start_date="2023-07-01",
    end_date="2023-12-31",
    time_interval="15m"
)

backtest_processor.download_data(ticker_list=forex_pairs)
backtest_processor.clean_data()
backtest_processor.fillna()
backtest_processor.add_technical_indicator(["macd", "rsi", "boll_ub", "boll_lb"])

# Create backtest environment
backtest_env = BestPracticeForexEnv(df=backtest_processor.dataframe, config=config)

# Run backtest
results = backtest_model("./trained_models/forex_ppo_trained", backtest_env, num_episodes=5)
```

### Multi-Timeframe Backtesting
```python
timeframes = ["15m", "30m", "1h", "4h"]
results_comparison = {}

for tf in timeframes:
    df = download_backtest_data(tf, start_date, end_date)
    env = create_environment(df)
    results = backtest_model(model_path, env, num_episodes=5)
    results_comparison[tf] = results
```

## 📊 Performance Metrics

### Key Metrics
- **Total Reward**: Cumulative reward across episode
- **Final Balance**: Ending portfolio value
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable episodes
- **Profit Factor**: Gross profit / Gross loss

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Expected Shortfall**: Average loss beyond VaR
- **Calmar Ratio**: Annual return / Max drawdown
- **Sortino Ratio**: Downside deviation adjusted return

## 🎯 Best Practices

### 1. Data Quality
- Ensure sufficient historical data (minimum 1000 data points)
- Handle missing values appropriately
- Remove outliers that could skew training
- Validate data consistency across timeframes

### 2. Training Strategy
- Start with simpler models and gradually increase complexity
- Use cross-validation to prevent overfitting
- Monitor training progress with TensorBoard
- Save checkpoints during training

### 3. Backtesting Strategy
- Use out-of-sample data for validation
- Test on multiple timeframes to ensure robustness
- Implement realistic transaction costs and slippage
- Consider market regime changes

### 4. Risk Management
- Set appropriate position size limits
- Implement stop-loss and take-profit mechanisms
- Monitor correlation between assets
- Use portfolio-level risk controls

## 🔍 Troubleshooting

### Common Issues

#### 1. MT5 Connection Problems
```python
import MetaTrader5 as mt5

# Check connection
if not mt5.initialize():
    print("MT5 connection failed!")
    print("Make sure MetaTrader 5 terminal is running")
    mt5.shutdown()
```

#### 2. Data Download Issues
- Verify symbol names match MT5 exactly
- Check date range availability
- Ensure sufficient historical data exists
- Verify timeframe is supported

#### 3. Training Issues
- Reduce learning rate if training is unstable
- Increase batch size for better stability
- Check observation space dimensions
- Verify reward function implementation

#### 4. Backtesting Issues
- Ensure data alignment between training and testing
- Check for data gaps or inconsistencies
- Verify environment configuration matches training
- Monitor memory usage for large datasets

## 📚 Advanced Features

### 1. Ensemble Methods
```python
# Train multiple models
models = []
for i in range(3):
    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=50000)
    models.append(model)

# Ensemble prediction
actions = [model.predict(obs)[0] for model in models]
ensemble_action = np.mean(actions, axis=0)
```

### 2. Hyperparameter Optimization
```python
from optuna import create_study

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, n_steps=n_steps)
    model.learn(total_timesteps=10000)
    
    # Evaluate and return metric
    return evaluate_model(model, test_env)

study = create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### 3. Custom Reward Functions
```python
class CustomRewardEnv(BestPracticeForexEnv):
    def calculate_reward(self, action, info):
        # Implement custom reward logic
        base_reward = super().calculate_reward(action, info)
        
        # Add custom components
        volatility_penalty = self.calculate_volatility_penalty()
        momentum_bonus = self.calculate_momentum_bonus()
        
        return base_reward + volatility_penalty + momentum_bonus
```

## 🚀 Next Steps

1. **Experiment with Different Algorithms**
   - Try SAC, TD3, or A2C
   - Implement custom algorithms
   - Use ensemble methods

2. **Advanced Data Processing**
   - Add more technical indicators
   - Implement feature engineering
   - Use market regime detection

3. **Production Deployment**
   - Implement real-time data feeds
   - Add risk management systems
   - Create monitoring dashboards

4. **Research and Development**
   - Test on different asset classes
   - Implement multi-agent systems
   - Explore meta-learning approaches

## 📖 Additional Resources

- [FinRL-Meta Documentation](https://finrl-meta.readthedocs.io/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [MetaTrader5 Python Documentation](https://www.mql5.com/en/docs/python_metatrader5)
- [Reinforcement Learning for Trading](https://github.com/AI4Finance-Foundation/FinRL)

## 🤝 Contributing

Feel free to contribute improvements to the MT5 data processor or training workflows:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
