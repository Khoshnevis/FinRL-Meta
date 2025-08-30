#!/usr/bin/env python3
"""
Complete MT5 Training and Backtesting Pipeline for FinRL-Meta

This script demonstrates:
1. Training on historical data (longer period, coarser timeframe)
2. Backtesting on different timeframes and periods
3. Performance comparison across timeframes
4. Risk metrics and analysis

Requirements:
- MetaTrader 5 terminal installed and running
- MetaTrader5 Python package installed
- FinRL-Meta framework
- Stable-Baselines3
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta.data_processor import DataProcessor, DataSource
from examples.Best_Practice_Forex_Env import BestPracticeForexEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns


def download_training_data():
    """Download training data (longer period, coarser timeframe)"""
    print("=== Downloading Training Data ===")
    
    # Training period: 1.5 years of hourly data
    end_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=540)).strftime("%Y-%m-%d")
    
    print(f"Training period: {start_date} to {end_date}")
    print("Timeframe: 1h (hourly candles)")
    
    try:
        processor = DataProcessor(
            data_source=DataSource.mt5,
            start_date=start_date,
            end_date=end_date,
            time_interval="1h"
        )
        
        # Major forex pairs for training
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        print(f"Downloading data for: {forex_pairs}")
        
        processor.download_data(ticker_list=forex_pairs)
        processor.clean_data()
        processor.fillna()
        
        # Add comprehensive technical indicators
        tech_indicators = [
            "macd", "macd_signal", "macd_hist",
            "rsi", "cci", "dx", "adx",
            "boll_ub", "boll_lb", "boll_mid",
            "close_10_sma", "close_30_sma", "close_50_sma",
            "volume_10_sma", "volume_30_sma"
        ]
        processor.add_technical_indicator(tech_indicator_list=tech_indicators)
        
        print(f"✓ Training data prepared: {processor.dataframe.shape[0]} records")
        print(f"  Columns: {list(processor.dataframe.columns)}")
        
        return processor.dataframe
        
    except Exception as e:
        print(f"✗ Error downloading training data: {str(e)}")
        raise


def download_backtest_data(timeframe, start_date, end_date):
    """Download backtesting data for specific timeframe and period"""
    print(f"\n=== Downloading Backtest Data ({timeframe}) ===")
    print(f"Period: {start_date} to {end_date}")
    
    try:
        processor = DataProcessor(
            data_source=DataSource.mt5,
            start_date=start_date,
            end_date=end_date,
            time_interval=timeframe
        )
        
        # Same forex pairs as training
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        processor.download_data(ticker_list=forex_pairs)
        processor.clean_data()
        processor.fillna()
        
        # Add same technical indicators
        tech_indicators = [
            "macd", "macd_signal", "macd_hist",
            "rsi", "cci", "dx", "adx",
            "boll_ub", "boll_lb", "boll_mid",
            "close_10_sma", "close_30_sma", "close_50_sma",
            "volume_10_sma", "volume_30_sma"
        ]
        processor.add_technical_indicator(tech_indicator_list=tech_indicators)
        
        print(f"✓ Backtest data prepared: {processor.dataframe.shape[0]} records")
        return processor.dataframe
        
    except Exception as e:
        print(f"✗ Error downloading backtest data: {str(e)}")
        raise


def create_environment(df, config=None):
    """Create trading environment with configuration"""
    default_config = {
        "initial_balance": 100000,      # Starting capital
        "max_position_size": 0.1,       # Max 10% of balance per trade
        "transaction_fee": 0.0001,      # 1 pip = 0.01%
        "slippage": 0.00005,            # 0.5 pip slippage
        "max_drawdown": 0.20,           # 20% max drawdown
        "risk_per_trade": 0.02,         # 2% risk per trade
        "reward_function": "sharpe",     # sharpe, calmar, sortino, simple
        "lookback_window": 20,          # Technical indicator lookback
    }
    
    if config:
        default_config.update(config)
    
    env = BestPracticeForexEnv(df=df, config=default_config)
    return env


def train_agent(env, total_timesteps=100000, save_path="./trained_models/forex_ppo_complete"):
    """Train PPO agent on the environment"""
    print(f"\n=== Training PPO Agent ({total_timesteps} timesteps) ===")
    
    try:
        # Wrap environment for Stable-Baselines3
        vec_env = DummyVecEnv([lambda: env])
        
        # Create PPO agent
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_log/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        
        print("✓ PPO agent created")
        print("Training started...")
        
        # Train the agent
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # Save the model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"✓ Model saved to {save_path}")
        
        return model
        
    except Exception as e:
        print(f"✗ Error training agent: {str(e)}")
        raise


def backtest_model(model_path, env, num_episodes=10):
    """Backtest the trained model on the environment"""
    print(f"\n=== Backtesting Model ({num_episodes} episodes) ===")
    
    try:
        # Load the trained model
        model = PPO.load(model_path)
        print("✓ Model loaded successfully")
        
        # Backtest
        total_rewards = []
        final_balances = []
        equity_curves = []
        trade_counts = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            step = 0
            episode_equity = []
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
                
                # Record equity
                balance = info.get('balance', env.balance)
                episode_equity.append(balance)
                
                if step % 100 == 0:
                    print(f"  Episode {episode + 1}, Step {step}, Balance: ${balance:.2f}")
            
            final_balance = info.get('balance', env.balance)
            total_rewards.append(total_reward)
            final_balances.append(final_balance)
            equity_curves.append(episode_equity)
            trade_counts.append(info.get('trade_count', 0))
            
            print(f"Episode {episode + 1}: Reward: {total_reward:.2f}, Final Balance: ${final_balance:.2f}")
        
        # Calculate statistics
        results = {
            'total_rewards': total_rewards,
            'final_balances': final_balances,
            'equity_curves': equity_curves,
            'trade_counts': trade_counts,
            'avg_reward': np.mean(total_rewards),
            'avg_balance': np.mean(final_balances),
            'std_reward': np.std(total_rewards),
            'sharpe_ratio': np.mean(total_rewards) / np.std(total_rewards) if np.std(total_rewards) > 0 else 0,
            'max_drawdown': calculate_max_drawdown(equity_curves),
            'win_rate': calculate_win_rate(total_rewards)
        }
        
        print(f"\nBacktest Results ({num_episodes} episodes):")
        print(f"  Average Total Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Average Final Balance: ${results['avg_balance']:.2f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"  Win Rate: {results['win_rate']:.2%}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error backtesting: {str(e)}")
        raise


def calculate_max_drawdown(equity_curves):
    """Calculate maximum drawdown from equity curves"""
    max_drawdown = 0
    for curve in equity_curves:
        peak = curve[0]
        for value in curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def calculate_win_rate(rewards):
    """Calculate win rate (positive rewards)"""
    positive_rewards = sum(1 for r in rewards if r > 0)
    return positive_rewards / len(rewards) if rewards else 0


def compare_timeframes(model_path, timeframes, start_date, end_date):
    """Compare model performance across different timeframes"""
    print(f"\n=== Comparing Performance Across Timeframes ===")
    
    results_comparison = {}
    
    for timeframe in timeframes:
        print(f"\n--- Testing {timeframe} timeframe ---")
        
        try:
            # Download data for this timeframe
            df = download_backtest_data(timeframe, start_date, end_date)
            
            # Create environment
            env = create_environment(df)
            
            # Backtest
            results = backtest_model(model_path, env, num_episodes=5)
            
            results_comparison[timeframe] = results
            
        except Exception as e:
            print(f"✗ Error testing {timeframe}: {str(e)}")
            continue
    
    return results_comparison


def plot_comparison_results(results_comparison):
    """Plot comparison results across timeframes"""
    print("\n=== Plotting Comparison Results ===")
    
    try:
        # Prepare data for plotting
        timeframes = list(results_comparison.keys())
        metrics = ['avg_reward', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison Across Timeframes', fontsize=16)
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            values = [results_comparison[tf][metric] for tf in timeframes]
            
            if metric == 'max_drawdown':
                # Convert to percentage for better visualization
                values = [v * 100 for v in values]
                ylabel = f'{metric.replace("_", " ").title()} (%)'
            elif metric == 'win_rate':
                values = [v * 100 for v in values]
                ylabel = f'{metric.replace("_", " ").title()} (%)'
            else:
                ylabel = metric.replace("_", " ").title()
            
            bars = ax.bar(timeframes, values, color='skyblue', alpha=0.7)
            ax.set_title(ylabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs("./results", exist_ok=True)
        plt.savefig("./results/timeframe_comparison.png", dpi=300, bbox_inches='tight')
        print("✓ Comparison plot saved to ./results/timeframe_comparison.png")
        
        plt.show()
        
    except Exception as e:
        print(f"✗ Error plotting results: {str(e)}")


def main():
    """Main function demonstrating the complete MT5 training and backtesting pipeline"""
    print("🚀 Complete MT5 Training and Backtesting Pipeline")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("./trained_models", exist_ok=True)
    os.makedirs("./tensorboard_log", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    try:
        # Step 1: Download training data
        train_df = download_training_data()
        
        # Step 2: Create training environment
        print("\n=== Creating Training Environment ===")
        train_env = create_environment(train_df)
        print("✓ Training environment created")
        
        # Step 3: Train the agent
        model = train_agent(train_env, total_timesteps=50000)
        
        # Step 4: Define backtesting parameters
        backtest_start = "2023-07-01"
        backtest_end = "2023-12-31"
        timeframes_to_test = ["15m", "30m", "1h", "4h"]
        
        # Step 5: Compare performance across timeframes
        results_comparison = compare_timeframes(
            "./trained_models/forex_ppo_complete",
            timeframes_to_test,
            backtest_start,
            backtest_end
        )
        
        # Step 6: Plot comparison results
        plot_comparison_results(results_comparison)
        
        print("\n🎉 Complete MT5 Training and Backtesting Pipeline Completed!")
        print("\nSummary:")
        print(f"  Training data: {train_df.shape[0]} records")
        print(f"  Training timeframe: 1h")
        print(f"  Backtesting timeframes: {timeframes_to_test}")
        print(f"  Backtesting period: {backtest_start} to {backtest_end}")
        
        print("\nNext Steps:")
        print("1. Monitor training: tensorboard --logdir ./tensorboard_log")
        print("2. Experiment with different hyperparameters")
        print("3. Test on live data (with proper risk management)")
        print("4. Implement ensemble methods for better performance")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
