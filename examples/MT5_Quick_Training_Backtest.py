#!/usr/bin/env python3
"""
Quick MT5 Training and Backtesting for FinRL-Meta

A simplified script to quickly train on historical data and backtest on different timeframes.
Perfect for getting started with MT5 data processor.

Usage:
    python MT5_Quick_Training_Backtest.py
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

from meta.data_processors.mt5 import MT5 as MT5Processor
from examples.Best_Practice_Forex_Env import BestPracticeForexEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import torch


def quick_mt5_pipeline():
    """Quick MT5 training and backtesting pipeline"""
    print("🚀 Quick MT5 Training and Backtesting Pipeline")
    print("=" * 50)
    
    # Create directories
    os.makedirs("./trained_models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    try:
        # Step 1: Download Training Data
        print("\n=== Step 1: Downloading Training Data ===")
        start_date = "2020-01-01"
        end_date = "2024-07-31"
        
        print(f"Training period: {start_date} to {end_date}")
        print("Timeframe: 1h")
        
        train_processor = MT5Processor(
            data_source="mt5",
            start_date=start_date,
            end_date=end_date,
            time_interval="1h",
            ticker_list=["EURUSD", "GBPUSD", "USDJPY"]
        )
        train_processor.download_data(ticker_list=["EURUSD", "GBPUSD", "USDJPY"])
        train_processor.add_technical_indicator(
            tech_indicator_list=["macd", "rsi", "boll_ub", "boll_lb", "atr", "ema_20"]
        )
        
        print(f"✓ Training data ready: {len(train_processor.dataframe)} records")
        
        # Step 2: Create training environment
        print("\n=== Step 2: Creating Training Environment ===")
        train_env = BestPracticeForexEnv(
            df=train_processor.dataframe,
            config={
                "initial_balance": 100000,
                "max_position_size": 0.05,        # Max 5% of balance per trade (more conservative)
                "transaction_fee": 0.0002,         # 2 pips spread (more realistic)
                "slippage": 0.0001,               # 1 pip slippage
                "max_drawdown": 0.15,             # 15% max drawdown (more conservative)
                "risk_per_trade": 0.01,           # 1% risk per trade
                "commission_per_lot": 4.0,        # $4 commission per lot
                "max_leverage": 30,               # Maximum portfolio leverage
                "reward_function": "sharpe",       # sharpe, calmar, sortino, simple
                "lookback_window": 20             # Technical indicator lookback
            }
        )
        print("✓ Training environment created")
        
        # Step 3: Train PPO Agent
        print("\n=== Step 3: Training PPO Agent ===")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                # Test CUDA works
                torch.zeros(1).cuda()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ GPU detected: {gpu_name}")
                
                # Check if it's RTX 5090 (sm_120) which has compatibility issues
                if "RTX 5090" in gpu_name:
                    print("⚠️ RTX 5090 detected - this GPU has CUDA compatibility issues with current PyTorch")
                    print("   Forcing CPU usage to avoid CUDA errors")
                    device = "cpu"
                    
            except Exception as e:
                print(f"⚠️ GPU detected, but CUDA is not working. Forcing CPU. Error: {e}")
                device = "cpu"
        else:
            print("ℹ️ No GPU detected, using CPU.")
            
        print(f"Using device: {device}")
            
        # PPO model hyperparameters
        model_params = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "gamma": 0.99
        }
        
        # Create the PPO agent
        model = PPO(
            "MlpPolicy",
            train_env,
            tensorboard_log="./tensorboard_log/",
            verbose=1,
            device=device,
            **model_params
        )

        # Train the model
        train_timesteps = 500000  # Increased training time
        print(f"Training for {train_timesteps:,} timesteps...")
        model.learn(total_timesteps=train_timesteps, progress_bar=True)
        
        # Save the trained model
        model_path = "./trained_models/forex_ppo_long_train.zip"
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Step 4: Backtest on different timeframes
        print("\n=== Step 4: Backtesting on Different Timeframes ===")
        
        # Define backtesting period (last 3 months)
        backtest_end = datetime.now().strftime("%Y-%m-%d")
        backtest_start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        # Test different timeframes
        timeframes = ["15m", "30m", "1h"]
        results = {}
        
        for tf in timeframes:
            print(f"\n--- Testing {tf} timeframe ---")
            
            try:
                # Download backtest data
                backtest_processor = MT5Processor(
                    start_date=backtest_start,
                    end_date=backtest_end,
                    time_interval=tf
                )
                
                backtest_processor.download_data(ticker_list=["EURUSD", "GBPUSD", "USDJPY"])
                backtest_processor.clean_data()
                backtest_processor.fillna()
                backtest_processor.add_technical_indicator(tech_indicator_list=["macd", "rsi", "boll_ub", "boll_lb", "atr", "ema_20"])
                
                # Create backtest environment
                backtest_env = BestPracticeForexEnv(
                    df=backtest_processor.dataframe,
                    config={
                        "initial_balance": 100000,
                        "max_position_size": 0.05,        # Max 5% of balance per trade (more conservative)
                        "transaction_fee": 0.0002,         # 2 pips spread (more realistic)
                        "slippage": 0.0001,               # 1 pip slippage
                        "max_drawdown": 0.15,             # 15% max drawdown (more conservative)
                        "risk_per_trade": 0.01,           # 1% risk per trade (more conservative)
                        "max_leverage": 30,               # Maximum portfolio leverage
                        "reward_function": "sharpe",       # sharpe, calmar, sortino, simple
                        "lookback_window": 20             # Technical indicator lookback
                    }
                )
                
                # Run backtest
                backtest_results = run_backtest(model, backtest_env, num_episodes=3)
                results[tf] = backtest_results
                
                print(f"✓ {tf} backtest completed")
                
            except Exception as e:
                print(f"✗ Error testing {tf}: {str(e)}")
                continue
        
        # Step 5: Display results
        print("\n=== Step 5: Results Summary ===")
        display_results(results)
        
        # Step 6: Save results
        save_results(results)
        
        print("\n🎉 Quick MT5 Pipeline Completed!")
        print("\nNext steps:")
        print("1. Experiment with different hyperparameters")
        print("2. Test on longer periods")
        print("3. Try different reward functions")
        print("4. Implement risk management rules")
        
        # GPU troubleshooting information
        if torch.cuda.is_available():
            print("\n🔧 GPU Troubleshooting for RTX 5090:")
            print("Your RTX 5090 is detected but has compatibility issues with current PyTorch.")
            print("To enable GPU acceleration:")
            print("1. Wait for official PyTorch support for RTX 5090 (sm_120)")
            print("2. Try building PyTorch from source with CUDA 12.9+ support")
            print("3. Use PyTorch nightly builds when they add RTX 5090 support")
            print("4. Consider using TensorFlow/Keras as alternative (may have better RTX 5090 support)")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()


def run_backtest(model, env, num_episodes=3):
    """Run backtest on the environment"""
    total_rewards = []
    final_balances = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        final_balance = info.get('balance', env.balance)
        total_rewards.append(total_reward)
        final_balances.append(final_balance)
        
        print(f"  Episode {episode + 1}: Reward: {total_reward:.2f}, Final Balance: ${final_balance:.2f}")
    
    return {
        'avg_reward': np.mean(total_rewards),
        'avg_balance': np.mean(final_balances),
        'sharpe_ratio': np.mean(total_rewards) / np.std(total_rewards) if np.std(total_rewards) > 0 else 0,
        'win_rate': sum(1 for r in total_rewards if r > 0) / len(total_rewards)
    }


def display_results(results):
    """Display backtest results"""
    print("\nTimeframe Comparison:")
    print("-" * 50)
    
    for tf, result in results.items():
        print(f"{tf:>6} | "
              f"Avg Reward: {result['avg_reward']:>8.2f} | "
              f"Avg Balance: ${result['avg_balance']:>8.0f} | "
              f"Sharpe: {result['sharpe_ratio']:>6.3f} | "
              f"Win Rate: {result['win_rate']:>6.1%}")


def save_results(results):
    """Save results to file"""
    try:
        # Convert to DataFrame
        df_results = pd.DataFrame(results).T
        df_results.index.name = 'timeframe'
        
        # Save to CSV
        results_file = "./results/backtest_results.csv"
        df_results.to_csv(results_file)
        print(f"\n✓ Results saved to {results_file}")
        
        # Create simple plot
        create_results_plot(results)
        
    except Exception as e:
        print(f"✗ Error saving results: {str(e)}")


def create_results_plot(results):
    """Create a simple results plot"""
    try:
        timeframes = list(results.keys())
        metrics = ['avg_reward', 'sharpe_ratio', 'win_rate']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Backtest Results by Timeframe', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [results[tf][metric] for tf in timeframes]
            
            if metric == 'win_rate':
                values = [v * 100 for v in values]
                ylabel = 'Win Rate (%)'
            else:
                ylabel = metric.replace('_', ' ').title()
            
            bars = ax.bar(timeframes, values, color='lightblue', alpha=0.7)
            ax.set_title(ylabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = "./results/backtest_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {plot_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"✗ Error creating plot: {str(e)}")


if __name__ == "__main__":
    quick_mt5_pipeline()
