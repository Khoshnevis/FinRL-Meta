#!/usr/bin/env python3
"""
MT5 1-Minute Training Script for FinRL-Meta

Simple script to train on 3 months of 1-minute candle data.
This script automatically handles GPU issues and falls back to CPU if needed.

Usage:
    python MT5_1min_Training.py
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


def setup_device():
    """Setup GPU/CPU device with automatic fallback"""
    print("🔍 Setting up device...")
    
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            test_tensor = torch.zeros(1).cuda()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            
            # Check for known problematic GPUs
            problematic_gpus = ["RTX 5090", "RTX 4090 Ti", "RTX 4090 Super"]
            for problematic in problematic_gpus:
                if problematic in gpu_name:
                    print("⚠️  Known problematic GPU detected - using CPU")
                    return "cpu"
            
            # Test memory allocation
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                print("✅ GPU memory test passed - using GPU")
                return "cuda:0"
            except Exception as e:
                print(f"⚠️  GPU memory test failed: {e} - using CPU")
                return "cpu"
                
        except Exception as e:
            print(f"⚠️  CUDA test failed: {e} - using CPU")
            return "cpu"
    else:
        print("ℹ️  No GPU detected - using CPU")
        return "cpu"


def download_1min_data():
    """Download 3 months of 1-minute data"""
    print("\n📥 Downloading 3 months of 1-minute data...")
    
    # Use a more conservative approach - download in smaller chunks
    # Start with smaller numbers and work our way up
    chunk_sizes = [1000, 5000, 10000, 20000, 50000]
    
    print("   Timeframe: 1 minute")
    print("   Symbols: EURUSD, GBPUSD, AUDUSD")
    print("   Strategy: Downloading data in chunks to find optimal size")
    
    try:
        # Create a simple data processor that works with available data
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta
        
        print("   Initializing MT5...")
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5")
        print("   ✅ MT5 initialized successfully")
        
        final_df = None
        symbols = ["EURUSD", "GBPUSD", "AUDUSD"]
        
        for symbol in symbols:
            print(f"   Downloading {symbol}...")
            
            # Try different chunk sizes
            symbol_data = None
            for chunk_size in chunk_sizes:
                try:
                    print(f"     Trying {chunk_size:,} records...")
                    
                    # Use a specific datetime object
                    current_time = datetime.now()
                    
                    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, current_time, chunk_size)
                    
                    if rates is not None and len(rates) > 0:
                        print(f"     ✅ Successfully downloaded {len(rates):,} records")
                        symbol_data = rates
                        break
                    else:
                        print(f"     ⚠️ No data for {chunk_size:,} records, trying smaller size...")
                        
                except Exception as e:
                    print(f"     ⚠️ Error with {chunk_size:,} records: {e}")
                    continue
            
            if symbol_data is not None and len(symbol_data) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(symbol_data)
                df['tic'] = symbol
                
                # Rename columns to match expected format
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
                
                # Sort by time
                df = df.sort_values('time').reset_index(drop=True)
                
                if final_df is None:
                    final_df = df
                else:
                    final_df = pd.concat([final_df, df], axis=0, ignore_index=True)
                
                print(f"     ✅ Processed {len(df):,} records for {symbol}")
                print(f"        Date range: {df['time'].min().strftime('%Y-%m-%d %H:%M')} to {df['time'].max().strftime('%Y-%m-%d %H:%M')}")
                
                # Calculate actual time span
                time_span = df['time'].max() - df['time'].min()
                days_span = time_span.total_seconds() / (24 * 3600)
                months_span = days_span / 30
                print(f"        Time span: {days_span:.1f} days ({months_span:.1f} months)")
            else:
                print(f"     ❌ No data found for {symbol} with any chunk size")
                
        print("   Shutting down MT5...")
        mt5.shutdown()
        
        if final_df is None or len(final_df) == 0:
            raise Exception("No data was downloaded from MT5")
        
        # Add technical indicators
        print("   Adding technical indicators...")
        
        # Simple technical indicators
        for symbol in symbols:
            symbol_data = final_df[final_df['tic'] == symbol].copy()
            if len(symbol_data) > 0:
                # Calculate simple moving averages
                symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
                symbol_data['sma_50'] = symbol_data['close'].rolling(window=50).mean()
                
                # Calculate RSI
                delta = symbol_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                symbol_data['rsi'] = 100 - (100 / (1 + rs))
                
                # Calculate MACD
                exp1 = symbol_data['close'].ewm(span=12).mean()
                exp2 = symbol_data['close'].ewm(span=26).mean()
                symbol_data['macd'] = exp1 - exp2
                symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
                
                # Update the main dataframe
                final_df.loc[final_df['tic'] == symbol, 'sma_20'] = symbol_data['sma_20']
                final_df.loc[final_df['tic'] == symbol, 'sma_50'] = symbol_data['sma_50']
                final_df.loc[final_df['tic'] == symbol, 'rsi'] = symbol_data['rsi']
                final_df.loc[final_df['tic'] == symbol, 'macd'] = symbol_data['macd']
                final_df.loc[final_df['tic'] == symbol, 'macd_signal'] = symbol_data['macd_signal']
        
        # Fill NaN values
        final_df = final_df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Data downloaded: {len(final_df)} records")
        print(f"   Shape: {final_df.shape}")
        print(f"   Columns: {list(final_df.columns)}")
        
        return final_df
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_environment(df):
    """Create trading environment"""
    print("\n🏗️  Creating trading environment...")
    
    config = {
        "initial_balance": 10000,        # Starting capital
        # Fix 5: Dynamic Costs Configuration
        "transaction_fee_atr_factor": 0.05, # Spread cost as a % of ATR (Dynamic)
        "slippage_atr_factor": 0.02,     # Slippage as a % of ATR (Dynamic)
        "commission_per_lot": 5.0,       # Fix 5: Realistic commission ($5)
        "max_drawdown": 0.20,            # 20% max drawdown (Equity based)
        "risk_per_trade": 0.01,          # 1% risk per trade (Equity based)
        "max_leverage": 30,              # Max portfolio leverage
        # Fix 3: Reward function selection: 'simple', 'drawdown_penalty', 'dsr'
        "reward_function": "drawdown_penalty", 
        "drawdown_penalty_factor": 10,   # Hyperparameter for drawdown penalty
        "dsr_gamma": 0.99,               # Discount factor (EMA decay) for DSR estimation
    }
    
    env = BestPracticeForexEnv(df=df, config=config)
    print("✅ Environment created successfully")
    return env


def train_model(env, device):
    """Train PPO model"""
    print(f"\n🚀 Starting training on {device}...")
    
    try:
        # Wrap environment for Stable-Baselines3
        vec_env = DummyVecEnv([lambda: env])
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_log/",
            device=device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256]
                ),
                activation_fn=torch.nn.ReLU,
            )
        )
        
        print("✅ PPO model created")
        print("🎯 Training started...")
        
        # Train the model
        total_timesteps = 500000
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        # Save the model
        model_path = f"./trained_models/forex_ppo_1min_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        
        print(f"✅ Training completed!")
        print(f"✅ Model saved to {model_path}")
        
        return model, model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def run_quick_backtest(model, env):
    """Run a quick backtest"""
    print("\n🧪 Running quick backtest...")
    
    try:
        total_rewards = []
        final_balances = []
        
        for episode in range(3):
            print(f"   Episode {episode + 1}/3")
            
            # Fix: Handle tuple return from env.reset()
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            done = False
            total_reward = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                
                # Fix: Handle tuple return from env.step()
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                
                total_reward += reward
            
            final_balance = info.get('balance', getattr(env, 'balance', 100000))
            total_rewards.append(total_reward)
            final_balances.append(final_balance)
            
            print(f"      Reward: {total_reward:.2f}, Final Balance: ${final_balance:.2f}")
        
        # Calculate metrics
        avg_reward = np.mean(total_rewards)
        avg_balance = np.mean(final_balances)
        win_rate = sum(1 for r in total_rewards if r > 0) / len(total_rewards)
        
        print(f"\n📊 Backtest Results:")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Average Final Balance: ${avg_balance:.2f}")
        print(f"   Win Rate: {win_rate:.1%}")
        
        return {
            'avg_reward': avg_reward,
            'avg_balance': avg_balance,
            'win_rate': win_rate
        }
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_backtest_on_trained_model(model_path, df):
    """Run backtest on a trained model using the same environment"""
    print(f"\n🧪 Running backtest on trained model: {model_path}")
    
    try:
        # Create the same environment used for training
        env = create_environment(df)
        
        # Load the trained model and force CPU usage (since it was trained on CPU)
        model = PPO.load(model_path, device="cpu")
        print("✅ Model loaded successfully on CPU")
        
        total_rewards = []
        final_balances = []
        
        for episode in range(5):  # Run 5 episodes
            print(f"   Episode {episode + 1}/5")
            
            # Handle tuple return from env.reset()
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            done = False
            total_reward = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                
                # Handle tuple return from env.step()
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                
                total_reward += reward
            
            final_balance = info.get('balance', getattr(env, 'balance', 100000))
            total_rewards.append(total_reward)
            final_balances.append(final_balance)
            
            print(f"      Reward: {total_reward:.2f}, Final Balance: ${final_balance:.2f}")
        
        # Calculate metrics
        avg_reward = np.mean(total_rewards)
        avg_balance = np.mean(final_balances)
        win_rate = sum(1 for r in total_rewards if r > 0) / len(total_rewards)
        
        print(f"\n📊 Backtest Results:")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Average Final Balance: ${avg_balance:.2f}")
        print(f"   Win Rate: {win_rate:.1%}")
        
        return {
            'avg_reward': avg_reward,
            'avg_balance': avg_balance,
            'win_rate': win_rate
        }
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main training pipeline"""
    print("🚀 MT5 1-Minute Training Pipeline")
    print("=" * 50)
    
    try:
        # Create directories
        os.makedirs("./trained_models", exist_ok=True)
        os.makedirs("./tensorboard_log", exist_ok=True)
        
        # Step 1: Setup device
        device = setup_device()
        print(f"📱 Using device: {device}")
        
        # Step 2: Download data
        df = download_1min_data()
        
        # Step 3: Create environment
        env = create_environment(df)
        
        # Step 4: Train model
        model, model_path = train_model(env, device)
        
        # Step 5: Quick backtest
        backtest_results = run_quick_backtest(model, env)
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"\n📊 Results:")
        if backtest_results:
            print(f"   Average Reward: {backtest_results['avg_reward']:.2f}")
            print(f"   Average Balance: ${backtest_results['avg_balance']:.2f}")
            print(f"   Win Rate: {backtest_results['win_rate']:.1%}")
        
        print(f"\n🤖 Model saved to: {model_path}")
        print(f"📈 Tensorboard logs: ./tensorboard_log/")
        print(f"\n💡 Next steps:")
        print(f"   1. Run backtest on trained model: python -c \"from MT5_1min_Training import run_backtest_on_trained_model, download_1min_data; df = download_1min_data(); run_backtest_on_trained_model('{model_path}', df)\"")
        print(f"   2. Monitor training: tensorboard --logdir ./tensorboard_log/")
        print(f"   3. Experiment with different hyperparameters")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
