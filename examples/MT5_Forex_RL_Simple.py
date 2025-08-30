#!/usr/bin/env python3
"""
MT5 Forex RL Training - Simple Version

A simplified version that demonstrates the core workflow:
1. Download MT5 forex data
2. Create forex environment
3. Train a PPO agent
4. Evaluate results

This version is easier to run and understand.
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
from meta.env_fx_trading.env_fx import tgym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    """Simple MT5 Forex RL Training Example"""
    print("🚀 MT5 Forex RL Training - Simple Version")
    print("=" * 50)
    
    # Create directories
    os.makedirs("./trained_models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    try:
        # Step 1: Download MT5 data (last 3 months)
        print("\n=== Step 1: Downloading MT5 Data ===")
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        print(f"Date range: {start_date} to {end_date}")
        
        processor = DataProcessor(
            data_source=DataSource.mt5,
            start_date=start_date,
            end_date=end_date,
            time_interval="1h"
        )
        
        # Download major forex pairs
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY"]
        processor.download_data(ticker_list=forex_pairs)
        print("✓ Data downloaded successfully")
        
        # Clean data
        processor.clean_data()
        processor.fillna()
        
        # Add basic technical indicators
        processor.add_technical_indicator(["macd", "rsi", "boll_ub", "boll_lb"])
        print("✓ Technical indicators added")
        
        # Get the dataframe with technical indicators
        df = processor.dataframe.copy()
        
        # Add weekday column required by tgym environment
        df['weekday'] = pd.to_datetime(df['time']).dt.weekday
        
        # Rename columns to match tgym environment expectations
        df = df.rename(columns={
            'tic': 'symbol',
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close'
        })
        print("✓ Weekday column added")
        print("✓ Column names adjusted for tgym compatibility")
        print(f"Final dataset: {df.shape[0]} records, {df.shape[1]} columns")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Step 2: Create forex environment
        print("\n=== Step 2: Creating Forex Environment ===")
        
        env_config = {
            "initial_balance": 10000,
            "max_position": 1.0,
            "transaction_fee": 0.001,
            "stop_loss": 0.02,
            "take_profit": 0.03,
            "max_steps": len(df) // 10,
            "reward_scaling": 1.0,
        }
        
        env = tgym(df=df, env_config_file="./examples/mt5_forex_config.json")
        print("✓ Forex environment created")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        # Step 3: Train PPO agent
        print("\n=== Step 3: Training PPO Agent ===")
        
        # Wrap environment for Stable-Baselines3
        vec_env = DummyVecEnv([lambda: env])
        
        # Create PPO agent with default hyperparameters
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_log/"
        )
        
        print("✓ PPO agent created")
        print("Training for 20,000 timesteps...")
        
        # Train the agent
        model.learn(total_timesteps=20000, progress_bar=True)
        
        # Save the model
        model.save("./trained_models/forex_ppo_simple")
        print("✓ Model saved to ./trained_models/forex_ppo_simple")
        
        # Step 4: Evaluate the agent
        print("\n=== Step 4: Evaluating Agent ===")
        
        total_rewards = []
        final_balances = []
        
        for episode in range(3):  # Test 3 episodes
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            final_balance = info.get('balance', 0)
            total_rewards.append(total_reward)
            final_balances.append(final_balance)
            
            print(f"Episode {episode + 1}: Reward: {total_reward:.2f}, Final Balance: ${final_balance:.2f}")
        
        # Results summary
        print(f"\n=== Results Summary ===")
        print(f"Average Reward: {np.mean(total_rewards):.2f}")
        print(f"Average Final Balance: ${np.mean(final_balances):.2f}")
        print(f"Best Episode: {max(total_rewards):.2f}")
        
        print("\n🎉 Training completed successfully!")
        print("\nNext steps:")
        print("1. Load model: model = PPO.load('./trained_models/forex_ppo_simple')")
        print("2. Test on different data")
        print("3. Adjust hyperparameters")
        print("4. Monitor with tensorboard: tensorboard --logdir ./tensorboard_log")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
