#!/usr/bin/env python3
"""
MT5 Forex RL Training Example for FinRL-Meta

This example demonstrates the complete pipeline:
1. Download historical forex data from MT5
2. Process and prepare data for RL training
3. Create and configure the forex trading environment
4. Train a PPO agent on historical forex data
5. Evaluate and test the trained agent

Requirements:
- MetaTrader 5 terminal installed and running
- FinRL-Meta framework
- Stable-Baselines3 for PPO implementation
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import FinRL-Meta modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta.data_processor import DataProcessor, DataSource
from meta.env_fx_trading.env_fx import tgym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import matplotlib.pyplot as plt
import torch


def download_and_prepare_mt5_data():
    """Download and prepare MT5 forex data for RL training"""
    print("=== Step 1: Downloading and Preparing MT5 Data ===\n")
    
    # Set date range (last 6 months for sufficient training data)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        # Initialize MT5 data processor
        processor = DataProcessor(
            data_source=DataSource.mt5,
            start_date=start_date,
            end_date=end_date,
            time_interval="1h"  # 1-hour candles for good balance
        )
        print("✓ MT5 Data Processor initialized")
        
        # Download major forex pairs
        forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        print(f"Downloading data for: {forex_pairs}")
        
        processor.download_data(ticker_list=forex_pairs)
        print("✓ Forex data downloaded successfully")
        
        # Clean and process data
        processor.clean_data()
        processor.fillna()
        print("✓ Data cleaned and processed")
        
        # Add technical indicators for better state representation
        tech_indicators = [
            "macd", "rsi", "cci", "dx", 
            "boll_ub", "boll_lb", "close_30_sma", "close_10_sma"
        ]
        processor.add_technical_indicator(tech_indicator_list=tech_indicators)
        print("✓ Technical indicators added")
        
        print(f"Final dataset shape: {processor.dataframe.shape}")
        print(f"Columns: {list(processor.dataframe.columns)}")
        
        return processor.dataframe
        
    except Exception as e:
        print(f"✗ Error preparing data: {str(e)}")
        raise


def create_forex_environment(df, config_file=None):
    """Create and configure the forex trading environment"""
    print("\n=== Step 2: Creating Forex Trading Environment ===\n")
    
    try:
        # Create environment configuration
        env_config = {
            "initial_balance": 10000,  # Starting capital in USD
            "max_position": 1.0,       # Maximum position size
            "transaction_fee": 0.001,  # 0.1% transaction fee
            "stop_loss": 0.02,         # 2% stop loss
            "take_profit": 0.03,      # 3% take profit
            "max_steps": len(df) // 5, # Maximum steps per episode
            "reward_scaling": 1.0,     # Reward scaling factor
        }
        
        print("Environment configuration:")
        for key, value in env_config.items():
            print(f"  {key}: {value}")
        
        # Create the forex environment
        env = tgym(
            df=df,
            env_config=env_config
        )
        
        print("✓ Forex trading environment created successfully")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        
        return env, env_config
        
    except Exception as e:
        print(f"✗ Error creating environment: {str(e)}")
        raise


def train_ppo_agent(env, total_timesteps=100000):
    """Train a PPO agent on the forex environment"""
    print("\n=== Step 3: Training PPO Agent ===\n")
    
    try:
        # Wrap environment for vectorization
        vec_env = DummyVecEnv([lambda: env])
        
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: env])
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./trained_models/",
            log_path="./tensorboard_log/",
            eval_freq=max(total_timesteps // 10, 1),
            deterministic=True,
            render=False
        )
        
        # Initialize PPO agent with optimized hyperparameters for forex
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,        # Lower learning rate for stability
            n_steps=2048,              # Larger batch size
            batch_size=64,             # Batch size for training
            n_epochs=10,               # Multiple epochs per update
            gamma=0.99,                # Discount factor
            gae_lambda=0.95,           # GAE lambda
            clip_range=0.2,            # PPO clip range
            clip_range_vf=None,        # No value function clipping
            normalize_advantage=True,   # Normalize advantages
            ent_coef=0.01,             # Entropy coefficient for exploration
            vf_coef=0.5,               # Value function coefficient
            max_grad_norm=0.5,         # Gradient clipping
            use_sde=True,              # State-dependent exploration
            sde_sample_freq=4,         # SDE sampling frequency
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],     # Policy network architecture
                    vf=[256, 256]      # Value function network architecture
                ),
                activation_fn=torch.nn.ReLU,
                log_std_init=-2,       # Initial log standard deviation
            ),
            verbose=1,
            tensorboard_log="./tensorboard_log/"
        )
        
        print("✓ PPO agent initialized with forex-optimized hyperparameters")
        print(f"Training for {total_timesteps} timesteps...")
        
        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        print("✓ PPO agent training completed")
        
        # Save the final model
        model.save("./trained_models/forex_ppo_final")
        print("✓ Final model saved to ./trained_models/forex_ppo_final")
        
        return model
        
    except Exception as e:
        print(f"✗ Error training agent: {str(e)}")
        raise


def evaluate_agent(model, env, num_episodes=5):
    """Evaluate the trained agent"""
    print("\n=== Step 4: Evaluating Trained Agent ===\n")
    
    try:
        total_rewards = []
        final_balances = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                step += 1
                
                if step % 100 == 0:
                    print(f"  Episode {episode + 1}, Step {step}, Balance: ${info.get('balance', 0):.2f}")
            
            final_balance = info.get('balance', 0)
            total_rewards.append(total_reward)
            final_balances.append(final_balance)
            
            print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Final Balance: ${final_balance:.2f}")
        
        # Calculate statistics
        avg_reward = np.mean(total_rewards)
        avg_balance = np.mean(final_balances)
        std_reward = np.std(total_rewards)
        
        print(f"\nEvaluation Results ({num_episodes} episodes):")
        print(f"  Average Total Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Final Balance: ${avg_balance:.2f}")
        print(f"  Best Episode Reward: {max(total_rewards):.2f}")
        print(f"  Worst Episode Reward: {min(total_rewards):.2f}")
        
        return total_rewards, final_balances
        
    except Exception as e:
        print(f"✗ Error evaluating agent: {str(e)}")
        raise


def plot_training_results(rewards, balances):
    """Plot training and evaluation results"""
    print("\n=== Step 5: Plotting Results ===\n")
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot rewards
        ax1.plot(rewards, 'b-o', linewidth=2, markersize=8)
        ax1.set_title('Episode Rewards', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=np.mean(rewards), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(rewards):.2f}')
        ax1.legend()
        
        # Plot final balances
        ax2.plot(balances, 'g-o', linewidth=2, markersize=8)
        ax2.set_title('Final Account Balances', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Final Balance ($)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=np.mean(balances), color='r', linestyle='--', 
                    label=f'Mean: ${np.mean(balances):.2f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('./results/forex_training_results.png', dpi=300, bbox_inches='tight')
        print("✓ Results plot saved to ./results/forex_training_results.png")
        
        plt.show()
        
    except Exception as e:
        print(f"✗ Error plotting results: {str(e)}")


def main():
    """Main function demonstrating the complete MT5 Forex RL pipeline"""
    print("🚀 MT5 Forex RL Training Pipeline")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("./trained_models", exist_ok=True)
    os.makedirs("./tensorboard_log", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    try:
        # Step 1: Download and prepare MT5 data
        df = download_and_prepare_mt5_data()
        
        # Step 2: Create forex environment
        env, env_config = create_forex_environment(df)
        
        # Step 3: Train PPO agent
        model = train_ppo_agent(env, total_timesteps=50000)  # Adjust timesteps as needed
        
        # Step 4: Evaluate the trained agent
        rewards, balances = evaluate_agent(model, env, num_episodes=5)
        
        # Step 5: Plot results
        plot_training_results(rewards, balances)
        
        print("\n🎉 MT5 Forex RL Training Pipeline Completed Successfully!")
        print("\nNext Steps:")
        print("1. Monitor training progress: tensorboard --logdir ./tensorboard_log")
        print("2. Load trained model: model = PPO.load('./trained_models/forex_ppo_final')")
        print("3. Deploy for live trading (with proper risk management)")
        print("4. Experiment with different hyperparameters and timeframes")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
