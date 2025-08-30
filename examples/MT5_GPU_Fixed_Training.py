#!/usr/bin/env python3
"""
MT5 GPU-Fixed Training and Backtesting Script for FinRL-Meta

This script addresses GPU compatibility issues and provides:
1. Automatic GPU detection and fallback to CPU if needed
2. Training with 3 months of 1-minute candle data
3. Backtesting on market data after the training period
4. Comprehensive error handling and GPU troubleshooting

Usage:
    python MT5_GPU_Fixed_Training.py --timeframe 1m --training_months 3 --backtest_months 1
"""

import os
import sys
import argparse
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
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import torch
import json


class GPUManager:
    """Manages GPU detection and compatibility issues"""
    
    @staticmethod
    def detect_gpu():
        """Detect and test GPU compatibility"""
        if not torch.cuda.is_available():
            return "cpu", "No CUDA available"
        
        try:
            # Test basic CUDA functionality
            test_tensor = torch.zeros(1).cuda()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Check for known problematic GPUs
            problematic_gpus = ["RTX 5090", "RTX 4090 Ti", "RTX 4090 Super"]
            for problematic in problematic_gpus:
                if problematic in gpu_name:
                    return "cpu", f"Known problematic GPU: {gpu_name}"
            
            # Test memory allocation
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                return f"cuda:0", f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
            except Exception as e:
                return "cpu", f"GPU memory test failed: {str(e)}"
                
        except Exception as e:
            return "cpu", f"CUDA test failed: {str(e)}"
    
    @staticmethod
    def setup_device(device_str):
        """Setup the device with proper error handling"""
        if device_str.startswith("cuda"):
            try:
                device = torch.device(device_str)
                # Test the device
                test_tensor = torch.zeros(1).to(device)
                return device
            except Exception as e:
                print(f"⚠️ CUDA device setup failed: {e}")
                print("🔄 Falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")


class MT5Trainer:
    """MT5 Training and Backtesting Manager"""
    
    def __init__(self, config):
        self.config = config
        self.device = None
        self.model = None
        
        # Create directories
        os.makedirs("./trained_models", exist_ok=True)
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./tensorboard_log", exist_ok=True)
        
        # Setup device
        self.setup_device()
    
    def setup_device(self):
        """Setup GPU/CPU device with compatibility checks"""
        print("🔍 Detecting GPU compatibility...")
        device_str, device_info = GPUManager.detect_gpu()
        print(f"📱 Device info: {device_info}")
        
        self.device = GPUManager.setup_device(device_str)
        print(f"✅ Using device: {self.device}")
        
        if str(self.device) == "cpu":
            print("⚠️  Training will be slower on CPU. Consider:")
            print("   1. Updating PyTorch to latest version")
            print("   2. Installing CUDA-compatible PyTorch")
            print("   3. Checking GPU drivers")
    
    def download_training_data(self):
        """Download training data for the specified period"""
        print(f"\n📥 Downloading training data...")
        print(f"   Period: {self.config['training_start']} to {self.config['training_end']}")
        print(f"   Timeframe: {self.config['timeframe']}")
        print(f"   Symbols: {self.config['symbols']}")
        
        try:
            processor = MT5Processor(
                data_source="mt5",
                start_date=self.config['training_start'],
                end_date=self.config['training_end'],
                time_interval=self.config['timeframe'],
                ticker_list=self.config['symbols']
            )
            
            processor.download_data(ticker_list=self.config['symbols'])
            processor.add_technical_indicator(
                tech_indicator_list=self.config['technical_indicators']
            )
            
            print(f"✅ Training data ready: {len(processor.dataframe)} records")
            return processor.dataframe
            
        except Exception as e:
            print(f"❌ Error downloading training data: {e}")
            raise
    
    def download_backtest_data(self):
        """Download backtest data for the period after training"""
        print(f"\n📥 Downloading backtest data...")
        print(f"   Period: {self.config['backtest_start']} to {self.config['backtest_end']}")
        print(f"   Timeframe: {self.config['timeframe']}")
        
        try:
            processor = MT5Processor(
                data_source="mt5",
                start_date=self.config['backtest_start'],
                end_date=self.config['backtest_end'],
                time_interval=self.config['timeframe'],
                ticker_list=self.config['symbols']
            )
            
            processor.download_data(ticker_list=self.config['symbols'])
            processor.add_technical_indicator(
                tech_indicator_list=self.config['technical_indicators']
            )
            
            print(f"✅ Backtest data ready: {len(processor.dataframe)} records")
            return processor.dataframe
            
        except Exception as e:
            print(f"❌ Error downloading backtest data: {e}")
            raise
    
    def create_environment(self, df, env_type="training"):
        """Create trading environment"""
        print(f"\n🏗️  Creating {env_type} environment...")
        
        config = {
            "initial_balance": self.config['initial_balance'],
            "max_position_size": self.config['max_position_size'],
            "transaction_fee": self.config['transaction_fee'],
            "slippage": self.config['slippage'],
            "max_drawdown": self.config['max_drawdown'],
            "risk_per_trade": self.config['risk_per_trade'],
            "reward_function": self.config['reward_function'],
            "lookback_window": self.config['lookback_window']
        }
        
        env = BestPracticeForexEnv(df=df, config=config)
        print(f"✅ {env_type.capitalize()} environment created")
        return env
    
    def train_model(self, train_env):
        """Train the PPO model"""
        print(f"\n🚀 Starting model training...")
        print(f"   Device: {self.device}")
        print(f"   Timesteps: {self.config['total_timesteps']:,}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        
        try:
            # Wrap environment for Stable-Baselines3
            vec_env = DummyVecEnv([lambda: train_env])
            
            # Create evaluation environment for callbacks
            eval_env = DummyVecEnv([lambda: train_env])
            
            # Setup evaluation callback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path="./trained_models/best_model/",
                log_path="./tensorboard_log/",
                eval_freq=max(1000, self.config['total_timesteps'] // 100),
                deterministic=True,
                render=False
            )
            
            # Create PPO model
            self.model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log="./tensorboard_log/",
                device=self.device,
                learning_rate=self.config['learning_rate'],
                n_steps=self.config['n_steps'],
                batch_size=self.config['batch_size'],
                n_epochs=self.config['n_epochs'],
                gamma=self.config['gamma'],
                gae_lambda=self.config['gae_lambda'],
                clip_range=self.config['clip_range'],
                ent_coef=self.config['ent_coef'],
                vf_coef=self.config['vf_coef'],
                max_grad_norm=self.config['max_grad_norm'],
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[256, 256],
                        vf=[256, 256]
                    ),
                    activation_fn=torch.nn.ReLU,
                )
            )
            
            print("✅ PPO model created successfully")
            
            # Train the model
            print("🎯 Training started...")
            self.model.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=eval_callback,
                progress_bar=True
            )
            
            # Save the final model
            model_path = f"./trained_models/forex_ppo_{self.config['timeframe']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            self.model.save(model_path)
            print(f"✅ Model saved to {model_path}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def run_backtest(self, model_path, backtest_env, num_episodes=5):
        """Run backtest on the trained model"""
        print(f"\n🧪 Running backtest...")
        print(f"   Episodes: {num_episodes}")
        print(f"   Model: {model_path}")
        
        try:
            # Load the trained model
            model = PPO.load(model_path, device=self.device)
            print("✅ Model loaded successfully")
            
            # Run backtest episodes
            results = []
            for episode in range(num_episodes):
                print(f"   Episode {episode + 1}/{num_episodes}")
                
                obs = backtest_env.reset()
                done = False
                total_reward = 0
                trades = []
                balance_history = []
                
                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = backtest_env.step(action)
                    total_reward += reward
                    
                    # Track balance
                    current_balance = info.get('balance', backtest_env.balance)
                    balance_history.append(current_balance)
                    
                    # Track trades if available
                    if 'trades' in info:
                        trades.extend(info['trades'])
                
                # Calculate metrics
                final_balance = balance_history[-1] if balance_history else backtest_env.balance
                initial_balance = self.config['initial_balance']
                total_return = (final_balance - initial_balance) / initial_balance
                
                # Calculate drawdown
                if balance_history:
                    peak = max(balance_history)
                    drawdown = min((b - peak) / peak for b in balance_history)
                else:
                    drawdown = 0
                
                episode_result = {
                    'episode': episode + 1,
                    'total_reward': total_reward,
                    'final_balance': final_balance,
                    'total_return': total_return,
                    'max_drawdown': drawdown,
                    'num_trades': len(trades) if trades else 0
                }
                
                results.append(episode_result)
                
                print(f"      Reward: {total_reward:.2f}, Balance: ${final_balance:.2f}, Return: {total_return:.2%}")
            
            # Calculate aggregate metrics
            avg_reward = np.mean([r['total_reward'] for r in results])
            avg_return = np.mean([r['total_return'] for r in results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in results])
            total_trades = sum([r['num_trades'] for r in results])
            
            backtest_summary = {
                'episodes': results,
                'summary': {
                    'avg_reward': avg_reward,
                    'avg_return': avg_return,
                    'avg_drawdown': avg_drawdown,
                    'total_trades': total_trades,
                    'win_rate': sum(1 for r in results if r['total_return'] > 0) / len(results)
                }
            }
            
            print(f"✅ Backtest completed")
            print(f"   Average Reward: {avg_reward:.2f}")
            print(f"   Average Return: {avg_return:.2%}")
            print(f"   Average Drawdown: {avg_drawdown:.2%}")
            print(f"   Total Trades: {total_trades}")
            print(f"   Win Rate: {backtest_summary['summary']['win_rate']:.1%}")
            
            return backtest_summary
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            raise
    
    def save_results(self, results, model_path):
        """Save backtest results"""
        print(f"\n💾 Saving results...")
        
        try:
            # Save detailed results
            results_file = f"./results/backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'model_path': model_path,
                    'config': self.config,
                    'results': results
                }, f, indent=2, default=str)
            
            print(f"✅ Results saved to {results_file}")
            
            # Create summary plot
            self.create_results_plot(results)
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def create_results_plot(self, results):
        """Create visualization of backtest results"""
        try:
            episodes = [r['episode'] for r in results['episodes']]
            rewards = [r['total_reward'] for r in results['episodes']]
            returns = [r['total_return'] * 100 for r in results['episodes']]
            balances = [r['final_balance'] for r in results['episodes']]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Backtest Results - {self.config["timeframe"]} Timeframe', fontsize=16)
            
            # Episode rewards
            axes[0, 0].bar(episodes, rewards, color='lightblue', alpha=0.7)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Episode returns
            axes[0, 1].bar(episodes, returns, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Episode Returns')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Final balances
            axes[1, 0].bar(episodes, balances, color='orange', alpha=0.7)
            axes[1, 0].set_title('Final Balances')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Balance ($)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Summary metrics
            summary = results['summary']
            metrics_text = f"""
            Average Reward: {summary['avg_reward']:.2f}
            Average Return: {summary['avg_return']:.2%}
            Average Drawdown: {summary['avg_drawdown']:.2%}
            Total Trades: {summary['total_trades']}
            Win Rate: {summary['win_rate']:.1%}
            """
            
            axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            axes[1, 1].set_title('Summary Metrics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = f"./results/backtest_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {plot_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating plot: {e}")
    
    def run_pipeline(self):
        """Run the complete training and backtesting pipeline"""
        print("🚀 Starting MT5 GPU-Fixed Training Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Download training data
            train_df = self.download_training_data()
            
            # Step 2: Create training environment
            train_env = self.create_environment(train_df, "training")
            
            # Step 3: Train the model
            model_path = self.train_model(train_env)
            
            # Step 4: Download backtest data
            backtest_df = self.download_backtest_data()
            
            # Step 5: Create backtest environment
            backtest_env = self.create_environment(backtest_df, "backtest")
            
            # Step 6: Run backtest
            backtest_results = self.run_backtest(model_path, backtest_env, self.config['backtest_episodes'])
            
            # Step 7: Save results
            self.save_results(backtest_results, model_path)
            
            print("\n🎉 Pipeline completed successfully!")
            print(f"\n📊 Results saved to ./results/")
            print(f"🤖 Model saved to {model_path}")
            print(f"📈 Tensorboard logs: ./tensorboard_log/")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='MT5 GPU-Fixed Training and Backtesting')
    
    parser.add_argument('--timeframe', type=str, default='1m', 
                       choices=['1m', '5m', '15m', '30m', '1h', '4h'],
                       help='Trading timeframe (default: 1m)')
    
    parser.add_argument('--training_months', type=int, default=3,
                       help='Number of months for training data (default: 3)')
    
    parser.add_argument('--backtest_months', type=int, default=1,
                       help='Number of months for backtest data (default: 1)')
    
    parser.add_argument('--symbols', nargs='+', 
                       default=['EURUSD', 'GBPUSD', 'USDJPY'],
                       help='Trading symbols (default: EURUSD GBPUSD USDJPY)')
    
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Training timesteps (default: 500000)')
    
    args = parser.parse_args()
    
    # Calculate dates
    end_date = datetime.now()
    training_start = end_date - timedelta(days=args.training_months * 30)
    training_end = end_date - timedelta(days=args.backtest_months * 30)
    backtest_start = training_end
    backtest_end = end_date
    
    # Configuration
    config = {
        'timeframe': args.timeframe,
        'symbols': args.symbols,
        'training_start': training_start.strftime('%Y-%m-%d'),
        'training_end': training_end.strftime('%Y-%m-%d'),
        'backtest_start': backtest_start.strftime('%Y-%m-%d'),
        'backtest_end': backtest_end.strftime('%Y-%m-%d'),
        'total_timesteps': args.timesteps,
        'initial_balance': 100000,
        'max_position_size': 0.05,
        'transaction_fee': 0.0002,
        'slippage': 0.0001,
        'max_drawdown': 0.15,
        'risk_per_trade': 0.01,
        'reward_function': 'sharpe',
        'lookback_window': 20,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'technical_indicators': ['macd', 'rsi', 'boll_ub', 'boll_lb', 'atr', 'ema_20'],
        'backtest_episodes': 5
    }
    
    print("📋 Configuration:")
    print(f"   Timeframe: {config['timeframe']}")
    print(f"   Training Period: {config['training_start']} to {config['training_end']}")
    print(f"   Backtest Period: {config['backtest_start']} to {config['backtest_end']}")
    print(f"   Symbols: {', '.join(config['symbols'])}")
    print(f"   Training Timesteps: {config['total_timesteps']:,}")
    
    # Create trainer and run pipeline
    trainer = MT5Trainer(config)
    trainer.run_pipeline()


if __name__ == "__main__":
    main()
