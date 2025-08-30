#!/usr/bin/env python3
"""
MT5 Backtesting After Training for FinRL-Meta

This script runs backtests on trained models using market data from after the training period.
It provides comprehensive performance metrics and visualizations.

Usage:
    python MT5_Backtest_After_Training.py --model_path ./trained_models/your_model.zip
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
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json


class MT5Backtester:
    """MT5 Backtesting Manager for Trained Models"""
    
    def __init__(self, config):
        """Initialize the backtester with configuration"""
        self.config = config
        self.model = None
        self.env = None
        self.episodes = config.get('episodes', 10)  # Add episodes attribute
        
        # Create results directory
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./backtest_plots", exist_ok=True)
        
        # Setup device
        self.setup_device()
    
    def setup_device(self):
        """Setup device with GPU compatibility check"""
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
                        self.device = "cpu"
                        return
                
                # Test memory allocation
                try:
                    test_tensor = torch.randn(1000, 1000).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    print("✅ GPU memory test passed - using GPU")
                    self.device = "cuda:0"
                    return
                except Exception as e:
                    print(f"⚠️  GPU memory test failed: {e} - using CPU")
                    self.device = "cpu"
                    return
                    
            except Exception as e:
                print(f"⚠️  CUDA test failed: {e} - using CPU")
                self.device = "cpu"
                return
        else:
            print("ℹ️  No GPU detected - using CPU")
            self.device = "cpu"
    
    def load_model(self, model_path):
        """Load the trained model"""
        print(f"\n🤖 Loading model from {model_path}...")
        
        try:
            self.model = PPO.load(model_path, device=self.device)
            print("✅ Model loaded successfully")
            
            # Get model info
            if hasattr(self.model, 'policy'):
                print(f"   Policy type: {type(self.model.policy).__name__}")
                if hasattr(self.model.policy, 'net_arch'):
                    print(f"   Network architecture: {self.model.policy.net_arch}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def download_backtest_data(self):
        """Download backtest data for the period after training"""
        print(f"\n📥 Downloading backtest data...")
        print(f"   Period: {self.config['backtest_start']} to {self.config['backtest_end']}")
        print(f"   Timeframe: {self.config['timeframe']}")
        print(f"   Symbols: {', '.join(self.config['symbols'])}")
        
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
            
            self.backtest_data = processor.dataframe
            print(f"✅ Backtest data ready: {len(self.backtest_data)} records")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading backtest data: {e}")
            return False
    
    def create_backtest_environment(self):
        """Create backtest environment"""
        print(f"\n🏗️  Creating backtest environment...")
        
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
        
        self.env = BestPracticeForexEnv(df=self.backtest_data, config=config)
        print(f"✅ Backtest environment created")
        return True
    
    def run_backtest(self):
        """Run backtest on the loaded model"""
        print(f"\n🧪 Running backtest with {self.episodes} episodes...")
        
        try:
            total_rewards = []
            final_balances = []
            trade_histories = []
            balance_histories = []
            
            for episode in range(self.episodes):
                print(f"   Episode {episode + 1}/{self.episodes}")
                
                # Fix: Handle tuple return from env.reset()
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs, info = reset_result
                else:
                    obs = reset_result
                    info = {}
                
                done = False
                total_reward = 0
                episode_trades = []
                episode_balances = [self.env.initial_balance]
                
                while not done:
                    action, _states = self.model.predict(obs, deterministic=True)
                    
                    # Fix: Handle tuple return from env.step()
                    step_result = self.env.step(action)
                    if len(step_result) == 4:
                        obs, reward, done, info = step_result
                    else:
                        obs, reward, done, truncated, info = step_result
                        done = done or truncated
                    
                    total_reward += reward
                    
                    # Record trade if action was taken
                    if action != 0:  # 0 = hold
                        trade_info = {
                            'step': len(episode_balances),
                            'action': action,
                            'reward': reward,
                            'balance': info.get('balance', getattr(self.env, 'balance', episode_balances[-1]))
                        }
                        episode_trades.append(trade_info)
                    
                    # Record balance
                    current_balance = info.get('balance', getattr(self.env, 'balance', episode_balances[-1]))
                    episode_balances.append(current_balance)
                
                final_balance = episode_balances[-1]
                total_rewards.append(total_reward)
                final_balances.append(final_balance)
                trade_histories.append(episode_trades)
                balance_histories.append(episode_balances)
                
                print(f"      Reward: {total_reward:.2f}, Final Balance: ${final_balance:.2f}")
                print(f"      Trades: {len(episode_trades)}")
            
            # Calculate aggregate metrics
            metrics = self.calculate_aggregate_metrics(total_rewards, final_balances, trade_histories, balance_histories)
            
            # Save results
            self.save_results(metrics, trade_histories, balance_histories)
            
            # Create plots
            self.create_comprehensive_plots(metrics, trade_histories, balance_histories)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_aggregate_metrics(self, results):
        """Calculate aggregate performance metrics"""
        returns = [r['total_return'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        trade_counts = [r['trade_count'] for r in results]
        
        return {
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_drawdown': np.mean(drawdowns),
            'max_drawdown': np.min(drawdowns),  # Most negative
            'total_trades': sum(trade_counts),
            'avg_trades': np.mean(trade_counts),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'profit_factor': sum(r for r in returns if r > 0) / abs(sum(r for r in returns if r < 0)) if sum(r for r in returns if r < 0) != 0 else float('inf'),
            'calmar_ratio': np.mean(returns) / abs(np.min(drawdowns)) if np.min(drawdowns) != 0 else 0
        }
    
    def save_results(self, results):
        """Save backtest results"""
        print(f"\n💾 Saving results...")
        
        try:
            # Save detailed results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"./results/backtest_results_{timestamp}.json"
            
            # Prepare data for JSON serialization
            json_results = {
                'config': self.config,
                'summary': results['summary'],
                'episodes': []
            }
            
            for episode in results['episodes']:
                # Remove numpy arrays and complex objects for JSON serialization
                json_episode = {k: v for k, v in episode.items() 
                              if k not in ['balance_history', 'trades']}
                json_episode['balance_history'] = [float(x) for x in episode['balance_history']]
                json_results['episodes'].append(json_episode)
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"✅ Results saved to {results_file}")
            
            # Create visualizations
            self.create_comprehensive_plots(results)
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def create_comprehensive_plots(self, results):
        """Create comprehensive visualization plots"""
        print(f"\n📊 Creating visualization plots...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create subplots
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Performance Overview
            ax1 = plt.subplot(3, 3, 1)
            episodes = [r['episode'] for r in results['episodes']]
            returns = [r['total_return'] * 100 for r in results['episodes']]
            bars = ax1.bar(episodes, returns, color='lightblue', alpha=0.7)
            ax1.set_title('Episode Returns (%)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Return (%)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, returns):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # 2. Balance Evolution
            ax2 = plt.subplot(3, 3, 2)
            for i, balance_history in enumerate(results['balance_histories']):
                ax2.plot(balance_history, label=f'Episode {i+1}', alpha=0.7)
            ax2.set_title('Balance Evolution')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Balance ($)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # 3. Risk Metrics
            ax3 = plt.subplot(3, 3, 3)
            drawdowns = [r['max_drawdown'] * 100 for r in results['episodes']]
            ax3.bar(episodes, drawdowns, color='lightcoral', alpha=0.7)
            ax3.set_title('Maximum Drawdown (%)')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Sharpe Ratios
            ax4 = plt.subplot(3, 3, 4)
            sharpe_ratios = [r['sharpe_ratio'] for r in results['episodes']]
            ax4.bar(episodes, sharpe_ratios, color='lightgreen', alpha=0.7)
            ax4.set_title('Sharpe Ratios')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.grid(True, alpha=0.3)
            
            # 5. Trade Counts
            ax5 = plt.subplot(3, 3, 5)
            trade_counts = [r['trade_count'] for r in results['episodes']]
            ax5.bar(episodes, trade_counts, color='orange', alpha=0.7)
            ax5.set_title('Trade Counts')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Number of Trades')
            ax5.grid(True, alpha=0.3)
            
            # 6. Cumulative Returns
            ax6 = plt.subplot(3, 3, 6)
            cumulative_returns = np.cumsum(returns)
            ax6.plot(episodes, cumulative_returns, marker='o', linewidth=2, markersize=6)
            ax6.set_title('Cumulative Returns (%)')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Cumulative Return (%)')
            ax6.grid(True, alpha=0.3)
            
            # 7. Risk-Return Scatter
            ax7 = plt.subplot(3, 3, 7)
            volatilities = [r['volatility'] * 100 for r in results['episodes']]
            ax7.scatter(volatilities, returns, alpha=0.7, s=100)
            ax7.set_title('Risk-Return Profile')
            ax7.set_xlabel('Volatility (%)')
            ax7.set_ylabel('Return (%)')
            ax7.grid(True, alpha=0.3)
            
            # 8. Performance Distribution
            ax8 = plt.subplot(3, 3, 8)
            ax8.hist(returns, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax8.set_title('Return Distribution')
            ax8.set_xlabel('Return (%)')
            ax8.set_ylabel('Frequency')
            ax8.grid(True, alpha=0.3)
            
            # 9. Summary Metrics
            ax9 = plt.subplot(3, 3, 9)
            summary = results['summary']
            metrics_text = f"""
            Average Return: {summary['avg_return']:.2%}
            Std Return: {summary['std_return']:.2%}
            Max Drawdown: {summary['max_drawdown']:.2%}
            Win Rate: {summary['win_rate']:.1%}
            Total Trades: {summary['total_trades']}
            Avg Sharpe: {summary['avg_sharpe']:.3f}
            Profit Factor: {summary['profit_factor']:.2f}
            Calmar Ratio: {summary['calmar_ratio']:.3f}
            """
            
            ax9.text(0.1, 0.5, metrics_text, transform=ax9.transAxes,
                     fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            ax9.set_title('Performance Summary')
            ax9.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = f"./backtest_plots/comprehensive_backtest_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✅ Comprehensive plot saved to {plot_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating plots: {e}")
            import traceback
            traceback.print_exc()
    
    def run_pipeline(self):
        """Run the complete backtesting pipeline"""
        print("🚀 Starting MT5 Backtesting Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Setup device
            self.setup_device()
            
            # Step 2: Load model
            if not self.load_model(self.config['model_path']):
                return None
            
            # Step 3: Download backtest data
            if not self.download_backtest_data():
                return None
            
            # Step 4: Create backtest environment
            if not self.create_backtest_environment():
                return None
            
            # Step 5: Run backtest
            backtest_results = self.run_backtest()
            
            if backtest_results:
                print("\n🎉 Backtesting completed successfully!")
                print(f"📊 Results saved to: ./results/")
                print(f"📈 Plots saved to: ./results/")
                return backtest_results
            else:
                print("\n❌ Backtesting failed")
                return None
                
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='MT5 Backtesting After Training')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model .zip file')
    
    parser.add_argument('--timeframe', type=str, default='1m',
                       choices=['1m', '5m', '15m', '30m', '1h', '4h'],
                       help='Trading timeframe (default: 1m)')
    
    parser.add_argument('--backtest_months', type=int, default=1,
                       help='Number of months for backtest data (default: 1)')
    
    parser.add_argument('--symbols', nargs='+',
                       default=['EURUSD', 'GBPUSD', 'USDJPY'],
                       help='Trading symbols (default: EURUSD GBPUSD USDJPY)')
    
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of backtest episodes (default: 10)')
    
    args = parser.parse_args()
    
    # Calculate backtest dates (after training period)
    end_date = datetime.now()
    backtest_start = end_date - timedelta(days=args.backtest_months * 30)
    backtest_end = end_date
    
    # Configuration
    config = {
        'model_path': args.model_path,
        'timeframe': args.timeframe,
        'symbols': args.symbols,
        'backtest_start': backtest_start.strftime('%Y-%m-%d'),
        'backtest_end': backtest_end.strftime('%Y-%m-%d'),
        'num_episodes': args.episodes,
        'initial_balance': 100000,
        'max_position_size': 0.05,
        'transaction_fee': 0.0002,
        'slippage': 0.0001,
        'max_drawdown': 0.15,
        'risk_per_trade': 0.01,
        'reward_function': 'sharpe',
        'lookback_window': 20,
        'technical_indicators': ['macd', 'rsi', 'boll_ub', 'boll_lb', 'atr', 'ema_20']
    }
    
    print("📋 Configuration:")
    print(f"   Model: {config['model_path']}")
    print(f"   Timeframe: {config['timeframe']}")
    print(f"   Backtest Period: {config['backtest_start']} to {config['backtest_end']}")
    print(f"   Symbols: {', '.join(config['symbols'])}")
    print(f"   Episodes: {config['num_episodes']}")
    
    # Create backtester and run pipeline
    backtester = MT5Backtester(config)
    backtester.run_pipeline()


if __name__ == "__main__":
    main()
