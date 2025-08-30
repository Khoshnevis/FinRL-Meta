#!/usr/bin/env python3
"""
Detailed Backtest Analysis Script
Analyzes the timeframe, profit calculations, and trading performance
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

from examples.MT5_1min_Training import run_backtest_on_trained_model, download_1min_data

def analyze_backtest_detailed(model_path):
    """Run detailed backtest analysis with timeframe and profit breakdown"""
    print("🔍 Detailed Backtest Analysis (3 Months)")
    print("=" * 50)
    
    # Download 3 months of data to get timeframe info
    print("\n📥 Downloading 3 months of 1-minute data...")
    df = download_1min_data()
    
    # Analyze data timeframe
    print("\n📅 Data Timeframe Analysis:")
    print(f"   Total Records: {len(df):,}")
    print(f"   Symbols: {df['tic'].unique()}")
    
    # Get time range for each symbol
    for symbol in df['tic'].unique():
        symbol_data = df[df['tic'] == symbol]
        start_time = symbol_data['time'].min()
        end_time = symbol_data['time'].max()
        duration = end_time - start_time
        
        print(f"\n   {symbol}:")
        print(f"     Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"     Duration: {duration}")
        print(f"     Trading Hours: {duration.total_seconds() / 3600:.1f} hours")
        print(f"     Trading Days: {duration.total_seconds() / (24 * 3600):.2f} days")
        print(f"     Trading Months: {duration.total_seconds() / (30 * 24 * 3600):.2f} months")
    
    # Run backtest with detailed tracking
    print(f"\n🧪 Running Detailed Backtest on 3 Months of Data...")
    
    try:
        # Create environment
        from examples.MT5_1min_Training import create_environment
        env = create_environment(df)
        
        # Load model
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")
        print("✅ Model loaded successfully on CPU")
        
        # Track detailed metrics
        episode_details = []
        
        for episode in range(3):  # Run 3 episodes for analysis
            print(f"\n   Episode {episode + 1}/3")
            
            # Reset environment
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            # Track step-by-step
            step_details = []
            initial_balance = env.initial_balance
            current_balance = initial_balance
            total_reward = 0
            trades = []
            
            done = False
            step = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                
                # Record before step
                old_balance = current_balance
                
                # Take step
                step_result = env.step(action)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, done, truncated, info = step_result
                    done = done or truncated
                
                # Get new balance
                new_balance = info.get('balance', getattr(env, 'balance', current_balance))
                balance_change = new_balance - old_balance
                
                # Extract lot size from positions (positions are stored in lots)
                positions = info.get('positions', {})
                current_lot_size = 0
                current_position = 0
                
                # Sum up all position sizes across all symbols
                for symbol, position in positions.items():
                    current_lot_size += abs(position)  # Position size in lots
                    current_position += position  # Net position (positive = long, negative = short)
                
                # Record step details
                step_detail = {
                    'step': step,
                    'action': action,
                    'reward': reward,
                    'old_balance': old_balance,
                    'new_balance': new_balance,
                    'balance_change': balance_change,
                    'cumulative_reward': total_reward + reward,
                    'lot_size': current_lot_size,  # Total lot size across all positions
                    'position': current_position,   # Net position across all symbols
                    'positions_detail': positions.copy()  # Detailed positions per symbol
                }
                step_details.append(step_detail)
                
                # Track trades
                # Fix: Handle action that might be an array
                action_value = action[0] if hasattr(action, '__len__') and len(action) > 0 else action
                if action_value != 0:  # Not hold
                    trades.append({
                        'step': step,
                        'action': action_value,
                        'balance_change': balance_change,
                        'reward': reward,
                        'lot_size': current_lot_size,  # Current lot size when trade was made
                        'position': current_position,   # Current net position when trade was made
                        'positions_detail': positions.copy()  # Detailed positions when trade was made
                    })
                
                current_balance = new_balance
                total_reward += reward
                step += 1
                
                # Progress indicator for long episodes
                if step % 1000 == 0:
                    print(f"       Step {step:,} - Balance: ${current_balance:,.2f}")
            
            # Calculate episode metrics
            final_balance = current_balance
            total_profit = final_balance - initial_balance
            profit_percentage = (total_profit / initial_balance) * 100
            
            episode_detail = {
                'episode': episode + 1,
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_profit': total_profit,
                'profit_percentage': profit_percentage,
                'total_reward': total_reward,
                'total_trades': len(trades),
                'step_details': step_details,
                'trades': trades,
                'avg_lot_size': np.mean([t['lot_size'] for t in trades]) if trades else 0,  # Add average lot size
                'max_lot_size': max([t['lot_size'] for t in trades]) if trades else 0       # Add max lot size
            }
            episode_details.append(episode_detail)
            
            print(f"     Initial Balance: ${initial_balance:,.2f}")
            print(f"     Final Balance: ${final_balance:,.2f}")
            print(f"     Total Profit: ${total_profit:,.2f}")
            print(f"     Profit %: {profit_percentage:.2f}%")
            print(f"     Total Reward: {total_reward:.2f}")
            print(f"     Total Trades: {len(trades)}")
            print(f"     Total Steps: {step:,}")
            if trades:
                print(f"     Average Lot Size: {np.mean([t['lot_size'] for t in trades]):.2f}")
                print(f"     Max Lot Size: {max([t['lot_size'] for t in trades]):.2f}")
                print(f"     Final Positions: {trades[-1]['positions_detail'] if trades else 'None'}")
        
        # Summary analysis
        print(f"\n📊 Summary Analysis:")
        print("=" * 50)
        
        initial_balances = [ep['initial_balance'] for ep in episode_details]
        final_balances = [ep['final_balance'] for ep in episode_details]
        profits = [ep['total_profit'] for ep in episode_details]
        profit_percentages = [ep['profit_percentage'] for ep in episode_details]
        rewards = [ep['total_reward'] for ep in episode_details]
        trade_counts = [ep['total_trades'] for ep in episode_details]
        avg_lot_sizes = [ep['avg_lot_size'] for ep in episode_details]
        max_lot_sizes = [ep['max_lot_size'] for ep in episode_details]
        
        print(f"   Average Initial Balance: ${np.mean(initial_balances):,.2f}")
        print(f"   Average Final Balance: ${np.mean(final_balances):,.2f}")
        print(f"   Average Profit: ${np.mean(profits):,.2f}")
        print(f"   Average Profit %: {np.mean(profit_percentages):.2f}%")
        print(f"   Average Reward: {np.mean(rewards):.2f}")
        print(f"   Average Trades: {np.mean(trade_counts):.1f}")
        print(f"   Average Lot Size: {np.mean(avg_lot_sizes):.2f}")
        print(f"   Max Lot Size: {np.mean(max_lot_sizes):.2f}")
        
        # Timeframe summary
        print(f"\n⏰ Timeframe Summary:")
        print("=" * 50)
        print(f"   Data Period: {df['time'].min().strftime('%Y-%m-%d %H:%M')} to {df['time'].max().strftime('%Y-%m-%d %H:%M')}")
        print(f"   Total Trading Time: {(df['time'].max() - df['time'].min()).total_seconds() / 3600:.1f} hours")
        print(f"   Total Trading Days: {(df['time'].max() - df['time'].min()).total_seconds() / (24 * 3600):.1f} days")
        print(f"   Total Trading Months: {(df['time'].max() - df['time'].min()).total_seconds() / (30 * 24 * 3600):.2f} months")
        print(f"   Data Points: {len(df):,} 1-minute candles")
        print(f"   Symbols: {', '.join(df['tic'].unique())}")
        
        # Annualized return calculation
        total_days = (df['time'].max() - df['time'].min()).total_seconds() / (24 * 3600)
        avg_profit_pct = np.mean(profit_percentages)
        annualized_return = ((1 + avg_profit_pct/100) ** (365/total_days) - 1) * 100
        
        print(f"\n📈 Annualized Performance:")
        print("=" * 50)
        print(f"   Test Period: {total_days:.1f} days")
        print(f"   Average Return: {avg_profit_pct:.2f}%")
        print(f"   Annualized Return: {annualized_return:.2f}%")
        
        return episode_details
        
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Use your trained model
    model_path = "./trained_models/forex_ppo_1min_20250831_001842.zip"
    
    if os.path.exists(model_path):
        results = analyze_backtest_detailed(model_path)
        if results:
            print(f"\n✅ Detailed analysis completed!")
    else:
        print(f"❌ Model not found: {model_path}")
        print("Please run training first or check the model path.")
