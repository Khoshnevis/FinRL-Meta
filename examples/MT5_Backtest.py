import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from meta.data_processors.mt5 import MT5 as MT5Processor
from examples.Best_Practice_Forex_Env import BestPracticeForexEnv

def run_backtest(model_path, start_date, end_date, timeframe, initial_balance=100000):
    """
    Runs a backtest for a trained model on a specified historical period.

    :param model_path: Path to the saved model .zip file.
    :param start_date: Backtest start date (e.g., "2023-01-01").
    :param end_date: Backtest end date (e.g., "2023-12-31").
    :param timeframe: MT5 timeframe (e.g., "1h", "15m").
    :param initial_balance: The starting balance for the backtest.
    """
    print("🚀 Starting MT5 Backtest")
    print("==========================")
    print(f"Model: {model_path}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Timeframe: {timeframe}")
    
    # --- 1. Load Trained Model ---
    try:
        # Force CPU usage to avoid GPU compatibility issues
        model = PPO.load(model_path, device='cpu')
        print("✓ Model loaded successfully on CPU.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- 2. Download Backtest Data ---
    TICKERS = ["EURUSD", "GBPUSD", "USDJPY"]
    backtest_processor = MT5Processor(
        data_source="mt5",
        start_date=start_date,
        end_date=end_date,
        time_interval=timeframe,
        ticker_list=TICKERS
    )
    backtest_processor.download_data(ticker_list=TICKERS)
    backtest_processor.add_technical_indicator(
        tech_indicator_list=["macd", "rsi", "boll_ub", "boll_lb", "atr", "ema_20"]
    )
    backtest_df = backtest_processor.dataframe
    print(f"✓ Backtest data downloaded: {len(backtest_df)} records.")

    # --- 3. Create Backtest Environment ---
    backtest_env = BestPracticeForexEnv(
        df=backtest_df,
        config={
            "initial_balance": initial_balance,
            "max_position_size": 0.05,
            "transaction_fee": 0.0002,
            "slippage": 0.0001,
            "max_drawdown": 0.15,
            "risk_per_trade": 0.01,
            "commission_per_lot": 4.0,
            "max_leverage": 30,
            "reward_function": "sharpe",
            "lookback_window": 20
        }
    )
    print("✓ Backtest environment created.")

    # --- 4. Run the Simulation ---
    obs, _ = backtest_env.reset()
    done = False
    print("\nRunning simulation...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = backtest_env.step(action)
        done = terminated or truncated

    print("✓ Simulation complete.")
    
    # --- 5. Performance Report ---
    print("\n📈 Performance Report")
    print("----------------------")
    
    final_balance = info['balance']
    total_profit = final_balance - initial_balance
    total_return = (total_profit / initial_balance) * 100
    sharpe_ratio = info['sharpe']
    max_drawdown_pct = info['drawdown'] * 100
    trade_history = pd.DataFrame(backtest_env.trade_history)
    num_trades = len(trade_history)
    
    print(f"Final Balance:    ${final_balance:,.2f}")
    print(f"Total Profit:     ${total_profit:,.2f}")
    print(f"Total Return:     {total_return:.2f}%")
    print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {max_drawdown_pct:.2f}%")
    print(f"Number of Trades: {num_trades}")

    if not trade_history.empty:
        winning_trades = trade_history[trade_history['pnl'] > 0]
        losing_trades = trade_history[trade_history['pnl'] <= 0]
        win_rate = (len(winning_trades) / num_trades) * 100 if num_trades > 0 else 0
        avg_profit = winning_trades['pnl'].mean()
        avg_loss = losing_trades['pnl'].mean()

        print(f"Win Rate:         {win_rate:.2f}%")
        print(f"Average Profit:   ${avg_profit:,.2f}")
        print(f"Average Loss:     ${avg_loss:,.2f}")

    # --- 6. Plot Equity Curve ---
    equity_curve = backtest_env.equity_curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f"Equity Curve ({start_date} to {end_date})")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    
    plot_filename = f"backtest_equity_curve_{start_date}_to_{end_date}.png"
    plt.savefig(plot_filename)
    print(f"\n✓ Equity curve plot saved to {plot_filename}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest a trained FinRL agent on MT5 data.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model .zip file.")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="2024-06-30", help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--timeframe", type=str, default="1h", help="MT5 timeframe (e.g., 1h, 15m, 4h).")
    args = parser.parse_args()

    run_backtest(args.model, args.start, args.end, args.timeframe)
