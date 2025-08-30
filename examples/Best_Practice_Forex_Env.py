#!/usr/bin/env python3
"""
Best Practice Forex Environment for FinRL-Meta (Revised and Corrected)

This environment incorporates critical fixes and enhancements for realistic RL trading simulation:
1. Correct Accounting (Fix 1): Distinction between Balance (realized cash) and Equity (mark-to-market). P&L realized only on position reduction/closure.
2. Data Integrity (Fix 2): Technical indicators calculated independently per asset (groupby) to prevent contamination.
3. Advanced Reward Functions (Fix 3): Options for Differential Sharpe Ratio (DSR) and P&L with Drawdown Penalty.
4. Enhanced Feature Engineering (Fix 4): Cyclical time embeddings (Sine/Cosine) and normalized observations.
5. Realistic Costs (Fix 5): Dynamic spread/slippage based on volatility (ATR) and realistic commissions.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Optional imports for data processing and RL (used in the main example)
try:
    # Add the parent directory to the path if necessary for local imports
    # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from meta.data_processor import DataProcessor, DataSource
except ImportError:
    print("Note: Could not import meta.data_processor. Ensure FinRL-Meta is installed if using that data source.")
    DataProcessor = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    # stable_baselines3 is not strictly required to define the environment
    pass


class BestPracticeForexEnv(gym.Env):
    """
    Best Practice Forex Trading Environment (Revised)
    """
    
    def __init__(self, df, config=None):
        super().__init__()
        
        # Default configuration (Enhanced and Corrected)
        default_config = {
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
        
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Constants
        self.LOT_SIZE = 100000
        
        # Data preparation (Fixes 2 and 4 implemented in _prepare_data)
        # This also sets self.time_indices
        self.df = self._prepare_data(df)
        self.current_step = 0
        self.max_steps = len(self.time_indices) # Steps based on unique time indices
        
        # Portfolio state (Fix 1: Equity Accounting Implementation)
        self.initial_balance = self.config["initial_balance"]
        self.balance = self.initial_balance  # Realized cash balance
        self.equity = self.initial_balance   # Mark-to-market value (Balance + Unrealized PnL)
        self.previous_equity = self.initial_balance
        
        self.positions = {}          # Current positions (in lots) per asset
        self.position_entries = {}   # Entry prices (Weighted Average)
        self.trade_history = []
        self.equity_curve = []
        
        # Risk management (Based on Equity)
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.peak_equity = self.equity
        self.step_returns = [] # Tracks per-step returns
        
        # DSR tracking variables (Fix 3)
        self.dsr_A = 0  # Estimate of E[R_t] (First moment)
        self.dsr_B = 0  # Estimate of E[R_t^2] (Second moment)
        
        # Action and observation spaces
        # Ensure consistent order of tickers
        self.unique_tickers = sorted(list(df['tic'].unique()))
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(len(self.unique_tickers),),
            dtype=np.float32
        )
        
        # Define feature columns (excluding metadata)
        metadata_cols = ['time', 'tic']
        self.feature_cols = [col for col in self.df.columns if col not in metadata_cols]
        features_per_ticker = len(self.feature_cols)
        
        # Observation space definition
        # 10 base metrics + 2 features per ticker (position size, direction indicator)
        self.portfolio_state_size = 10 + len(self.unique_tickers) * 2
        
        obs_size = (
            len(self.unique_tickers) * features_per_ticker +
            self.portfolio_state_size
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        self._print_env_info(features_per_ticker, obs_size)

    def _print_env_info(self, features_per_ticker, obs_size):
        print(f"✓ Best Practice Forex Environment Initialized (Revised)")
        print(f"  Assets: {self.unique_tickers}")
        print(f"  Time steps: {self.max_steps}")
        print(f"  Initial Equity: ${self.equity:,.2f}")
        print(f"  Reward Function: {self.config['reward_function']}")
        print(f"  Observation size: {obs_size} ({features_per_ticker} features/ticker)")

    # =====================================================================================
    # Data Preparation and Feature Engineering (Fixes 2, 4)
    # =====================================================================================

    def _prepare_data(self, df):
        """Prepare data: Handle indicators correctly and add time features."""
        print("Preparing data...")
        
        df = df.copy()
        
        # Ensure required columns and time format
        required_cols = ['tic', 'time', 'open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                if col == 'volume':
                     df['volume'] = 0 # Add volume if missing
                else:
                    raise ValueError(f"Missing required column: {col}")

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by=['time', 'tic']).reset_index(drop=True)

        # 1. Add Technical Indicators (Fix 2: GroupBy 'tic')
        # Ensure calculations are done per asset to prevent contamination.
        
        if 'rsi' not in df.columns:
            df['rsi'] = df.groupby('tic')['close'].transform(lambda x: self._calculate_rsi(x))
        
        if 'macd' not in df.columns:
            df['macd'] = df.groupby('tic')['close'].transform(lambda x: self._calculate_macd(x))
        
        if 'boll_ub' not in df.columns:
            # Handle indicators returning multiple columns using apply
            def apply_bollinger(group):
                ub, lb = self._calculate_bollinger_bands(group['close'])
                group['boll_ub'] = ub
                group['boll_lb'] = lb
                return group
            # Use group_keys=False to maintain original index structure
            df = df.groupby('tic', group_keys=False).apply(apply_bollinger)

        # ATR Calculation (Crucial for dynamic costs and position sizing)
        def apply_atr(group):
            group['atr'] = self._calculate_atr(group['high'], group['low'], group['close'])
            return group
        df = df.groupby('tic', group_keys=False).apply(apply_atr)
        
        if 'ema_20' not in df.columns:
             df['ema_20'] = df.groupby('tic')['close'].transform(lambda x: self._calculate_ema(x, 20))

        # 2. Add Time Features (Fix 4: Sine/Cosine Embeddings)
        hour = df['time'].dt.hour
        day_of_week = df['time'].dt.dayofweek
        
        # Normalized Sine/Cosine transformations
        df['hour_sin'] = np.sin(2 * np.pi * hour/23.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour/23.0)
        # Assuming 0-4 for Mon-Fri (common in Forex)
        df['dow_sin'] = np.sin(2 * np.pi * day_of_week/4.0) 
        df['dow_cos'] = np.cos(2 * np.pi * day_of_week/4.0)
        
        # Clean NaNs created by indicators (usually at the start of the series)
        df = df.fillna(0)

        # Create a time index mapping for efficient slicing during steps
        self.time_indices = df['time'].unique()
        
        # Set multi-index (time, tic) for faster lookups during simulation
        df = df.set_index(['time', 'tic'])

        print("Data preparation complete.")
        return df

    # Helper functions for indicators (Standard implementations)
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        # Use min_periods=1 to allow calculation early in the series
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def _calculate_atr(self, high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Use EMA for ATR calculation (common practice, e.g., Wilder's smoothing/MetaTrader)
        return tr.ewm(span=period, adjust=False).mean()
    
    def _calculate_ema(self, prices, period):
        return prices.ewm(span=period, adjust=False).mean()

    # =====================================================================================
    # Observation Space
    # =====================================================================================

    def _get_observation(self):
        """Get current observation vector using multi-index."""
        if self.current_step >= len(self.time_indices):
            # Handle end of data gracefully
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_time = self.time_indices[self.current_step]
        
        # 1. Market data features
        obs = []
        for ticker in self.unique_tickers:
            try:
                # Efficient lookup using multi-index
                ticker_data = self.df.loc[(current_time, ticker)]
                
                ticker_features = []
                for col in self.feature_cols:
                    value = ticker_data.get(col, 0)
                    # Clean potential NaN/Inf values (robustness check)
                    if pd.isna(value) or np.isinf(value):
                        value = 0.0
                    ticker_features.append(float(value))
                obs.extend(ticker_features)
                
            except KeyError:
                # If data is missing for this ticker at this specific timestamp
                obs.extend([0.0] * len(self.feature_cols))
        
        # 2. Portfolio state (Fix 4: Normalized)
        # Normalize equity relative to initial balance
        normalized_equity = self.equity / self.initial_balance
        
        # Position state per ticker (Size and Direction)
        position_features = []
        for ticker in self.unique_tickers:
            pos_size = self.positions.get(ticker, 0)
            position_features.append(pos_size)
            # Direction indicator (1 for long, -1 for short, 0 for flat)
            position_features.append(1 if pos_size > 0 else (-1 if pos_size < 0 else 0))

        portfolio_features = [
            normalized_equity,
            len(self.positions),
            self.current_drawdown,
            # Including risk metrics in observation (optional)
            self._calculate_sharpe_ratio(),
            self._calculate_calmar_ratio(),
            self._calculate_sortino_ratio(),
            self._calculate_win_rate(),
            self._calculate_profit_factor(),
            self._calculate_avg_trade(),
            self._calculate_max_consecutive_losses()
        ]
        
        portfolio_features.extend(position_features)

        # Clean and combine
        portfolio_features = [0.0 if pd.isna(x) or np.isinf(x) else float(x) for x in portfolio_features]
        obs.extend(portfolio_features)
        
        # Final validation
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs

    # =====================================================================================
    # Core Trading Logic and Accounting (Fix 1, 5)
    # =====================================================================================

    def _get_current_market_data(self, asset):
        """Helper to get current price and ATR for an asset using multi-index."""
        if self.current_step >= len(self.time_indices):
            return None, None

        current_time = self.time_indices[self.current_step]
        
        try:
            # Efficient lookup
            data = self.df.loc[(current_time, asset)]
            
            current_price = data['close']
            atr = data.get('atr')

            return current_price, atr
        except KeyError:
            # Handle missing data for the asset at this timestamp
            return None, None
        except Exception as e:
            print(f"Error accessing data for {asset} at {current_time}: {e}")
            return None, None

    def _execute_trade(self, asset, action):
        """Execute a trade based on action, implementing correct accounting and dynamic costs."""
        
        # Action threshold
        if abs(action) < 0.01: 
            return 0
        
        current_price, atr = self._get_current_market_data(asset)
        if current_price is None or atr is None or pd.isna(atr) or atr <= 0:
            return 0

        # 1. Calculate Position Size Change
        # The action dictates the fraction of the maximum allowed risk to take.
        position_size_change = self._calculate_position_size(asset, action, current_price, atr)
        
        # Minimum lot size threshold (micro lot)
        if abs(position_size_change) < 0.01: 
            return 0

        # 2. Calculate Dynamic Costs (Fix 5)
        # Dynamic Spread (based on ATR)
        spread_factor = self.config["transaction_fee_atr_factor"]
        dynamic_spread = atr * spread_factor
        spread_cost = abs(position_size_change) * self.LOT_SIZE * dynamic_spread
        
        # Dynamic Slippage (based on ATR)
        slippage_factor = self.config["slippage_atr_factor"]
        dynamic_slippage = atr * slippage_factor
        slippage_cost = abs(position_size_change) * self.LOT_SIZE * dynamic_slippage

        # Commission
        commission = abs(position_size_change) * self.config["commission_per_lot"]
        
        total_cost = spread_cost + slippage_cost + commission
        
        # Deduct costs from realized balance immediately
        self.balance -= total_cost

        # 3. Update Positions and PnL (Fix 1: Correct Accounting Logic)
        current_position = self.positions.get(asset, 0)
        current_entry = self.position_entries.get(asset, 0)
        
        new_position = current_position + position_size_change
        
        realized_pnl = 0
        closed_lots = 0

        if current_position == 0:
            # Opening a new position
            self.positions[asset] = new_position
            self.position_entries[asset] = current_price
        
        elif np.sign(new_position) == np.sign(current_position):
            # Scaling in (Increasing existing position)
            # Update weighted average entry price
            new_entry = (current_entry * current_position + current_price * position_size_change) / new_position
            self.positions[asset] = new_position
            self.position_entries[asset] = new_entry
            
        else:
            # Scaling out, Closing, or Flipping direction
            if abs(position_size_change) <= abs(current_position):
                # Partial close or full close
                # When reducing position, we realize PnL on the amount closed.
                closed_lots = abs(position_size_change)
                # PnL calculation: (Exit - Entry) * Lots * LotSize * Direction
                realized_pnl = (current_price - current_entry) * closed_lots * self.LOT_SIZE * np.sign(current_position)
                
                if abs(new_position) < 0.01:
                    # Full close
                    del self.positions[asset]
                    del self.position_entries[asset]
                else:
                    self.positions[asset] = new_position
                    # Entry price remains the same for the remaining position
            else:
                # Flipping position (Close existing and open new opposite position)
                # Realize PnL on the entire existing position
                closed_lots = abs(current_position)
                realized_pnl = (current_price - current_entry) * closed_lots * self.LOT_SIZE * np.sign(current_position)
                
                # The remainder of the action opens the new position at the current price
                self.positions[asset] = new_position
                self.position_entries[asset] = current_price

        # Update realized balance with realized PnL
        if realized_pnl != 0:
            self.balance += realized_pnl
            
            # Record the trade (only when PnL is realized)
            self.trade_history.append({
                'asset': asset,
                'pnl': realized_pnl,
                'costs': total_cost, # Associate costs with the action that caused realization
                'type': 'realized',
                'step': self.current_step
            })
        elif total_cost > 0 and realized_pnl == 0:
             # Optionally record costs associated with scaling in
             self.trade_history.append({
                'asset': asset,
                'pnl': 0,
                'costs': total_cost,
                'type': 'cost_scaling_in',
                'step': self.current_step
            })

        return -total_cost
    
    def _calculate_position_size(self, asset, action, current_price, atr):
        """Calculate position size based on risk and leverage constraints using Equity."""
        
        # Fix 1: Use Equity for risk calculations
        current_equity = self.equity
        
        # 1. Risk-based sizing (Volatility adjusted using ATR)
        risk_amount = current_equity * self.config["risk_per_trade"]
        
        # Ensure ATR is reasonable (Minimum stop loss distance proxy)
        min_atr = 0.0005 # e.g., 5 pips for standard pairs
        # Heuristic for JPY pairs which have different pip definitions
        if "JPY" in asset:
             min_atr = 0.05 

        atr = max(atr, min_atr)
        
        # Risk per lot assuming stop loss is 1 ATR away
        risk_per_lot = atr * self.LOT_SIZE
        
        max_lots_by_risk = risk_amount / risk_per_lot if risk_per_lot > 0 else 0
        
        # 2. Leverage constraint
        max_leverage = self.config["max_leverage"]
        max_portfolio_value = current_equity * max_leverage
        
        # Calculate current utilized portfolio value (Notional Value)
        current_portfolio_value = 0
        for pos_asset, pos_size in self.positions.items():
            # Need current price for the asset being held
            pos_price, _ = self._get_current_market_data(pos_asset)
            if pos_price:
                current_portfolio_value += abs(pos_size) * self.LOT_SIZE * pos_price

        remaining_value = max(0, max_portfolio_value - current_portfolio_value)
        max_lots_by_leverage = remaining_value / (current_price * self.LOT_SIZE) if current_price > 0 else 0
        
        # Final max lots is the minimum of constraints
        max_lots = min(max_lots_by_risk, max_lots_by_leverage)
        
        # Apply action scaling
        position_size_change = max_lots * action
        
        # Hard cap (e.g., max 10 standard lots per trade action)
        max_lots_per_trade = 10.0
        position_size_change = np.clip(position_size_change, -max_lots_per_trade, max_lots_per_trade)
        
        # Round to 2 decimal places (mini lots)
        position_size_change = round(position_size_change, 2)
        
        return position_size_change

    def _calculate_unrealized_pnl(self):
        """Calculate total unrealized P&L for all positions."""
        total_unrealized_pnl = 0
        
        for ticker, position_size in self.positions.items():
            if position_size != 0:
                current_price, _ = self._get_current_market_data(ticker)
                if current_price:
                    entry_price = self.position_entries[ticker]
                    price_change = current_price - entry_price
                    
                    # PnL calculation: (Price Change) * Position Size (Lots) * Lot Size (Units)
                    pnl = price_change * position_size * self.LOT_SIZE
                    total_unrealized_pnl += pnl
        
        return total_unrealized_pnl

    # =====================================================================================
    # Step Function (Fix 1: Equity based)
    # =====================================================================================

    def step(self, action):
        """Execute one step in the environment"""
        
        # 1. Execute Trades
        # This updates self.balance (realized cash) based on costs and realized PnL
        total_cost = 0
        for i, ticker in enumerate(self.unique_tickers):
            if i < len(action):
                trade_cost = self._execute_trade(ticker, action[i])
                total_cost += trade_cost

        # 2. Update Equity (Fix 1)
        # Equity = Realized Balance + Unrealized PnL
        unrealized_pnl = self._calculate_unrealized_pnl()
        self.equity = self.balance + unrealized_pnl
        
        # 3. Calculate Change in Equity
        change_in_equity = self.equity - self.previous_equity
        self.previous_equity = self.equity

        # 4. Update Metrics and History
        self.equity_curve.append(self.equity)
        
        # Track returns (based on equity change)
        current_return = 0.0
        if len(self.equity_curve) > 1 and self.equity_curve[-2] > 0:
             # Return = Change / Previous Value
             current_return = change_in_equity / self.equity_curve[-2]
             self.step_returns.append(current_return)
        
        # 5. Update Risk Metrics (Equity based)
        risk_stop = self._update_risk_metrics()
        
        # 6. Calculate Reward (Fix 3)
        reward = self._calculate_reward(change_in_equity, current_return)
        
        # 7. Move to next time step
        self.current_step += 1
        
        # 8. Check if episode is done
        done = (
            self.current_step >= self.max_steps - 1 or
            self.equity <= self.initial_balance * 0.1 or # Ruin condition (e.g. 90% loss)
            risk_stop           # Risk management stop
        )
        
        # If episode is done, close all positions to realize final balance
        if done:
            final_realized_pnl = self._close_all_positions()
            self.balance += final_realized_pnl
            self.equity = self.balance # Equity equals balance when flat

        # 9. Get Observation
        obs = self._get_observation()
        
        # 10. Info dictionary
        info = {
            "balance": self.balance,
            "equity": self.equity,
            "unrealized_pnl": unrealized_pnl,
            "change_in_equity": change_in_equity,
            "total_cost": total_cost,
            "positions": self.positions.copy(),
            "drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "sharpe": self._calculate_sharpe_ratio(),
            "return": current_return
        }
        
        # Gymnasium API compliance
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info

    # =====================================================================================
    # Reward Functions (Fix 3: DSR and Drawdown Penalty)
    # =====================================================================================

    def _calculate_reward(self, change_in_equity, current_return):
        """Calculate reward based on configured reward function."""
        reward_function = self.config["reward_function"]
        
        if reward_function == "simple":
            # Reward is simply the change in equity (P&L of the step)
            return change_in_equity
        
        elif reward_function == "drawdown_penalty":
            # P&L with a penalty that increases quadratically with drawdown
            penalty_factor = self.config["drawdown_penalty_factor"]
            # Scale the penalty relative to the initial balance for consistency across account sizes
            # This strongly encourages minimizing drawdown while maximizing profit
            penalty = penalty_factor * (self.current_drawdown**2) * self.initial_balance
            return change_in_equity - penalty
        
        elif reward_function == "dsr":
            # Differential Sharpe Ratio (DSR)
            # Measures the marginal contribution of the current return to the overall Sharpe ratio.
            
            gamma = self.config["dsr_gamma"]
            R_t = current_return # Use the return (percentage)
            
            # Calculate the update terms for the moments (A=E[R], B=E[R^2])
            delta_A = R_t - self.dsr_A
            delta_B = R_t**2 - self.dsr_B
            
            # Calculate DSR (d_t) based on Moody & Wu (2001)
            # Formula: (B*dA - 0.5*A*dB) / (B - A^2)^1.5
            numerator = self.dsr_B * delta_A - 0.5 * self.dsr_A * delta_B
            variance_estimate = self.dsr_B - self.dsr_A**2
            
            # Handle edge cases where variance is zero or negative (can happen early in estimation)
            if variance_estimate <= 1e-8:
                dsr_reward = 0
            else:
                denominator = (variance_estimate)**1.5
                dsr_reward = numerator / denominator

            # Update the moments using Exponential Moving Average (EMA)
            # Learning rate (eta) is (1 - gamma)
            eta = 1 - gamma
            self.dsr_A += eta * delta_A
            self.dsr_B += eta * delta_B
            
            # DSR rewards are unitless and often require scaling/clipping during training
            return dsr_reward
            
        else:
            # Default to simple P&L
            return change_in_equity

    # =====================================================================================
    # Risk Management and Metrics (Updated to use Equity)
    # =====================================================================================
    
    def _update_risk_metrics(self):
        """Update risk management metrics based on Equity."""
        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        else:
            self.current_drawdown = 0
            
        # Update max drawdown
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Check if max drawdown limit exceeded
        if self.current_drawdown > self.config["max_drawdown"]:
            return True # Stop trading signal
        
        return False

    # Metric calculations (Now correctly based on equity returns)
    def _calculate_sharpe_ratio(self, risk_free_rate=0.02):
        # Note: Assumes 252 trading periods per year. Adjust if using hourly data (e.g., 252*24).
        periods_per_year = 252 
        if len(self.step_returns) < 2:
            return 0
        returns = np.array(self.step_returns)
        excess_returns = returns - (risk_free_rate / periods_per_year)
        std_dev = np.std(excess_returns)
        if std_dev == 0:
            return 0
        return np.mean(excess_returns) / std_dev * np.sqrt(periods_per_year)
    
    def _calculate_calmar_ratio(self):
        if self.max_drawdown == 0:
            return 0
        total_return = (self.equity - self.initial_balance) / self.initial_balance
        return total_return / self.max_drawdown
    
    def _calculate_sortino_ratio(self, risk_free_rate=0.02):
        periods_per_year = 252
        if len(self.step_returns) < 2:
            return 0
        returns = np.array(self.step_returns)
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        std_dev_downside = np.std(downside_returns)
        if len(downside_returns) == 0 or std_dev_downside == 0:
            return 0
        return np.mean(excess_returns) / std_dev_downside * np.sqrt(periods_per_year)
    
    def _get_realized_trades(self):
        return [t for t in self.trade_history if t['type'] == 'realized']

    def _calculate_win_rate(self):
        realized_trades = self._get_realized_trades()
        if not realized_trades:
            return 0
        winning_trades = sum(1 for trade in realized_trades if trade['pnl'] > 0)
        return winning_trades / len(realized_trades)
    
    def _calculate_profit_factor(self):
        realized_trades = self._get_realized_trades()
        if not realized_trades:
            return 0
        gross_profit = sum(trade['pnl'] for trade in realized_trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in realized_trades if trade['pnl'] < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    def _calculate_avg_trade(self):
        realized_trades = self._get_realized_trades()
        if not realized_trades:
             return 0
        return np.mean([trade['pnl'] for trade in realized_trades])
    
    def _calculate_max_consecutive_losses(self):
        realized_trades = self._get_realized_trades()
        if not realized_trades:
            return 0
        max_consecutive = 0
        current_consecutive = 0
        for trade in realized_trades:
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            elif trade['pnl'] > 0:
                current_consecutive = 0
        return max_consecutive

    # =====================================================================================
    # Environment Lifecycle (Reset, Render, Helpers)
    # =====================================================================================

    def reset(self, *, seed=None, options=None):
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
        
        # Reset portfolio state
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.previous_equity = self.initial_balance
        
        self.positions = {}
        self.position_entries = {}
        self.trade_history = []
        self.equity_curve = [self.equity]
        
        # Reset risk metrics
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.peak_equity = self.equity
        self.step_returns = []
        
        # Reset DSR tracking
        self.dsr_A = 0
        self.dsr_B = 0
        
        return self._get_observation(), {}
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Equity: ${self.equity:,.2f} (Balance: ${self.balance:,.2f})")
            print(f"Positions: {self.positions}")
            print(f"Drawdown: {self.current_drawdown:.2%} (Max: {self.max_drawdown:.2%})")
            print(f"Sharpe Ratio: {self._calculate_sharpe_ratio():.2f}")

    def _close_all_positions(self):
        """Close all open positions at current market prices (End of episode liquidation)."""
        total_realized_pnl = 0
        
        # Iterate over a copy of the keys as we modify the dictionary
        for ticker in list(self.positions.keys()):
            position_size = self.positions[ticker]
            if position_size != 0:
                current_price, _ = self._get_current_market_data(ticker)
                if current_price:
                    entry_price = self.position_entries[ticker]
                    
                    # Calculate final P&L
                    pnl = (current_price - entry_price) * position_size * self.LOT_SIZE
                    total_realized_pnl += pnl
                    
                    # Record the final trade
                    self.trade_history.append({
                        'asset': ticker,
                        'pnl': pnl,
                        'costs': 0, # Assume no extra costs for final liquidation
                        'type': 'realized',
                        'step': self.current_step
                    })
                    
                    # Close position
                    del self.positions[ticker]
                    del self.position_entries[ticker]
        
        return total_realized_pnl

    def get_performance_metrics(self):
        """Get comprehensive performance metrics"""
        return {
            "total_return": (self.equity - self.initial_balance) / self.initial_balance,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "calmar_ratio": self._calculate_calmar_ratio(),
            "sortino_ratio": self._calculate_sortino_ratio(),
            "max_drawdown": self.max_drawdown,
            "win_rate": self._calculate_win_rate(),
            "profit_factor": self._calculate_profit_factor(),
            "avg_trade": self._calculate_avg_trade(),
            "max_consecutive_losses": self._calculate_max_consecutive_losses()
        }


# =====================================================================================
# Main Execution (Demo)
# =====================================================================================

def main():
    """Example usage of the Revised Best Practice Forex Environment"""
    print("\n🚀 Refactored Best Practice Forex Environment Demo")
    print("=" * 50)
    
    # Helper function to create synthetic data for demonstration
    def create_synthetic_data():
        print("\n=== Creating Synthetic Data ===")
        dates = pd.date_range(start="2024-01-01", periods=500, freq='h')
        tickers = ["EURUSD", "GBPUSD"]
        data = []
        for ticker in tickers:
            # Initialize price and seed for reproducibility
            np.random.seed(42 if ticker == "EURUSD" else 1337)
            price = 1.10 if ticker == "EURUSD" else 1.25
            for date in dates:
                # Simple random walk simulation
                change = np.random.normal(0, 0.0005)
                price += change
                data.append({
                    "time": date,
                    "tic": ticker,
                    "open": price,
                    "high": price + abs(np.random.normal(0, 0.0002)),
                    "low": price - abs(np.random.normal(0, 0.0002)),
                    "close": price,
                    "volume": np.random.randint(100, 1000)
                })
        return pd.DataFrame(data)

    try:
        # Use synthetic data for a reliable, reproducible example
        df = create_synthetic_data()
            
        print("✓ Data loaded successfully")
        
        # Configuration examples
        # Example using Drawdown Penalty
        config_penalty = {
            "initial_balance": 10000,
            "reward_function": "drawdown_penalty", 
            "drawdown_penalty_factor": 10
        }

        # Example using DSR
        config_dsr = {
            "initial_balance": 10000,
            "reward_function": "dsr", 
            "dsr_gamma": 0.99
        }
        
        # Choose configuration to test
        test_config = config_dsr

        # Create best practice environment
        print(f"\n=== Creating Environment (Reward: {test_config['reward_function']}) ===")
        
        env = BestPracticeForexEnv(df=df, config=test_config)
        
        # Test environment simulation
        print("\n=== Testing Environment Simulation (100 steps with random actions) ===")
        obs, info = env.reset(seed=42)
        
        for step in range(100):
            # Sample random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0:
                print(f"Step {step}: Equity: ${info['equity']:,.2f}, Reward: {reward:.4f}, Drawdown: {info['drawdown']:.2%}")
            
            if terminated or truncated:
                print(f"Episode finished at step {step}")
                break
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        print(f"\n=== Performance Metrics ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                # Format percentages correctly
                if key in ["total_return", "max_drawdown", "win_rate"]:
                     print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n🎉 Refactored Environment test completed!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if gymnasium is installed before running the main function
    try:
        import gymnasium
        main()
    except ImportError:
        print("\n❌ Error: Gymnasium library not found.")
        print("Please install it using: pip install gymnasium")