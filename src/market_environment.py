"""
Market Environment Simulator for PV Intraday Trading

This module implements a realistic simulation environment for the Danish intraday market
that can be used for both the linear policy and reinforcement learning approaches.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MarketState:
    """Represents the current state of the market"""
    datetime: pd.Timestamp
    da_commitment: float
    id_forecast: float
    actual_generation: float
    da_price: float
    id_ask_price: float
    id_bid_price: float
    imbalance_price: float
    liquidity_ask: float
    liquidity_bid: float
    features: np.ndarray


@dataclass
class TradingAction:
    """Represents a trading action"""
    action_type: str  # 'buy', 'sell', 'hold'
    volume: float
    price: float


@dataclass
class TradingResult:
    """Results of a trading action"""
    profit: float
    imbalance_cost: float
    trading_cost: float
    final_imbalance: float
    executed_volume: float


class IntradayMarketEnvironment(gym.Env):
    """
    Gym environment for intraday PV trading
    
    This environment simulates the Danish DK1 intraday market with:
    - Continuous trading
    - Market liquidity constraints
    - Imbalance pricing
    - Transaction costs
    """
    
    def __init__(self, 
                 market_data: pd.DataFrame,
                 pv_data: pd.DataFrame,
                 features: pd.DataFrame,
                 transaction_cost: float = 0.1,  # EUR/MWh
                 max_position: float = 50.0,     # MW
                 episode_length: int = 168):     # 1 week
        
        super().__init__()
        
        self.market_data = market_data
        self.pv_data = pv_data
        self.features = features
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.episode_length = episode_length
        
        # Align all data
        common_index = market_data.index.intersection(pv_data.index).intersection(features.index)
        self.market_data = market_data.loc[common_index]
        self.pv_data = pv_data.loc[common_index]
        self.features = features.loc[common_index]
        
        # Define action and observation spaces
        # Action: [action_type, volume_fraction]
        # action_type: 0=hold, 1=buy, 2=sell
        # volume_fraction: 0-1 (fraction of available liquidity/forecast deviation)
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([2, 1]), 
            dtype=np.float32
        )
        
        # Observation space: normalized features
        n_features = len(self.features.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to start a new episode"""
        
        # Random starting point
        max_start = len(self.market_data) - self.episode_length
        self.start_idx = np.random.randint(0, max_start)
        self.current_step = 0
        
        # Initialize episode data
        self.episode_profits = []
        self.episode_actions = []
        self.episode_states = []
        
        # Get initial state
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one trading step"""
        
        current_idx = self.start_idx + self.current_step
        
        # Get current market state
        state = self._get_market_state(current_idx)
        
        # Parse action
        action_type = int(np.round(action[0]))
        volume_fraction = np.clip(action[1], 0, 1)
        
        # Execute trading action
        trading_action = self._parse_action(action_type, volume_fraction, state)
        result = self._execute_trade(trading_action, state)
        
        # Calculate reward
        reward = self._calculate_reward(result, state)
        
        # Store episode data
        self.episode_profits.append(result.profit)
        self.episode_actions.append(trading_action)
        self.episode_states.append(state)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        # Get next observation
        next_obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        # Info dictionary
        info = {
            'profit': result.profit,
            'imbalance_cost': result.imbalance_cost,
            'trading_cost': result.trading_cost,
            'final_imbalance': result.final_imbalance,
            'executed_volume': result.executed_volume,
            'action_type': trading_action.action_type,
            'state': state
        }
        
        return next_obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (normalized features)"""
        current_idx = self.start_idx + self.current_step
        
        if current_idx >= len(self.features):
            return np.zeros(self.observation_space.shape)
        
        return self.features.iloc[current_idx].values.astype(np.float32)
    
    def _get_market_state(self, idx: int) -> MarketState:
        """Get complete market state for given index"""
        
        return MarketState(
            datetime=self.market_data.index[idx],
            da_commitment=self.pv_data.iloc[idx]['da_forecast'],
            id_forecast=self.pv_data.iloc[idx]['id_forecast'],
            actual_generation=self.pv_data.iloc[idx]['actual_generation'],
            da_price=self.market_data.iloc[idx]['da_price'],
            id_ask_price=self.market_data.iloc[idx]['id_ask_price'],
            id_bid_price=self.market_data.iloc[idx]['id_bid_price'],
            imbalance_price=self.market_data.iloc[idx]['imbalance_price'],
            liquidity_ask=self.market_data.iloc[idx]['liquidity_ask'],
            liquidity_bid=self.market_data.iloc[idx]['liquidity_bid'],
            features=self.features.iloc[idx].values
        )
    
    def _parse_action(self, action_type: int, volume_fraction: float, state: MarketState) -> TradingAction:
        """Parse raw action into trading action"""
        
        if action_type == 0:  # Hold
            return TradingAction('hold', 0.0, 0.0)
        
        elif action_type == 1:  # Buy
            # Calculate maximum buy volume
            forecast_deficit = max(0, state.da_commitment - state.id_forecast)
            max_volume = min(forecast_deficit, state.liquidity_bid, self.max_position)
            volume = max_volume * volume_fraction
            
            return TradingAction('buy', volume, state.id_bid_price)
        
        elif action_type == 2:  # Sell
            # Calculate maximum sell volume
            forecast_surplus = max(0, state.id_forecast - state.da_commitment)
            max_volume = min(forecast_surplus, state.liquidity_ask, self.max_position)
            volume = max_volume * volume_fraction
            
            return TradingAction('sell', volume, state.id_ask_price)
        
        else:
            return TradingAction('hold', 0.0, 0.0)
    
    def _execute_trade(self, action: TradingAction, state: MarketState) -> TradingResult:
        """Execute trading action and calculate results"""
        
        if action.action_type == 'hold' or action.volume <= 0:
            # No trade - only imbalance cost
            imbalance = state.da_commitment - state.actual_generation
            imbalance_cost = abs(imbalance) * abs(state.imbalance_price)
            
            return TradingResult(
                profit=-imbalance_cost,
                imbalance_cost=imbalance_cost,
                trading_cost=0.0,
                final_imbalance=imbalance,
                executed_volume=0.0
            )
        
        # Calculate trading costs
        trading_cost = action.volume * self.transaction_cost
        
        if action.action_type == 'buy':
            # Buy energy in intraday market
            intraday_cost = action.volume * action.price
            final_position = state.da_commitment - action.volume
            
        elif action.action_type == 'sell':
            # Sell energy in intraday market
            intraday_cost = -action.volume * action.price  # Revenue (negative cost)
            final_position = state.da_commitment + action.volume
        
        else:
            intraday_cost = 0
            final_position = state.da_commitment
        
        # Calculate final imbalance
        final_imbalance = final_position - state.actual_generation
        imbalance_cost = abs(final_imbalance) * abs(state.imbalance_price)
        
        # Total profit
        profit = -intraday_cost - trading_cost - imbalance_cost
        
        return TradingResult(
            profit=profit,
            imbalance_cost=imbalance_cost,
            trading_cost=trading_cost,
            final_imbalance=final_imbalance,
            executed_volume=action.volume
        )
    
    def _calculate_reward(self, result: TradingResult, state: MarketState) -> float:
        """Calculate reward for the trading action"""
        
        # Base reward is the profit
        reward = result.profit
        
        # Add penalty for large imbalances (risk management)
        imbalance_penalty = -0.1 * abs(result.final_imbalance)
        
        # Add small penalty for excessive trading (transaction costs)
        trading_penalty = -0.01 * result.executed_volume
        
        total_reward = reward + imbalance_penalty + trading_penalty
        
        # Normalize reward (optional)
        return total_reward / 100.0  # Scale down for better learning
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the completed episode"""
        
        if not self.episode_profits:
            return {}
        
        total_profit = sum(self.episode_profits)
        avg_profit = np.mean(self.episode_profits)
        profit_std = np.std(self.episode_profits)
        
        # Count actions
        action_counts = {}
        for action in self.episode_actions:
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1
        
        return {
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'profit_std': profit_std,
            'sharpe_ratio': avg_profit / (profit_std + 1e-6),
            'action_counts': action_counts,
            'n_steps': len(self.episode_profits)
        }


class MarketSimulator:
    """Simplified market simulator for backtesting strategies"""
    
    def __init__(self, 
                 market_data: pd.DataFrame,
                 pv_data: pd.DataFrame,
                 transaction_cost: float = 0.1):
        
        self.market_data = market_data
        self.pv_data = pv_data
        self.transaction_cost = transaction_cost
    
    def simulate_strategy(self, 
                         strategy_func,
                         features: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate a trading strategy
        
        Args:
            strategy_func: Function that takes features and returns decisions
            features: Feature matrix for decision making
        
        Returns:
            DataFrame with detailed results for each time step
        """
        
        results = []
        
        for i in range(len(features)):
            # Get current state
            current_features = features.iloc[i:i+1]
            market_state = self.market_data.iloc[i]
            pv_state = self.pv_data.iloc[i]
            
            # Get strategy decision
            decision = strategy_func(current_features)
            if hasattr(decision, '__len__'):
                decision = decision[0]
            
            # Execute trade
            result = self._execute_single_trade(decision, market_state, pv_state)
            result['datetime'] = features.index[i]
            result['decision'] = decision
            
            results.append(result)
        
        return pd.DataFrame(results).set_index('datetime')
    
    def _execute_single_trade(self, 
                             decision: float,
                             market_state: pd.Series,
                             pv_state: pd.Series) -> Dict[str, float]:
        """Execute a single trade and return results"""
        
        da_commitment = pv_state['da_forecast']
        actual_generation = pv_state['actual_generation']
        
        # Determine action based on decision
        threshold = 0.1
        
        if decision > threshold:
            # Sell action
            forecast_surplus = max(0, pv_state['id_forecast'] - da_commitment)
            volume = min(forecast_surplus, abs(decision), market_state['liquidity_ask'])
            
            if volume > 0:
                trading_revenue = volume * market_state['id_ask_price']
                trading_cost = volume * self.transaction_cost
                final_position = da_commitment + volume
                action = 'sell'
            else:
                trading_revenue = 0
                trading_cost = 0
                final_position = da_commitment
                volume = 0
                action = 'hold'
                
        elif decision < -threshold:
            # Buy action
            forecast_deficit = max(0, da_commitment - pv_state['id_forecast'])
            volume = min(forecast_deficit, abs(decision), market_state['liquidity_bid'])
            
            if volume > 0:
                trading_revenue = -volume * market_state['id_bid_price']  # Cost (negative revenue)
                trading_cost = volume * self.transaction_cost
                final_position = da_commitment - volume
                action = 'buy'
            else:
                trading_revenue = 0
                trading_cost = 0
                final_position = da_commitment
                volume = 0
                action = 'hold'
        else:
            # Hold
            trading_revenue = 0
            trading_cost = 0
            final_position = da_commitment
            volume = 0
            action = 'hold'
        
        # Calculate imbalance
        final_imbalance = final_position - actual_generation
        imbalance_cost = abs(final_imbalance) * abs(market_state['imbalance_price'])
        
        # Total profit
        total_profit = trading_revenue - trading_cost - imbalance_cost
        
        return {
            'action': action,
            'volume': volume,
            'trading_revenue': trading_revenue,
            'trading_cost': trading_cost,
            'imbalance_cost': imbalance_cost,
            'total_profit': total_profit,
            'final_imbalance': final_imbalance,
            'da_commitment': da_commitment,
            'actual_generation': actual_generation
        }


if __name__ == "__main__":
    # Example usage
    from data_processing import load_danish_data
    
    # Load data
    processor = load_danish_data()
    train_data, test_data = processor.get_training_data()
    
    # Create environment
    env = IntradayMarketEnvironment(
        market_data=processor.market_data.loc[train_data.index],
        pv_data=processor.pv_data.loc[train_data.index],
        features=train_data
    )
    
    # Test random policy
    obs = env.reset()
    total_reward = 0
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"Total reward: {total_reward:.4f}")
    print("Episode summary:", env.get_episode_summary())
