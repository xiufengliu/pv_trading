"""
Baseline Trading Strategies for PV Intraday Trading

This module implements the baseline strategies mentioned in the paper:
1. Spot-Only: No intraday trading (all deviations go to imbalance)
2. Deterministic: Perfect foresight of imbalance prices
3. Naive Forecast: Simple forecast-based trading
4. Simple Heuristic: Rule-based trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BaseTradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def make_decision(self, 
                     features: pd.Series,
                     market_state: pd.Series,
                     pv_state: pd.Series) -> float:
        """Make trading decision for a single time step"""
        pass
    
    def fit(self, 
            features: pd.DataFrame,
            market_data: pd.DataFrame,
            pv_data: pd.DataFrame) -> 'BaseTradingStrategy':
        """Fit the strategy (if needed)"""
        self.is_fitted = True
        return self
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make decisions for multiple time steps"""
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted first")
        
        decisions = []
        for i in range(len(features)):
            decision = self.make_decision(
                features.iloc[i],
                None,  # Will be handled by specific strategies
                None
            )
            decisions.append(decision)
        
        return np.array(decisions)


class SpotOnlyStrategy(BaseTradingStrategy):
    """Baseline strategy: No intraday trading"""
    
    def __init__(self):
        super().__init__("Spot-Only")
    
    def make_decision(self, 
                     features: pd.Series,
                     market_state: pd.Series,
                     pv_state: pd.Series) -> float:
        """Always return 0 (no trading)"""
        return 0.0
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return zeros for all time steps"""
        return np.zeros(len(features))


class DeterministicStrategy(BaseTradingStrategy):
    """Perfect foresight strategy using actual generation and imbalance prices"""
    
    def __init__(self, market_data: pd.DataFrame, pv_data: pd.DataFrame):
        super().__init__("Deterministic")
        self.market_data = market_data
        self.pv_data = pv_data
    
    def make_decision(self, 
                     features: pd.Series,
                     market_state: pd.Series = None,
                     pv_state: pd.Series = None) -> float:
        """Make optimal decision with perfect foresight"""
        
        # Get data for this timestamp
        timestamp = features.name
        market_row = self.market_data.loc[timestamp]
        pv_row = self.pv_data.loc[timestamp]
        
        da_commitment = pv_row['da_forecast']
        actual_generation = pv_row['actual_generation']
        
        id_ask_price = market_row['id_ask_price']
        id_bid_price = market_row['id_bid_price']
        imbalance_price = market_row['imbalance_price']
        
        # Calculate optimal action with perfect foresight
        imbalance_without_trade = da_commitment - actual_generation
        
        if imbalance_without_trade > 0:  # Surplus (will be selling excess)
            # Should we sell in intraday?
            if id_ask_price > imbalance_price:
                return imbalance_without_trade  # Sell surplus
            else:
                return 0.0  # Don't trade
        else:  # Deficit (will be buying shortfall)
            # Should we buy in intraday?
            if id_bid_price < imbalance_price:
                return imbalance_without_trade  # Buy deficit (negative value)
            else:
                return 0.0  # Don't trade
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make optimal decisions for all time steps"""
        decisions = []
        
        for timestamp in features.index:
            decision = self.make_decision(features.loc[timestamp])
            decisions.append(decision)
        
        return np.array(decisions)


class NaiveForecastStrategy(BaseTradingStrategy):
    """Simple forecast-based trading strategy"""
    
    def __init__(self, 
                 threshold: float = 0.5,  # MW threshold for trading
                 aggressiveness: float = 0.8):  # Fraction of forecast deviation to trade
        super().__init__("Naive-Forecast")
        self.threshold = threshold
        self.aggressiveness = aggressiveness
    
    def make_decision(self, 
                     features: pd.Series,
                     market_state: pd.Series = None,
                     pv_state: pd.Series = None) -> float:
        """Trade based on forecast deviation"""
        
        # Use forecast deviation feature
        if 'forecast_deviation' in features:
            forecast_deviation = features['forecast_deviation']
        else:
            # Fallback: assume we have access to the deviation somehow
            forecast_deviation = 0.0
        
        # Only trade if deviation is significant
        if abs(forecast_deviation) < self.threshold:
            return 0.0
        
        # Trade a fraction of the forecast deviation
        return forecast_deviation * self.aggressiveness


class SimpleHeuristicStrategy(BaseTradingStrategy):
    """Rule-based heuristic strategy"""
    
    def __init__(self, 
                 price_threshold: float = 5.0,  # EUR/MWh price difference threshold
                 volume_threshold: float = 0.5):  # MW volume threshold
        super().__init__("Simple-Heuristic")
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold
    
    def make_decision(self, 
                     features: pd.Series,
                     market_state: pd.Series = None,
                     pv_state: pd.Series = None) -> float:
        """Make decision based on simple rules"""
        
        # Get relevant features
        forecast_deviation = features.get('forecast_deviation', 0.0)
        id_spread = features.get('id_spread', 0.0)
        price_trend = features.get('price_trend', 0.0)
        
        # Don't trade if deviation is too small
        if abs(forecast_deviation) < self.volume_threshold:
            return 0.0
        
        # Don't trade if spread is too wide
        if id_spread > self.price_threshold:
            return 0.0
        
        # Consider price trend
        if forecast_deviation > 0:  # Surplus
            # Sell if prices are trending up or stable
            if price_trend >= -1.0:
                return forecast_deviation * 0.7
        else:  # Deficit
            # Buy if prices are trending down or stable
            if price_trend <= 1.0:
                return forecast_deviation * 0.7
        
        return 0.0


class MeanReversionStrategy(BaseTradingStrategy):
    """Mean reversion strategy based on price patterns"""
    
    def __init__(self, 
                 lookback_window: int = 24,
                 reversion_threshold: float = 1.5):  # Standard deviations
        super().__init__("Mean-Reversion")
        self.lookback_window = lookback_window
        self.reversion_threshold = reversion_threshold
        self.price_history = []
    
    def make_decision(self, 
                     features: pd.Series,
                     market_state: pd.Series = None,
                     pv_state: pd.Series = None) -> float:
        """Trade based on mean reversion signals"""
        
        current_price = features.get('da_price', 50.0)
        forecast_deviation = features.get('forecast_deviation', 0.0)
        
        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.lookback_window:
            self.price_history.pop(0)
        
        # Need sufficient history
        if len(self.price_history) < self.lookback_window:
            return 0.0
        
        # Calculate mean and std
        mean_price = np.mean(self.price_history)
        std_price = np.std(self.price_history)
        
        if std_price == 0:
            return 0.0
        
        # Z-score
        z_score = (current_price - mean_price) / std_price
        
        # Mean reversion signal
        if abs(z_score) > self.reversion_threshold:
            # Price is extreme, expect reversion
            reversion_signal = -np.sign(z_score) * min(abs(z_score), 3.0) / 3.0
            
            # Combine with forecast deviation
            if abs(forecast_deviation) > 0.5:
                return forecast_deviation * 0.5 + reversion_signal * abs(forecast_deviation) * 0.3
        
        # Default to simple forecast-based trading
        return forecast_deviation * 0.6 if abs(forecast_deviation) > 0.5 else 0.0


class AdaptiveThresholdStrategy(BaseTradingStrategy):
    """Adaptive threshold strategy that learns from recent performance"""
    
    def __init__(self, 
                 initial_threshold: float = 0.5,
                 adaptation_rate: float = 0.1,
                 performance_window: int = 168):  # 1 week
        super().__init__("Adaptive-Threshold")
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        self.recent_profits = []
        self.recent_decisions = []
    
    def make_decision(self, 
                     features: pd.Series,
                     market_state: pd.Series = None,
                     pv_state: pd.Series = None) -> float:
        """Make adaptive decision based on recent performance"""
        
        forecast_deviation = features.get('forecast_deviation', 0.0)
        
        # Adapt threshold based on recent performance
        if len(self.recent_profits) >= 10:
            avg_profit = np.mean(self.recent_profits[-self.performance_window:])
            if avg_profit < 0:
                # Increase threshold (trade less)
                self.threshold = min(self.threshold * (1 + self.adaptation_rate), 2.0)
            else:
                # Decrease threshold (trade more)
                self.threshold = max(self.threshold * (1 - self.adaptation_rate), 0.1)
        
        # Make decision
        if abs(forecast_deviation) < self.threshold:
            decision = 0.0
        else:
            decision = forecast_deviation * 0.8
        
        # Store decision for adaptation
        self.recent_decisions.append(decision)
        
        return decision
    
    def update_performance(self, profit: float):
        """Update performance history for adaptation"""
        self.recent_profits.append(profit)
        if len(self.recent_profits) > self.performance_window:
            self.recent_profits.pop(0)


def create_all_baseline_strategies(market_data: pd.DataFrame, 
                                  pv_data: pd.DataFrame) -> Dict[str, BaseTradingStrategy]:
    """Create all baseline strategies"""
    
    strategies = {
        'spot_only': SpotOnlyStrategy(),
        'deterministic': DeterministicStrategy(market_data, pv_data),
        'naive_forecast': NaiveForecastStrategy(),
        'simple_heuristic': SimpleHeuristicStrategy(),
        'mean_reversion': MeanReversionStrategy(),
        'adaptive_threshold': AdaptiveThresholdStrategy()
    }
    
    return strategies


def evaluate_baseline_strategies(strategies: Dict[str, BaseTradingStrategy],
                                features: pd.DataFrame,
                                market_data: pd.DataFrame,
                                pv_data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all baseline strategies"""
    
    from market_environment import MarketSimulator
    
    simulator = MarketSimulator(market_data, pv_data)
    results = {}
    
    for name, strategy in strategies.items():
        print(f"Evaluating {name} strategy...")
        
        # Fit strategy if needed
        strategy.fit(features, market_data, pv_data)
        
        # Create decision function
        def decision_func(feat):
            return strategy.predict(feat)
        
        # Simulate strategy
        simulation_results = simulator.simulate_strategy(decision_func, features)
        
        # Calculate summary statistics
        total_profit = simulation_results['total_profit'].sum()
        avg_profit = simulation_results['total_profit'].mean()
        profit_std = simulation_results['total_profit'].std()
        
        n_trades = len(simulation_results[simulation_results['action'] != 'hold'])
        trade_ratio = n_trades / len(simulation_results)
        
        total_imbalance_cost = simulation_results['imbalance_cost'].sum()
        avg_imbalance = simulation_results['final_imbalance'].abs().mean()
        
        results[name] = {
            'total_profit': total_profit,
            'avg_profit_per_hour': avg_profit,
            'profit_std': profit_std,
            'sharpe_ratio': avg_profit / (profit_std + 1e-6),
            'n_trades': n_trades,
            'trade_ratio': trade_ratio,
            'total_imbalance_cost': total_imbalance_cost,
            'avg_abs_imbalance': avg_imbalance
        }
    
    return pd.DataFrame(results).T


if __name__ == "__main__":
    # Example usage
    from data_processing import load_danish_data
    
    # Load data
    processor = load_danish_data()
    train_data, test_data = processor.get_training_data()
    
    # Get corresponding market and PV data
    test_market = processor.market_data.loc[test_data.index]
    test_pv = processor.pv_data.loc[test_data.index]
    
    # Create baseline strategies
    strategies = create_all_baseline_strategies(test_market, test_pv)
    
    # Evaluate strategies
    results = evaluate_baseline_strategies(strategies, test_data, test_market, test_pv)
    
    print("Baseline Strategy Results:")
    print(results.round(4))
