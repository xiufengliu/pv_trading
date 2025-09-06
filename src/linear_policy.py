"""
Linear Feature-Driven Trading Policy

This module implements the linear decision rule approach described in the paper:
d_t = q^T * X_t

Where:
- d_t is the decision score for time t
- q is the learned weight vector
- X_t is the feature vector at time t
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class LinearTradingPolicy:
    """Linear feature-driven trading policy implementation"""
    
    def __init__(self, 
                 regularization: str = 'ridge',
                 alpha: float = 1.0,
                 threshold: float = 0.1):
        """
        Initialize the linear trading policy
        
        Args:
            regularization: Type of regularization ('ridge', 'lasso', 'none')
            alpha: Regularization strength
            threshold: Decision threshold to avoid marginal trades
        """
        self.regularization = regularization
        self.alpha = alpha
        self.threshold = threshold
        self.weights = None
        self.feature_names = None
        self.is_fitted = False
        
    def _calculate_profit(self, 
                         decisions: np.ndarray,
                         market_data: pd.DataFrame,
                         pv_data: pd.DataFrame) -> np.ndarray:
        """Calculate profit for given decisions"""
        
        profits = np.zeros(len(decisions))
        
        for i, decision in enumerate(decisions):
            # Get market data for this time step
            da_commitment = pv_data.iloc[i]['da_forecast']
            actual_generation = pv_data.iloc[i]['actual_generation']
            
            id_ask_price = market_data.iloc[i]['id_ask_price']
            id_bid_price = market_data.iloc[i]['id_bid_price']
            imbalance_price = market_data.iloc[i]['imbalance_price']
            
            liquidity_ask = market_data.iloc[i]['liquidity_ask']
            liquidity_bid = market_data.iloc[i]['liquidity_bid']
            
            # Determine action based on decision score
            if decision > self.threshold:
                # Sell action
                forecast_surplus = max(0, pv_data.iloc[i]['id_forecast'] - da_commitment)
                volume = min(forecast_surplus, abs(decision), liquidity_ask)
                
                if volume > 0:
                    # Revenue from intraday sale
                    intraday_revenue = volume * id_ask_price
                    
                    # Calculate final imbalance
                    final_position = da_commitment + volume
                    imbalance = final_position - actual_generation
                    imbalance_cost = imbalance * imbalance_price
                    
                    profits[i] = intraday_revenue - imbalance_cost
                else:
                    # No trade, only imbalance cost
                    imbalance = da_commitment - actual_generation
                    profits[i] = -imbalance * imbalance_price
                    
            elif decision < -self.threshold:
                # Buy action
                forecast_deficit = max(0, da_commitment - pv_data.iloc[i]['id_forecast'])
                volume = min(forecast_deficit, abs(decision), liquidity_bid)
                
                if volume > 0:
                    # Cost of intraday purchase
                    intraday_cost = volume * id_bid_price
                    
                    # Calculate final imbalance
                    final_position = da_commitment - volume
                    imbalance = final_position - actual_generation
                    imbalance_cost = imbalance * imbalance_price
                    
                    profits[i] = -intraday_cost - imbalance_cost
                else:
                    # No trade, only imbalance cost
                    imbalance = da_commitment - actual_generation
                    profits[i] = -imbalance * imbalance_price
            else:
                # No trade (decision near zero)
                imbalance = da_commitment - actual_generation
                profits[i] = -imbalance * imbalance_price
        
        return profits
    
    def fit(self, 
            features: pd.DataFrame,
            market_data: pd.DataFrame,
            pv_data: pd.DataFrame,
            method: str = 'profit_optimization') -> 'LinearTradingPolicy':
        """
        Train the linear policy
        
        Args:
            features: Feature matrix X_t
            market_data: Market price and liquidity data
            pv_data: PV generation and forecast data
            method: Training method ('profit_optimization' or 'supervised')
        """
        
        self.feature_names = features.columns.tolist()
        X = features.values
        
        if method == 'profit_optimization':
            # Direct profit optimization approach
            def objective(weights):
                decisions = X @ weights
                profits = self._calculate_profit(decisions, market_data, pv_data)
                return -np.sum(profits)  # Minimize negative profit
            
            # Initial guess
            initial_weights = np.random.normal(0, 0.1, X.shape[1])
            
            # Optimize
            result = minimize(objective, initial_weights, method='L-BFGS-B')
            self.weights = result.x
            
        elif method == 'supervised':
            # Supervised learning approach - predict optimal decisions
            optimal_decisions = self._compute_optimal_decisions(market_data, pv_data)
            
            if self.regularization == 'ridge':
                model = Ridge(alpha=self.alpha)
            elif self.regularization == 'lasso':
                model = Lasso(alpha=self.alpha)
            else:
                model = LinearRegression()
            
            model.fit(X, optimal_decisions)
            self.weights = model.coef_
            
        else:
            raise ValueError(f"Unknown training method: {method}")
        
        self.is_fitted = True
        return self
    
    def _compute_optimal_decisions(self, 
                                  market_data: pd.DataFrame,
                                  pv_data: pd.DataFrame) -> np.ndarray:
        """Compute optimal decisions with perfect foresight (for supervised learning)"""
        
        optimal_decisions = np.zeros(len(market_data))
        
        for i in range(len(market_data)):
            da_commitment = pv_data.iloc[i]['da_forecast']
            actual_generation = pv_data.iloc[i]['actual_generation']
            
            id_ask_price = market_data.iloc[i]['id_ask_price']
            id_bid_price = market_data.iloc[i]['id_bid_price']
            imbalance_price = market_data.iloc[i]['imbalance_price']
            
            # Calculate optimal action with perfect foresight
            imbalance_without_trade = da_commitment - actual_generation
            
            if imbalance_without_trade > 0:  # Surplus
                # Should we sell in intraday?
                if id_ask_price > imbalance_price:
                    optimal_decisions[i] = imbalance_without_trade  # Sell surplus
                else:
                    optimal_decisions[i] = 0  # Don't trade
            else:  # Deficit
                # Should we buy in intraday?
                if id_bid_price < imbalance_price:
                    optimal_decisions[i] = imbalance_without_trade  # Buy deficit (negative)
                else:
                    optimal_decisions[i] = 0  # Don't trade
        
        return optimal_decisions
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make trading decisions for given features"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = features.values
        decisions = X @ self.weights
        return decisions
    
    def get_trading_actions(self, 
                           features: pd.DataFrame,
                           market_data: pd.DataFrame,
                           pv_data: pd.DataFrame) -> pd.DataFrame:
        """Get detailed trading actions and volumes"""
        
        decisions = self.predict(features)
        actions = []
        
        for i, decision in enumerate(decisions):
            da_commitment = pv_data.iloc[i]['da_forecast']
            id_forecast = pv_data.iloc[i]['id_forecast']
            
            liquidity_ask = market_data.iloc[i]['liquidity_ask']
            liquidity_bid = market_data.iloc[i]['liquidity_bid']
            
            if decision > self.threshold:
                # Sell action
                forecast_surplus = max(0, id_forecast - da_commitment)
                volume = min(forecast_surplus, abs(decision), liquidity_ask)
                action = 'sell' if volume > 0 else 'no_trade'
            elif decision < -self.threshold:
                # Buy action
                forecast_deficit = max(0, da_commitment - id_forecast)
                volume = min(forecast_deficit, abs(decision), liquidity_bid)
                action = 'buy' if volume > 0 else 'no_trade'
            else:
                action = 'no_trade'
                volume = 0
            
            actions.append({
                'datetime': features.index[i],
                'decision_score': decision,
                'action': action,
                'volume': volume,
                'forecast_deviation': id_forecast - da_commitment
            })
        
        return pd.DataFrame(actions).set_index('datetime')
    
    def evaluate(self, 
                features: pd.DataFrame,
                market_data: pd.DataFrame,
                pv_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the policy performance"""
        
        decisions = self.predict(features)
        profits = self._calculate_profit(decisions, market_data, pv_data)
        actions_df = self.get_trading_actions(features, market_data, pv_data)
        
        # Calculate metrics
        total_profit = np.sum(profits)
        avg_profit = np.mean(profits)
        profit_std = np.std(profits)
        
        # Trading statistics
        n_trades = len(actions_df[actions_df['action'] != 'no_trade'])
        n_sells = len(actions_df[actions_df['action'] == 'sell'])
        n_buys = len(actions_df[actions_df['action'] == 'buy'])
        
        # Imbalance reduction
        baseline_imbalances = pv_data['da_forecast'] - pv_data['actual_generation']
        baseline_imbalance_cost = np.sum(np.abs(baseline_imbalances) * market_data['imbalance_price'])
        
        # Calculate actual imbalances after trading
        actual_imbalances = []
        for i, decision in enumerate(decisions):
            da_commitment = pv_data.iloc[i]['da_forecast']
            actual_generation = pv_data.iloc[i]['actual_generation']
            
            if decision > self.threshold:
                volume = min(max(0, pv_data.iloc[i]['id_forecast'] - da_commitment), 
                           abs(decision), market_data.iloc[i]['liquidity_ask'])
                final_position = da_commitment + volume
            elif decision < -self.threshold:
                volume = min(max(0, da_commitment - pv_data.iloc[i]['id_forecast']), 
                           abs(decision), market_data.iloc[i]['liquidity_bid'])
                final_position = da_commitment - volume
            else:
                final_position = da_commitment
            
            actual_imbalances.append(final_position - actual_generation)
        
        actual_imbalance_cost = np.sum(np.abs(actual_imbalances) * market_data['imbalance_price'])
        imbalance_reduction = (baseline_imbalance_cost - actual_imbalance_cost) / baseline_imbalance_cost
        
        return {
            'total_profit': total_profit,
            'avg_profit_per_hour': avg_profit,
            'profit_std': profit_std,
            'sharpe_ratio': avg_profit / (profit_std + 1e-6),
            'n_trades': n_trades,
            'n_sells': n_sells,
            'n_buys': n_buys,
            'trade_ratio': n_trades / len(features),
            'imbalance_reduction': imbalance_reduction,
            'baseline_imbalance_cost': baseline_imbalance_cost,
            'actual_imbalance_cost': actual_imbalance_cost
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (weights)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'weight': self.weights,
            'abs_weight': np.abs(self.weights)
        })
        
        return importance_df.sort_values('abs_weight', ascending=False)


if __name__ == "__main__":
    # Example usage
    from data_processing import load_danish_data
    
    # Load data
    processor = load_danish_data()
    train_data, test_data = processor.get_training_data()
    
    # Get corresponding market and PV data
    train_market = processor.market_data.loc[train_data.index]
    train_pv = processor.pv_data.loc[train_data.index]
    test_market = processor.market_data.loc[test_data.index]
    test_pv = processor.pv_data.loc[test_data.index]
    
    # Train policy
    policy = LinearTradingPolicy(regularization='ridge', alpha=0.1)
    policy.fit(train_data, train_market, train_pv, method='profit_optimization')
    
    # Evaluate
    train_results = policy.evaluate(train_data, train_market, train_pv)
    test_results = policy.evaluate(test_data, test_market, test_pv)
    
    print("Training Results:")
    for key, value in train_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nFeature Importance:")
    print(policy.get_feature_importance().head(10))
