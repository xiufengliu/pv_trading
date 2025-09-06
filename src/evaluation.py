"""
Evaluation and Visualization Framework for PV Trading Strategies

This module provides comprehensive evaluation metrics, statistical testing,
and visualization tools for comparing different trading strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TradingEvaluator:
    """Comprehensive evaluation framework for trading strategies"""
    
    def __init__(self):
        self.results = {}
        self.simulation_data = {}
    
    def add_strategy_results(self, 
                           strategy_name: str,
                           simulation_results: pd.DataFrame,
                           strategy_decisions: Optional[pd.DataFrame] = None):
        """Add results for a trading strategy"""
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(simulation_results)
        
        self.results[strategy_name] = metrics
        self.simulation_data[strategy_name] = {
            'simulation': simulation_results,
            'decisions': strategy_decisions
        }
    
    def _calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        profits = results['total_profit']
        
        # Basic profit metrics
        total_profit = profits.sum()
        avg_profit = profits.mean()
        profit_std = profits.std()
        
        # Risk-adjusted metrics
        sharpe_ratio = avg_profit / (profit_std + 1e-6)
        max_drawdown = self._calculate_max_drawdown(profits.cumsum())
        
        # Trading activity metrics
        n_trades = len(results[results['action'] != 'hold'])
        trade_ratio = n_trades / len(results)
        
        # Volume metrics
        total_volume = results['volume'].sum()
        avg_volume = results[results['volume'] > 0]['volume'].mean() if n_trades > 0 else 0
        
        # Imbalance metrics
        total_imbalance_cost = results['imbalance_cost'].sum()
        avg_imbalance = results['final_imbalance'].abs().mean()
        imbalance_reduction = self._calculate_imbalance_reduction(results)
        
        # Win rate
        profitable_hours = len(results[results['total_profit'] > 0])
        win_rate = profitable_hours / len(results)
        
        # Profit factor
        gross_profit = results[results['total_profit'] > 0]['total_profit'].sum()
        gross_loss = abs(results[results['total_profit'] < 0]['total_profit'].sum())
        profit_factor = gross_profit / (gross_loss + 1e-6)
        
        return {
            'total_profit': total_profit,
            'avg_profit_per_hour': avg_profit,
            'profit_std': profit_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'trade_ratio': trade_ratio,
            'total_volume': total_volume,
            'avg_volume_per_trade': avg_volume,
            'total_imbalance_cost': total_imbalance_cost,
            'avg_abs_imbalance': avg_imbalance,
            'imbalance_reduction': imbalance_reduction,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def _calculate_max_drawdown(self, cumulative_profits: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = cumulative_profits.expanding().max()
        drawdown = (cumulative_profits - peak) / (peak + 1e-6)
        return drawdown.min()
    
    def _calculate_imbalance_reduction(self, results: pd.DataFrame) -> float:
        """Calculate imbalance reduction compared to no trading"""
        # Baseline imbalance (no trading)
        baseline_imbalance = results['da_commitment'] - results['actual_generation']
        baseline_cost = (baseline_imbalance.abs() * 
                        results.get('imbalance_price', 50)).sum()  # Assume 50 EUR/MWh if not available
        
        actual_cost = results['imbalance_cost'].sum()
        
        if baseline_cost > 0:
            return (baseline_cost - actual_cost) / baseline_cost
        else:
            return 0.0
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table of all strategies"""
        return pd.DataFrame(self.results).T
    
    def statistical_comparison(self, 
                             strategy1: str, 
                             strategy2: str,
                             metric: str = 'total_profit') -> Dict[str, Any]:
        """Perform statistical comparison between two strategies"""
        
        if strategy1 not in self.simulation_data or strategy2 not in self.simulation_data:
            raise ValueError("Both strategies must be added first")
        
        data1 = self.simulation_data[strategy1]['simulation'][metric]
        data2 = self.simulation_data[strategy2]['simulation'][metric]
        
        # T-test
        t_stat, t_pvalue = stats.ttest_ind(data1, data2)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                             (len(data2) - 1) * data2.var()) / 
                            (len(data1) + len(data2) - 2))
        cohens_d = (data1.mean() - data2.mean()) / pooled_std
        
        return {
            'strategy1': strategy1,
            'strategy2': strategy2,
            'metric': metric,
            'mean_diff': data1.mean() - data2.mean(),
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'u_statistic': u_stat,
            'u_pvalue': u_pvalue,
            'cohens_d': cohens_d,
            'significant_at_5pct': min(t_pvalue, u_pvalue) < 0.05
        }
    
    def plot_cumulative_profits(self, 
                               strategies: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot cumulative profits over time"""
        
        if strategies is None:
            strategies = list(self.simulation_data.keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for strategy in strategies:
            data = self.simulation_data[strategy]['simulation']
            cumulative_profit = data['total_profit'].cumsum()
            ax.plot(cumulative_profit.index, cumulative_profit.values, 
                   label=strategy, linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Profit (EUR)')
        ax.set_title('Cumulative Profit Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_profit_distribution(self, 
                                strategies: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot profit distribution comparison"""
        
        if strategies is None:
            strategies = list(self.simulation_data.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Histogram
        for strategy in strategies:
            data = self.simulation_data[strategy]['simulation']
            axes[0].hist(data['total_profit'], alpha=0.7, label=strategy, bins=50)
        axes[0].set_xlabel('Hourly Profit (EUR)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Profit Distribution')
        axes[0].legend()
        
        # Box plot
        profit_data = [self.simulation_data[s]['simulation']['total_profit'] 
                      for s in strategies]
        axes[1].boxplot(profit_data, labels=strategies)
        axes[1].set_ylabel('Hourly Profit (EUR)')
        axes[1].set_title('Profit Box Plot')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Q-Q plot against normal distribution
        for i, strategy in enumerate(strategies[:2]):  # Limit to 2 for clarity
            data = self.simulation_data[strategy]['simulation']['total_profit']
            stats.probplot(data, dist="norm", plot=axes[2])
            axes[2].set_title('Q-Q Plot vs Normal Distribution')
        
        # Drawdown plot
        for strategy in strategies:
            data = self.simulation_data[strategy]['simulation']
            cumulative = data['total_profit'].cumsum()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / (peak + 1e-6) * 100
            axes[3].plot(drawdown.index, drawdown.values, label=strategy)
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Drawdown (%)')
        axes[3].set_title('Drawdown Over Time')
        axes[3].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_trading_activity(self, 
                             strategies: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot trading activity analysis"""
        
        if strategies is None:
            strategies = list(self.simulation_data.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Trading frequency by hour of day
        for strategy in strategies:
            data = self.simulation_data[strategy]['simulation']
            trades = data[data['action'] != 'hold']
            if len(trades) > 0:
                hourly_trades = trades.groupby(trades.index.hour).size()
                axes[0, 0].plot(hourly_trades.index, hourly_trades.values, 
                               marker='o', label=strategy)
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Number of Trades')
        axes[0, 0].set_title('Trading Frequency by Hour')
        axes[0, 0].legend()
        
        # Volume distribution
        for strategy in strategies:
            data = self.simulation_data[strategy]['simulation']
            volumes = data[data['volume'] > 0]['volume']
            if len(volumes) > 0:
                axes[0, 1].hist(volumes, alpha=0.7, label=strategy, bins=30)
        axes[0, 1].set_xlabel('Trade Volume (MW)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Trade Volume Distribution')
        axes[0, 1].legend()
        
        # Action type distribution
        action_data = {}
        for strategy in strategies:
            data = self.simulation_data[strategy]['simulation']
            action_counts = data['action'].value_counts()
            action_data[strategy] = action_counts
        
        action_df = pd.DataFrame(action_data).fillna(0)
        action_df.plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Action Type Distribution')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Profit vs Volume scatter
        for strategy in strategies:
            data = self.simulation_data[strategy]['simulation']
            trades = data[data['volume'] > 0]
            if len(trades) > 0:
                axes[1, 0].scatter(trades['volume'], trades['total_profit'], 
                                  alpha=0.6, label=strategy)
        axes[1, 0].set_xlabel('Trade Volume (MW)')
        axes[1, 0].set_ylabel('Profit (EUR)')
        axes[1, 0].set_title('Profit vs Trade Volume')
        axes[1, 0].legend()
        
        # Imbalance reduction over time
        for strategy in strategies:
            data = self.simulation_data[strategy]['simulation']
            baseline_imbalance = (data['da_commitment'] - data['actual_generation']).abs()
            actual_imbalance = data['final_imbalance'].abs()
            reduction = (baseline_imbalance - actual_imbalance).rolling(168).mean()  # Weekly average
            axes[1, 1].plot(reduction.index, reduction.values, label=strategy)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Imbalance Reduction (MW)')
        axes[1, 1].set_title('Rolling Imbalance Reduction (Weekly)')
        axes[1, 1].legend()
        
        # Performance metrics radar chart
        metrics_to_plot = ['sharpe_ratio', 'win_rate', 'profit_factor', 'trade_ratio']
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False)
        
        ax_radar = plt.subplot(2, 3, 6, projection='polar')
        for strategy in strategies:
            values = [self.results[strategy][metric] for metric in metrics_to_plot]
            # Normalize values for better visualization
            normalized_values = [(v - min(values)) / (max(values) - min(values) + 1e-6) 
                                for v in values]
            ax_radar.plot(angles, normalized_values, 'o-', linewidth=2, label=strategy)
            ax_radar.fill(angles, normalized_values, alpha=0.25)
        
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(metrics_to_plot)
        ax_radar.set_title('Performance Metrics Comparison')
        ax_radar.legend()
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_path: str = "trading_evaluation_report.html"):
        """Generate comprehensive HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PV Trading Strategy Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>PV Trading Strategy Evaluation Report</h1>
            
            <h2>Summary Statistics</h2>
            {self.get_summary_table().to_html(classes='summary-table')}
            
            <h2>Key Findings</h2>
            <ul>
        """
        
        # Add key findings
        summary = self.get_summary_table()
        best_profit = summary['total_profit'].idxmax()
        best_sharpe = summary['sharpe_ratio'].idxmax()
        
        html_content += f"""
                <li>Best overall profit: <strong>{best_profit}</strong> 
                    (€{summary.loc[best_profit, 'total_profit']:.2f})</li>
                <li>Best risk-adjusted return: <strong>{best_sharpe}</strong> 
                    (Sharpe ratio: {summary.loc[best_sharpe, 'sharpe_ratio']:.3f})</li>
                <li>Most active strategy: <strong>{summary['n_trades'].idxmax()}</strong> 
                    ({summary['n_trades'].max():.0f} trades)</li>
            </ul>
            
            <h2>Statistical Significance Tests</h2>
            <p>Detailed statistical comparisons between strategies...</p>
            
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    from data_processing import load_danish_data
    from baseline_strategies import create_all_baseline_strategies, evaluate_baseline_strategies
    from market_environment import MarketSimulator
    
    # Load data
    processor = load_danish_data()
    train_data, test_data = processor.get_training_data()
    
    test_market = processor.market_data.loc[test_data.index]
    test_pv = processor.pv_data.loc[test_data.index]
    
    # Create evaluator
    evaluator = TradingEvaluator()
    
    # Evaluate baseline strategies
    strategies = create_all_baseline_strategies(test_market, test_pv)
    simulator = MarketSimulator(test_market, test_pv)
    
    for name, strategy in strategies.items():
        strategy.fit(test_data, test_market, test_pv)
        
        def decision_func(feat):
            return strategy.predict(feat)
        
        results = simulator.simulate_strategy(decision_func, test_data)
        evaluator.add_strategy_results(name, results)
    
    # Generate plots
    fig1 = evaluator.plot_cumulative_profits()
    fig2 = evaluator.plot_profit_distribution()
    fig3 = evaluator.plot_trading_activity()
    
    plt.show()
    
    # Print summary
    print("Strategy Comparison:")
    print(evaluator.get_summary_table().round(4))
