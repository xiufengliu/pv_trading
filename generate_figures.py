#!/usr/bin/env python3
"""
Generate all PDF figures for the paper
"""

import sys
sys.path.append('src')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation import TradingEvaluator
from market_environment import MarketSimulator
from linear_policy import LinearTradingPolicy
from baseline_strategies import create_all_baseline_strategies
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend and style
import matplotlib
matplotlib.use('Agg')
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300
})

def load_data():
    """Load all experimental data"""
    features = pd.read_csv('data/danish_features_2022.csv', index_col=0, parse_dates=True)
    market_data = pd.read_csv('data/danish_market_2022.csv', index_col=0, parse_dates=True)
    pv_data = pd.read_csv('data/danish_pv_2022.csv', index_col=0, parse_dates=True)
    results = pd.read_csv('data/comprehensive_results.csv', index_col=0)
    
    # Split data
    train_size = int(0.7 * len(features))
    train_features = features.iloc[:train_size]
    test_features = features.iloc[train_size:]
    test_market = market_data.loc[test_features.index]
    test_pv = pv_data.loc[test_features.index]
    
    return features, market_data, pv_data, results, train_features, test_features, test_market, test_pv

def generate_cumulative_profits():
    """Generate cumulative profits figure"""
    features, market_data, pv_data, results, train_features, test_features, test_market, test_pv = load_data()
    
    # Setup evaluator and simulator
    evaluator = TradingEvaluator()
    simulator = MarketSimulator(test_market, test_pv)
    
    # Add strategies
    strategies_dict = create_all_baseline_strategies(test_market, test_pv)
    
    # Spot-only
    spot_strategy = strategies_dict['spot_only']
    spot_sim_results = simulator.simulate_strategy(lambda feat: spot_strategy.predict(feat), test_features)
    evaluator.add_strategy_results('Spot-Only', spot_sim_results)
    
    # Naive forecast
    naive_strategy = strategies_dict['naive_forecast']
    naive_strategy.fit(test_features, test_market, test_pv)
    naive_sim_results = simulator.simulate_strategy(lambda feat: naive_strategy.predict(feat), test_features)
    evaluator.add_strategy_results('Naive Forecast', naive_sim_results)
    
    # Linear policy
    train_market = market_data.loc[train_features.index]
    train_pv = pv_data.loc[train_features.index]
    linear_policy = LinearTradingPolicy(regularization='lasso', alpha=0.01, threshold=0.1)
    linear_policy.fit(train_features, train_market, train_pv, method='profit_optimization')
    linear_sim_results = simulator.simulate_strategy(lambda feat: linear_policy.predict(feat), test_features)
    evaluator.add_strategy_results('Linear Policy', linear_sim_results)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'orange', 'blue']
    linestyles = ['--', '-.', '-']
    linewidths = [2, 2, 3]
    
    for i, strategy_name in enumerate(['Spot-Only', 'Naive Forecast', 'Linear Policy']):
        data = evaluator.simulation_data[strategy_name]['simulation']
        cumulative_profit = data['total_profit'].cumsum()
        ax.plot(range(len(cumulative_profit)), cumulative_profit.values, 
               label=strategy_name, color=colors[i], linestyle=linestyles[i], 
               linewidth=linewidths[i], alpha=0.8)
    
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Cumulative Profit (EUR)')
    ax.set_title('Cumulative Profit Evolution Over Test Period')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('paper/fig_cumulative_profits.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Generated fig_cumulative_profits.pdf')

def generate_feature_importance():
    """Generate feature importance figure"""
    importance = pd.read_csv('data/feature_importance.csv', index_col=0)
    top_features = importance.head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if w < 0 else 'blue' for w in top_features['weight']]
    bars = ax.barh(range(len(top_features)), top_features['weight'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Weight')
    ax.set_title('Top 10 Feature Importance in Linear Policy')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for i, (bar, weight) in enumerate(zip(bars, top_features['weight'])):
        width = bar.get_width()
        ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
               f'{weight:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('paper/fig_feature_importance.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Generated fig_feature_importance.pdf')

def generate_sensitivity_analysis():
    """Generate sensitivity analysis figure"""
    sensitivity_df = pd.read_csv('data/sensitivity_analysis.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Profit vs threshold
    ax1.plot(sensitivity_df['threshold'], sensitivity_df['total_profit'], 'o-', 
             color='blue', linewidth=2, markersize=6, alpha=0.8)
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('Total Profit (EUR)')
    ax1.set_title('(a) Profit vs Decision Threshold')
    ax1.grid(True, alpha=0.3)
    
    # Trades vs threshold
    ax2.plot(sensitivity_df['threshold'], sensitivity_df['n_trades'], 'o-', 
             color='red', linewidth=2, markersize=6, alpha=0.8)
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Number of Trades')
    ax2.set_title('(b) Trading Activity vs Decision Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper/fig_sensitivity_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ Generated fig_sensitivity_analysis.pdf')

if __name__ == "__main__":
    print("=== Generating PDF Figures for Paper ===")
    
    try:
        generate_cumulative_profits()
        generate_feature_importance()
        generate_sensitivity_analysis()
        
        print("\n=== All PDF Figures Generated Successfully ===")
        
        # List generated files
        import os
        pdf_files = [f for f in os.listdir('paper/') if f.endswith('.pdf') and f.startswith('fig_')]
        print("Generated figures:")
        for pdf_file in sorted(pdf_files):
            print(f"  - {pdf_file}")
            
    except Exception as e:
        print(f"Error generating figures: {e}")
        import traceback
        traceback.print_exc()
