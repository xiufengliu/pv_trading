"""
Real Data Example for PV Intraday Trading

This script demonstrates how to:
1. Load real Danish market data from Energinet API
2. Process and validate the data
3. Run trading strategies with real data
4. Generate results suitable for IEEE TSG paper

Requirements:
- Internet connection for API access
- requests library for API calls
- All other dependencies from requirements.txt
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import our modules
from data_processing import load_danish_data, DataProcessor
from linear_policy import LinearTradingPolicy
from baseline_strategies import create_all_baseline_strategies, evaluate_baseline_strategies
from market_environment import MarketSimulator
from evaluation import TradingEvaluator


def main():
    """Main function to demonstrate real data usage"""
    
    print("=== PV Intraday Trading with Real Danish Data ===\n")
    
    # Configuration
    START_DATE = "2022-01-01"
    END_DATE = "2022-12-31"
    PV_CAPACITY = 10.0  # MW
    USE_REAL_DATA = True  # Set to False to use synthetic data
    
    print(f"Configuration:")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  PV Capacity: {PV_CAPACITY} MW")
    print(f"  Use Real Data: {USE_REAL_DATA}")
    print()
    
    # Step 1: Load Data
    print("Step 1: Loading Data...")
    try:
        processor = load_danish_data(
            start_date=START_DATE,
            end_date=END_DATE,
            pv_capacity=PV_CAPACITY,
            use_real_data=USE_REAL_DATA
        )
        
        print(f"✓ Successfully loaded data:")
        print(f"  Market data: {len(processor.market_data)} hours")
        print(f"  PV data: {len(processor.pv_data)} hours")
        print(f"  Weather data: {len(processor.weather_data)} hours")
        print(f"  Features: {len(processor.processed_features)} samples")
        print()
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("This might be due to API access issues. Check your internet connection.")
        print("The script will continue with synthetic data for demonstration.")
        
        processor = load_danish_data(
            start_date=START_DATE,
            end_date=END_DATE,
            pv_capacity=PV_CAPACITY,
            use_real_data=False
        )
        print("✓ Loaded synthetic data for demonstration")
        print()
    
    # Step 2: Data Analysis
    print("Step 2: Data Analysis...")
    
    # Basic statistics
    market_stats = processor.market_data.describe()
    pv_stats = processor.pv_data.describe()
    
    print("Market Data Statistics:")
    print(f"  Day-ahead price: {market_stats.loc['mean', 'da_price']:.2f} ± {market_stats.loc['std', 'da_price']:.2f} EUR/MWh")
    print(f"  Imbalance price: {market_stats.loc['mean', 'imbalance_price']:.2f} ± {market_stats.loc['std', 'imbalance_price']:.2f} EUR/MWh")
    print(f"  Intraday spread: {(market_stats.loc['mean', 'id_ask_price'] - market_stats.loc['mean', 'id_bid_price']):.2f} EUR/MWh")
    
    print("\nPV Generation Statistics:")
    print(f"  Actual generation: {pv_stats.loc['mean', 'actual_generation']:.2f} ± {pv_stats.loc['std', 'actual_generation']:.2f} MW")
    print(f"  DA forecast error: {pv_stats.loc['mean', 'da_forecast_error']:.3f} ± {pv_stats.loc['std', 'da_forecast_error']:.3f} MW")
    print(f"  ID forecast error: {pv_stats.loc['mean', 'id_forecast_error']:.3f} ± {pv_stats.loc['std', 'id_forecast_error']:.3f} MW")
    print()
    
    # Step 3: Split Data
    print("Step 3: Splitting Data...")
    train_data, test_data = processor.get_training_data(train_ratio=0.7)
    
    train_market = processor.market_data.loc[train_data.index]
    train_pv = processor.pv_data.loc[train_data.index]
    test_market = processor.market_data.loc[test_data.index]
    test_pv = processor.pv_data.loc[test_data.index]
    
    print(f"  Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"  Testing period: {test_data.index[0]} to {test_data.index[-1]}")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Testing samples: {len(test_data)}")
    print()
    
    # Step 4: Train Linear Policy
    print("Step 4: Training Linear Feature-Driven Policy...")
    
    linear_policy = LinearTradingPolicy(regularization='ridge', alpha=0.1)
    linear_policy.fit(train_data, train_market, train_pv, method='profit_optimization')
    
    # Evaluate on test data
    linear_results = linear_policy.evaluate(test_data, test_market, test_pv)
    
    print("Linear Policy Results:")
    for key, value in linear_results.items():
        print(f"  {key}: {value:.4f}")
    print()
    
    # Feature importance
    importance = linear_policy.get_feature_importance()
    print("Top 5 Most Important Features:")
    for i, row in importance.head().iterrows():
        print(f"  {row['feature']}: {row['weight']:.4f}")
    print()
    
    # Step 5: Evaluate Baseline Strategies
    print("Step 5: Evaluating Baseline Strategies...")
    
    baseline_strategies = create_all_baseline_strategies(test_market, test_pv)
    baseline_results = evaluate_baseline_strategies(
        baseline_strategies, test_data, test_market, test_pv
    )
    
    print("Baseline Strategy Results:")
    print(baseline_results.round(4))
    print()
    
    # Step 6: Comprehensive Evaluation
    print("Step 6: Comprehensive Evaluation...")
    
    evaluator = TradingEvaluator()
    simulator = MarketSimulator(test_market, test_pv)
    
    # Add linear policy results
    def linear_decision_func(feat):
        return linear_policy.predict(feat)
    
    linear_sim_results = simulator.simulate_strategy(linear_decision_func, test_data)
    evaluator.add_strategy_results('Linear Policy', linear_sim_results)
    
    # Add baseline results
    for name, strategy in baseline_strategies.items():
        if name in ['spot_only', 'deterministic', 'naive_forecast']:  # Focus on main baselines
            strategy.fit(test_data, test_market, test_pv)
            
            def decision_func(feat):
                return strategy.predict(feat)
            
            sim_results = simulator.simulate_strategy(decision_func, test_data)
            evaluator.add_strategy_results(name.replace('_', ' ').title(), sim_results)
    
    # Generate summary table
    summary = evaluator.get_summary_table()
    print("Strategy Comparison Summary:")
    print(summary[['total_profit', 'sharpe_ratio', 'n_trades', 'imbalance_reduction']].round(4))
    print()
    
    # Step 7: Statistical Analysis
    print("Step 7: Statistical Analysis...")
    
    # Compare linear policy vs spot-only
    stat_comparison = evaluator.statistical_comparison('Linear Policy', 'Spot Only', 'total_profit')
    
    print("Statistical Comparison (Linear Policy vs Spot-Only):")
    print(f"  Mean profit difference: {stat_comparison['mean_diff']:.4f} EUR/hour")
    print(f"  T-test p-value: {stat_comparison['t_pvalue']:.6f}")
    print(f"  Mann-Whitney p-value: {stat_comparison['u_pvalue']:.6f}")
    print(f"  Cohen's d (effect size): {stat_comparison['cohens_d']:.4f}")
    print(f"  Statistically significant: {stat_comparison['significant_at_5pct']}")
    print()
    
    # Step 8: Generate Visualizations
    print("Step 8: Generating Visualizations...")
    
    try:
        # Cumulative profit plot
        fig1 = evaluator.plot_cumulative_profits(figsize=(12, 6))
        fig1.savefig('results/cumulative_profits.png', dpi=300, bbox_inches='tight')
        
        # Profit distribution plot
        fig2 = evaluator.plot_profit_distribution(figsize=(15, 10))
        fig2.savefig('results/profit_distribution.png', dpi=300, bbox_inches='tight')
        
        # Trading activity plot
        fig3 = evaluator.plot_trading_activity(figsize=(18, 12))
        fig3.savefig('results/trading_activity.png', dpi=300, bbox_inches='tight')
        
        print("✓ Visualizations saved to results/ directory")
        
    except Exception as e:
        print(f"✗ Error generating visualizations: {e}")
        print("Make sure the results/ directory exists")
    
    print()
    
    # Step 9: Generate Report
    print("Step 9: Generating Report...")
    
    try:
        evaluator.generate_report("results/trading_evaluation_report.html")
        print("✓ HTML report generated: results/trading_evaluation_report.html")
    except Exception as e:
        print(f"✗ Error generating report: {e}")
    
    print()
    
    # Step 10: Key Findings for Paper
    print("Step 10: Key Findings for IEEE TSG Paper...")
    
    best_strategy = summary['total_profit'].idxmax()
    profit_improvement = (summary.loc[best_strategy, 'total_profit'] / 
                         summary.loc['Spot Only', 'total_profit'] - 1) * 100
    
    print("Key Results for Paper:")
    print(f"  • Best performing strategy: {best_strategy}")
    print(f"  • Profit improvement over spot-only: {profit_improvement:.1f}%")
    print(f"  • Imbalance cost reduction: {summary.loc[best_strategy, 'imbalance_reduction']*100:.1f}%")
    print(f"  • Number of trades: {summary.loc[best_strategy, 'n_trades']:.0f}")
    print(f"  • Sharpe ratio: {summary.loc[best_strategy, 'sharpe_ratio']:.3f}")
    print(f"  • Statistical significance: p < 0.001" if stat_comparison['t_pvalue'] < 0.001 else f"  • Statistical significance: p = {stat_comparison['t_pvalue']:.3f}")
    
    print("\n=== Analysis Complete ===")
    print("Results are ready for inclusion in your IEEE TSG paper!")


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run the main analysis
    main()
