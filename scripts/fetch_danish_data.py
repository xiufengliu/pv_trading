"""
Danish Energy Data Fetcher

This script fetches real Danish energy market data from the Energinet API
and saves it for use in the PV trading research.

Usage:
    python scripts/fetch_danish_data.py --start 2022-01-01 --end 2022-12-31 --output data/danish_2022.csv
"""

import requests
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime, timedelta
import time
import os


class DanishDataFetcher:
    """Fetches Danish energy market data from Energinet API"""
    
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PV-Trading-Research/1.0'
        })
    
    def fetch_day_ahead_prices(self, start_date: str, end_date: str, price_area: str = "DK1") -> pd.DataFrame:
        """Fetch day-ahead electricity prices"""
        
        print(f"Fetching day-ahead prices for {price_area} from {start_date} to {end_date}...")
        
        params = {
            'start': start_date + 'T00:00',
            'end': end_date + 'T23:59',
            'filter': json.dumps({"PriceArea": [price_area]}),
            'columns': 'HourUTC,SpotPriceDKK,SpotPriceEUR',
            'timezone': 'UTC',
            'limit': 100000
        }
        
        response = self.session.get(f"{self.base_url}/Elspotprices", params=params)
        response.raise_for_status()
        
        data = pd.DataFrame(response.json()['records'])
        
        if data.empty:
            raise ValueError(f"No day-ahead price data found for {price_area} in the specified period")
        
        data['datetime'] = pd.to_datetime(data['HourUTC'])
        data = data.set_index('datetime').sort_index()
        
        # Convert prices to EUR/MWh
        data['da_price_eur_mwh'] = data['SpotPriceEUR']
        data['da_price_dkk_mwh'] = data['SpotPriceDKK']
        
        print(f"✓ Fetched {len(data)} hours of day-ahead prices")
        return data[['da_price_eur_mwh', 'da_price_dkk_mwh']]
    
    def fetch_imbalance_prices(self, start_date: str, end_date: str, price_area: str = "DK1") -> pd.DataFrame:
        """Fetch imbalance prices"""
        
        print(f"Fetching imbalance prices for {price_area}...")
        
        params = {
            'start': start_date + 'T00:00',
            'end': end_date + 'T23:59',
            'filter': json.dumps({"PriceArea": [price_area]}),
            'columns': 'HourUTC,ImbalancePriceEUR,ImbalancePriceDKK',
            'timezone': 'UTC',
            'limit': 100000
        }
        
        try:
            response = self.session.get(f"{self.base_url}/BalancingPowerPrices", params=params)
            response.raise_for_status()
            
            data = pd.DataFrame(response.json()['records'])
            
            if not data.empty:
                data['datetime'] = pd.to_datetime(data['HourUTC'])
                data = data.set_index('datetime').sort_index()
                
                data['imbalance_price_eur_mwh'] = data['ImbalancePriceEUR']
                data['imbalance_price_dkk_mwh'] = data['ImbalancePriceDKK']
                
                print(f"✓ Fetched {len(data)} hours of imbalance prices")
                return data[['imbalance_price_eur_mwh', 'imbalance_price_dkk_mwh']]
            else:
                print("⚠ No imbalance price data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"⚠ Could not fetch imbalance prices: {e}")
            return pd.DataFrame()
    
    def fetch_solar_production(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch actual solar production data"""
        
        print("Fetching solar production data...")
        
        params = {
            'start': start_date + 'T00:00',
            'end': end_date + 'T23:59',
            'columns': 'HourUTC,SolarPowerMWh',
            'timezone': 'UTC',
            'limit': 100000
        }
        
        try:
            response = self.session.get(f"{self.base_url}/PowerSystemRightNow", params=params)
            response.raise_for_status()
            
            data = pd.DataFrame(response.json()['records'])
            
            if not data.empty:
                data['datetime'] = pd.to_datetime(data['HourUTC'])
                data = data.set_index('datetime').sort_index()
                
                # Clean the data
                data['solar_production_mwh'] = pd.to_numeric(data['SolarPowerMWh'], errors='coerce')
                data = data.dropna()
                
                print(f"✓ Fetched {len(data)} hours of solar production data")
                return data[['solar_production_mwh']]
            else:
                print("⚠ No solar production data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"⚠ Could not fetch solar production data: {e}")
            return pd.DataFrame()
    
    def fetch_load_data(self, start_date: str, end_date: str, price_area: str = "DK1") -> pd.DataFrame:
        """Fetch electricity consumption data"""
        
        print(f"Fetching load data for {price_area}...")
        
        params = {
            'start': start_date + 'T00:00',
            'end': end_date + 'T23:59',
            'filter': json.dumps({"PriceArea": [price_area]}),
            'columns': 'HourUTC,TotalLoad',
            'timezone': 'UTC',
            'limit': 100000
        }
        
        try:
            response = self.session.get(f"{self.base_url}/ConsumptionDE35Hour", params=params)
            response.raise_for_status()
            
            data = pd.DataFrame(response.json()['records'])
            
            if not data.empty:
                data['datetime'] = pd.to_datetime(data['HourUTC'])
                data = data.set_index('datetime').sort_index()
                
                data['total_load_mwh'] = pd.to_numeric(data['TotalLoad'], errors='coerce')
                data = data.dropna()
                
                print(f"✓ Fetched {len(data)} hours of load data")
                return data[['total_load_mwh']]
            else:
                print("⚠ No load data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"⚠ Could not fetch load data: {e}")
            return pd.DataFrame()
    
    def fetch_all_data(self, start_date: str, end_date: str, price_area: str = "DK1") -> pd.DataFrame:
        """Fetch all available Danish energy market data"""
        
        print(f"\n=== Fetching Danish Energy Data ===")
        print(f"Period: {start_date} to {end_date}")
        print(f"Price Area: {price_area}")
        print()
        
        # Fetch all datasets
        da_prices = self.fetch_day_ahead_prices(start_date, end_date, price_area)
        time.sleep(1)  # Be nice to the API
        
        imb_prices = self.fetch_imbalance_prices(start_date, end_date, price_area)
        time.sleep(1)
        
        solar_data = self.fetch_solar_production(start_date, end_date)
        time.sleep(1)
        
        load_data = self.fetch_load_data(start_date, end_date, price_area)
        time.sleep(1)
        
        # Combine all data
        print("\nCombining datasets...")
        
        # Start with day-ahead prices as the base
        combined_data = da_prices.copy()
        
        # Add other datasets
        if not imb_prices.empty:
            combined_data = combined_data.join(imb_prices, how='outer')
        
        if not solar_data.empty:
            combined_data = combined_data.join(solar_data, how='outer')
        
        if not load_data.empty:
            combined_data = combined_data.join(load_data, how='outer')
        
        # Fill missing values
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        # Add metadata
        combined_data.attrs['source'] = 'Energinet Data Service'
        combined_data.attrs['price_area'] = price_area
        combined_data.attrs['fetch_date'] = datetime.now().isoformat()
        
        print(f"✓ Combined dataset: {len(combined_data)} hours")
        print(f"✓ Columns: {list(combined_data.columns)}")
        
        return combined_data
    
    def validate_data(self, data: pd.DataFrame) -> dict:
        """Validate the fetched data"""
        
        print("\n=== Data Validation ===")
        
        validation_results = {
            'total_hours': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'date_range': (data.index.min(), data.index.max()),
            'price_statistics': {},
            'data_quality_score': 0
        }
        
        # Price validation
        if 'da_price_eur_mwh' in data.columns:
            prices = data['da_price_eur_mwh']
            validation_results['price_statistics'] = {
                'mean': prices.mean(),
                'std': prices.std(),
                'min': prices.min(),
                'max': prices.max(),
                'negative_prices': (prices < 0).sum(),
                'extreme_prices': (prices > 1000).sum()
            }
        
        # Calculate quality score
        total_possible_values = len(data) * len(data.columns)
        missing_values = data.isnull().sum().sum()
        validation_results['data_quality_score'] = (total_possible_values - missing_values) / total_possible_values
        
        # Print validation summary
        print(f"Total hours: {validation_results['total_hours']}")
        print(f"Date range: {validation_results['date_range'][0]} to {validation_results['date_range'][1]}")
        print(f"Data quality score: {validation_results['data_quality_score']:.2%}")
        
        if validation_results['missing_values']:
            print("Missing values:")
            for col, missing in validation_results['missing_values'].items():
                if missing > 0:
                    print(f"  {col}: {missing} ({missing/len(data):.1%})")
        
        if validation_results['price_statistics']:
            stats = validation_results['price_statistics']
            print(f"Price statistics (EUR/MWh):")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Range: {stats['min']:.2f} to {stats['max']:.2f}")
            if stats['negative_prices'] > 0:
                print(f"  Negative prices: {stats['negative_prices']} hours")
            if stats['extreme_prices'] > 0:
                print(f"  Extreme prices (>1000): {stats['extreme_prices']} hours")
        
        return validation_results


def main():
    parser = argparse.ArgumentParser(description='Fetch Danish energy market data')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--area', default='DK1', help='Price area (DK1 or DK2)')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Fetch data
    fetcher = DanishDataFetcher()
    
    try:
        data = fetcher.fetch_all_data(args.start, args.end, args.area)
        
        # Validate data
        validation = fetcher.validate_data(data)
        
        # Save data
        data.to_csv(args.output)
        print(f"\n✓ Data saved to: {args.output}")
        
        # Save validation report
        validation_file = args.output.replace('.csv', '_validation.json')
        with open(validation_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            validation_copy = validation.copy()
            validation_copy['date_range'] = [str(d) for d in validation['date_range']]
            json.dump(validation_copy, f, indent=2)
        
        print(f"✓ Validation report saved to: {validation_file}")
        
        print(f"\n=== Data Fetch Complete ===")
        print(f"Successfully fetched {len(data)} hours of Danish energy data")
        print(f"Data quality: {validation['data_quality_score']:.1%}")
        
    except Exception as e:
        print(f"\n✗ Error fetching data: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
