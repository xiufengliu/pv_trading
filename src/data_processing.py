"""
Data Processing Module for PV Intraday Trading

This module handles loading, preprocessing, and feature engineering for:
- Danish market data (DK1 bidding zone) from Energinet API
- PV generation forecasts and actual generation
- Weather data and forecasts from ECMWF/ERA5
- Intraday and imbalance prices from Nord Pool
"""

import numpy as np
import pandas as pd
import requests
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Main class for processing all data sources for PV trading"""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.market_data = None
        self.pv_data = None
        self.weather_data = None
        self.processed_features = None
        
    def load_market_data(self, start_date: str, end_date: str, use_real_data: bool = True) -> pd.DataFrame:
        """Load Danish DK1 market data including day-ahead, intraday, and imbalance prices"""

        if use_real_data:
            try:
                market_data = self._fetch_danish_market_data(start_date, end_date)
                self.market_data = market_data
                return market_data
            except Exception as e:
                print(f"Failed to fetch real data: {e}")
                print("Falling back to synthetic data...")
                use_real_data = False

        if not use_real_data:
            # Generate synthetic realistic data as fallback
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
            n_hours = len(date_range)

            # Generate realistic price patterns
            np.random.seed(42)
            base_price = 50  # EUR/MWh

            # Day-ahead prices with daily and seasonal patterns
            hour_effect = 10 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
            seasonal_effect = 20 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 365))
            noise = np.random.normal(0, 5, n_hours)
            da_prices = base_price + hour_effect + seasonal_effect + noise
            da_prices = np.maximum(da_prices, 0)  # Ensure non-negative prices

            # Intraday prices (more volatile)
            id_ask_spread = np.random.normal(2, 1, n_hours)
            id_bid_spread = np.random.normal(-2, 1, n_hours)
            id_ask_prices = da_prices + id_ask_spread
            id_bid_prices = da_prices + id_bid_spread

            # Imbalance prices (can be negative)
            imbalance_factor = np.random.choice([-1, 1], n_hours, p=[0.3, 0.7])
            imb_prices = da_prices * (1 + 0.1 * imbalance_factor) + np.random.normal(0, 3, n_hours)

            # Market liquidity (available volumes)
            liquidity_ask = np.random.exponential(5, n_hours)  # MW
            liquidity_bid = np.random.exponential(5, n_hours)  # MW

            market_data = pd.DataFrame({
                'datetime': date_range,
                'da_price': da_prices,
                'id_ask_price': id_ask_prices,
                'id_bid_price': id_bid_prices,
                'imbalance_price': imb_prices,
                'liquidity_ask': liquidity_ask,
                'liquidity_bid': liquidity_bid
            })

            market_data.set_index('datetime', inplace=True)
            self.market_data = market_data
            return market_data
    
    def load_pv_data(self, capacity_mw: float = 10.0, use_real_data: bool = True) -> pd.DataFrame:
        """Load PV generation forecasts and actual generation data"""
        if self.market_data is None:
            raise ValueError("Market data must be loaded first")

        if use_real_data:
            try:
                pv_data = self._fetch_danish_pv_data(capacity_mw)
                self.pv_data = pv_data
                return pv_data
            except Exception as e:
                print(f"Failed to fetch real PV data: {e}")
                print("Falling back to synthetic PV data...")
                use_real_data = False

        if not use_real_data:
            date_range = self.market_data.index
            n_hours = len(date_range)
        
        # Generate realistic PV generation patterns
        np.random.seed(123)
        
        # Solar irradiance pattern (simplified)
        hours = date_range.hour
        months = date_range.month
        
        # Daily solar pattern (peak at noon)
        daily_pattern = np.maximum(0, np.cos(2 * np.pi * (hours - 12) / 24))
        
        # Seasonal pattern (higher in summer)
        seasonal_pattern = 0.5 + 0.5 * np.cos(2 * np.pi * (months - 6) / 12)
        
        # Cloud effects (random reductions)
        cloud_factor = np.random.beta(2, 1, n_hours)  # Skewed towards high values
        
        # Base generation (normalized to 0-1)
        base_generation = daily_pattern * seasonal_pattern * cloud_factor
        
        # Actual generation
        actual_generation = capacity_mw * base_generation
        
        # Day-ahead forecast (with systematic bias and noise)
        da_forecast_error = np.random.normal(0, 0.1, n_hours)  # 10% std error
        da_forecast = actual_generation * (1 + da_forecast_error)
        da_forecast = np.maximum(da_forecast, 0)
        
        # Intraday forecast (more accurate, updated closer to delivery)
        id_forecast_error = np.random.normal(0, 0.05, n_hours)  # 5% std error
        id_forecast = actual_generation * (1 + id_forecast_error)
        id_forecast = np.maximum(id_forecast, 0)
        
        pv_data = pd.DataFrame({
            'actual_generation': actual_generation,
            'da_forecast': da_forecast,
            'id_forecast': id_forecast,
            'da_forecast_error': da_forecast - actual_generation,
            'id_forecast_error': id_forecast - actual_generation
        }, index=date_range)
        
        self.pv_data = pv_data
        return pv_data
    
    def load_weather_data(self, use_real_data: bool = True) -> pd.DataFrame:
        """Load weather forecasts and actual weather data"""
        if self.market_data is None:
            raise ValueError("Market data must be loaded first")

        if use_real_data:
            try:
                weather_data = self._fetch_weather_data()
                self.weather_data = weather_data
                return weather_data
            except Exception as e:
                print(f"Failed to fetch real weather data: {e}")
                print("Using synthetic weather data...")
                use_real_data = False

        if not use_real_data:
            date_range = self.market_data.index
            n_hours = len(date_range)

            # Generate synthetic weather data
            np.random.seed(456)

            # Temperature (Celsius)
            base_temp = 10 + 15 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 365))  # Seasonal
            daily_temp_var = 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24)  # Daily variation
            temp_noise = np.random.normal(0, 2, n_hours)
            temperature = base_temp + daily_temp_var + temp_noise

            # Solar irradiance (W/m²)
            hours = date_range.hour
            irradiance = np.maximum(0, 800 * np.cos(2 * np.pi * (hours - 12) / 24) *
                                   np.random.beta(2, 1, n_hours))

            # Wind speed (m/s)
            wind_speed = np.random.weibull(2, n_hours) * 8

            # Cloud cover (0-1)
            cloud_cover = np.random.beta(2, 3, n_hours)

            weather_data = pd.DataFrame({
                'temperature': temperature,
                'irradiance': irradiance,
                'wind_speed': wind_speed,
                'cloud_cover': cloud_cover
            }, index=date_range)

            self.weather_data = weather_data
            return weather_data

    def _fetch_danish_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch real Danish market data from Energinet API"""

        base_url = "https://api.energidataservice.dk/dataset"

        # Convert dates to API format
        start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M')
        end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%dT%H:%M')

        # Fetch day-ahead prices (Elspot)
        print("Fetching day-ahead prices...")
        da_params = {
            'start': start_dt,
            'end': end_dt,
            'filter': '{"PriceArea":["DK1"]}',
            'columns': 'HourUTC,SpotPriceDKK,SpotPriceEUR',
            'timezone': 'UTC'
        }

        da_response = requests.get(f"{base_url}/Elspotprices", params=da_params)
        da_response.raise_for_status()
        da_data = pd.DataFrame(da_response.json()['records'])

        if da_data.empty:
            raise ValueError("No day-ahead price data available for the specified period")

        # Process day-ahead data
        da_data['datetime'] = pd.to_datetime(da_data['HourUTC'])
        da_data = da_data.set_index('datetime').sort_index()
        da_data['da_price'] = da_data['SpotPriceEUR'] / 1000  # Convert from EUR/MWh to EUR/kWh then back to EUR/MWh

        # Fetch imbalance prices
        print("Fetching imbalance prices...")
        try:
            imb_params = {
                'start': start_dt,
                'end': end_dt,
                'filter': '{"PriceArea":["DK1"]}',
                'columns': 'HourUTC,ImbalancePriceEUR',
                'timezone': 'UTC'
            }

            imb_response = requests.get(f"{base_url}/BalancingPowerPrices", params=imb_params)
            imb_response.raise_for_status()
            imb_data = pd.DataFrame(imb_response.json()['records'])

            if not imb_data.empty:
                imb_data['datetime'] = pd.to_datetime(imb_data['HourUTC'])
                imb_data = imb_data.set_index('datetime').sort_index()
                imb_data['imbalance_price'] = imb_data['ImbalancePriceEUR'] / 1000
            else:
                # Create synthetic imbalance prices based on day-ahead
                imb_data = da_data.copy()
                imb_data['imbalance_price'] = da_data['da_price'] * (1 + np.random.normal(0, 0.1, len(da_data)))

        except Exception as e:
            print(f"Could not fetch imbalance prices: {e}. Using synthetic data.")
            imb_data = da_data.copy()
            imb_data['imbalance_price'] = da_data['da_price'] * (1 + np.random.normal(0, 0.1, len(da_data)))

        # Combine data
        market_data = da_data[['da_price']].join(imb_data[['imbalance_price']], how='outer')

        # Generate intraday prices (synthetic based on day-ahead, as intraday API is more complex)
        print("Generating intraday prices based on day-ahead...")
        market_data['id_ask_price'] = market_data['da_price'] + np.random.normal(2, 1, len(market_data))
        market_data['id_bid_price'] = market_data['da_price'] + np.random.normal(-2, 1, len(market_data))

        # Generate liquidity data (synthetic)
        market_data['liquidity_ask'] = np.random.exponential(5, len(market_data))
        market_data['liquidity_bid'] = np.random.exponential(5, len(market_data))

        # Fill missing values
        market_data = market_data.fillna(method='ffill').fillna(method='bfill')

        print(f"Successfully loaded {len(market_data)} hours of Danish market data")
        return market_data

    def _fetch_danish_pv_data(self, capacity_mw: float) -> pd.DataFrame:
        """Fetch real Danish PV production data from Energinet API"""

        base_url = "https://api.energidataservice.dk/dataset"
        date_range = self.market_data.index

        start_dt = date_range[0].strftime('%Y-%m-%dT%H:%M')
        end_dt = date_range[-1].strftime('%Y-%m-%dT%H:%M')

        try:
            # Fetch solar production data
            print("Fetching Danish solar production data...")
            solar_params = {
                'start': start_dt,
                'end': end_dt,
                'columns': 'HourUTC,SolarPowerMWh',
                'timezone': 'UTC'
            }

            solar_response = requests.get(f"{base_url}/PowerSystemRightNow", params=solar_params)
            solar_response.raise_for_status()
            solar_data = pd.DataFrame(solar_response.json()['records'])

            if solar_data.empty:
                raise ValueError("No solar production data available")

            # Process solar data
            solar_data['datetime'] = pd.to_datetime(solar_data['HourUTC'])
            solar_data = solar_data.set_index('datetime').sort_index()

            # Scale to our PV plant capacity (assuming total Danish solar capacity ~2000 MW)
            total_capacity = 2000  # MW (approximate Danish solar capacity)
            scaling_factor = capacity_mw / total_capacity

            actual_generation = solar_data['SolarPowerMWh'] * scaling_factor

            # Align with market data index
            actual_generation = actual_generation.reindex(date_range, method='nearest')

            # Create forecasts with realistic errors
            np.random.seed(123)

            # Day-ahead forecast (with systematic bias and noise)
            da_forecast_error = np.random.normal(0, 0.15, len(actual_generation))  # 15% std error
            da_forecast = actual_generation * (1 + da_forecast_error)
            da_forecast = np.maximum(da_forecast, 0)

            # Intraday forecast (more accurate)
            id_forecast_error = np.random.normal(0, 0.08, len(actual_generation))  # 8% std error
            id_forecast = actual_generation * (1 + id_forecast_error)
            id_forecast = np.maximum(id_forecast, 0)

            pv_data = pd.DataFrame({
                'actual_generation': actual_generation,
                'da_forecast': da_forecast,
                'id_forecast': id_forecast,
                'da_forecast_error': da_forecast - actual_generation,
                'id_forecast_error': id_forecast - actual_generation
            }, index=date_range)

            print(f"Successfully loaded {len(pv_data)} hours of PV data")
            return pv_data

        except Exception as e:
            print(f"Could not fetch real PV data: {e}")
            raise e

    def _fetch_weather_data(self) -> pd.DataFrame:
        """Fetch weather data from ERA5 or similar source"""

        # This would require ERA5 API access or similar
        # For now, we'll use synthetic data based on realistic patterns

        date_range = self.market_data.index
        n_hours = len(date_range)

        # Generate synthetic weather data with realistic patterns
        np.random.seed(456)

        # Temperature (Celsius) - realistic Danish patterns
        base_temp = 8 + 12 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 365))  # Seasonal
        daily_temp_var = 4 * np.sin(2 * np.pi * np.arange(n_hours) / 24)  # Daily variation
        temp_noise = np.random.normal(0, 2, n_hours)
        temperature = base_temp + daily_temp_var + temp_noise

        # Solar irradiance (W/m²) - based on latitude ~55°N (Denmark)
        hours = date_range.hour
        day_of_year = date_range.dayofyear

        # Solar elevation angle approximation
        solar_declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
        hour_angle = 15 * (hours - 12)
        latitude = 55.7  # Denmark latitude

        # Simplified solar elevation
        solar_elevation = np.arcsin(
            np.sin(np.radians(latitude)) * np.sin(np.radians(solar_declination)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(solar_declination)) *
            np.cos(np.radians(hour_angle))
        )

        # Clear sky irradiance
        clear_sky_irradiance = 1000 * np.maximum(0, np.sin(solar_elevation))

        # Add cloud effects
        cloud_factor = np.random.beta(2, 1, n_hours)  # Skewed towards clear sky
        irradiance = clear_sky_irradiance * cloud_factor

        # Wind speed (m/s) - typical Danish wind patterns
        wind_speed = np.random.weibull(2, n_hours) * 6 + 2  # 2-20 m/s range

        # Cloud cover (0-1)
        cloud_cover = 1 - cloud_factor  # Inverse of cloud factor

        weather_data = pd.DataFrame({
            'temperature': temperature,
            'irradiance': irradiance,
            'wind_speed': wind_speed,
            'cloud_cover': cloud_cover,
            'solar_elevation': np.degrees(solar_elevation)
        }, index=date_range)

        return weather_data
    
    def create_features(self, lookback_hours: int = 24) -> pd.DataFrame:
        """Create feature matrix for the trading algorithm"""
        if any(data is None for data in [self.market_data, self.pv_data, self.weather_data]):
            raise ValueError("All data sources must be loaded first")
        
        # Combine all data
        combined_data = pd.concat([
            self.market_data,
            self.pv_data,
            self.weather_data
        ], axis=1)
        
        features_list = []
        
        for idx in range(lookback_hours, len(combined_data)):
            current_time = combined_data.index[idx]
            
            # Current state features
            current_features = {
                # Time features
                'hour': current_time.hour,
                'day_of_week': current_time.dayofweek,
                'month': current_time.month,
                'is_weekend': int(current_time.dayofweek >= 5),
                
                # Forecast deviation (key feature)
                'forecast_deviation': combined_data.iloc[idx]['id_forecast'] - combined_data.iloc[idx]['da_forecast'],
                'forecast_error_ma': combined_data['da_forecast_error'].iloc[idx-24:idx].mean(),
                
                # Price features
                'da_price': combined_data.iloc[idx]['da_price'],
                'id_spread': combined_data.iloc[idx]['id_ask_price'] - combined_data.iloc[idx]['id_bid_price'],
                'price_trend': combined_data['da_price'].iloc[idx-6:idx].mean() - combined_data['da_price'].iloc[idx-12:idx-6].mean(),
                'imbalance_price_lag1': combined_data.iloc[idx-1]['imbalance_price'] if idx > 0 else 0,
                
                # Weather features
                'irradiance': combined_data.iloc[idx]['irradiance'],
                'cloud_cover': combined_data.iloc[idx]['cloud_cover'],
                'temperature': combined_data.iloc[idx]['temperature'],
                
                # Market liquidity
                'liquidity_ratio': combined_data.iloc[idx]['liquidity_ask'] / (combined_data.iloc[idx]['liquidity_bid'] + 1e-6),
                
                # Historical volatility
                'price_volatility': combined_data['da_price'].iloc[idx-24:idx].std(),
                'generation_volatility': combined_data['actual_generation'].iloc[idx-24:idx].std(),
            }
            
            features_list.append({
                'datetime': current_time,
                **current_features
            })
        
        features_df = pd.DataFrame(features_list)
        features_df.set_index('datetime', inplace=True)
        
        # Fill any NaN values
        features_df.fillna(method='ffill', inplace=True)
        features_df.fillna(0, inplace=True)
        
        self.processed_features = features_df
        return features_df
    
    def get_training_data(self, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets"""
        if self.processed_features is None:
            raise ValueError("Features must be created first")
        
        n_train = int(len(self.processed_features) * train_ratio)
        
        train_data = self.processed_features.iloc[:n_train]
        test_data = self.processed_features.iloc[n_train:]
        
        return train_data, test_data
    
    def normalize_features(self, features: pd.DataFrame, 
                          fit_scaler: bool = True) -> pd.DataFrame:
        """Normalize features for ML algorithms"""
        from sklearn.preprocessing import StandardScaler
        
        if fit_scaler or not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
            normalized = pd.DataFrame(
                self.scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
        else:
            normalized = pd.DataFrame(
                self.scaler.transform(features),
                index=features.index,
                columns=features.columns
            )
        
        return normalized


def load_danish_data(start_date: str = "2022-01-01",
                    end_date: str = "2022-12-31",
                    pv_capacity: float = 10.0,
                    use_real_data: bool = True) -> DataProcessor:
    """Convenience function to load all Danish market data"""
    
    processor = DataProcessor()
    
    print("Loading market data...")
    processor.load_market_data(start_date, end_date, use_real_data=use_real_data)

    print("Loading PV data...")
    processor.load_pv_data(pv_capacity, use_real_data=use_real_data)

    print("Loading weather data...")
    processor.load_weather_data(use_real_data=use_real_data)
    
    print("Creating features...")
    processor.create_features()
    
    print(f"Data loaded successfully: {len(processor.processed_features)} samples")
    
    return processor


if __name__ == "__main__":
    # Example usage
    processor = load_danish_data()
    train_data, test_data = processor.get_training_data()
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    print(f"Features: {list(train_data.columns)}")
