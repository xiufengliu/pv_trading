# Real-World Data Sources for PV Intraday Trading Research

This guide provides detailed information on accessing real Danish and European energy market data for your IEEE TSG paper.

## 1. Danish Energy Data (Primary Recommendation)

### Energinet Data Service (Free API)
- **Website**: https://www.energidataservice.dk/
- **Documentation**: https://www.energidataservice.dk/tso-electricity/DatahubPriceList
- **API Base URL**: `https://api.energidataservice.dk/dataset/`

#### Key Datasets:

**Day-Ahead Prices (Elspot)**
```
Dataset: Elspotprices
Endpoint: https://api.energidataservice.dk/dataset/Elspotprices
Columns: HourUTC, PriceArea, SpotPriceDKK, SpotPriceEUR
Coverage: 2013-present, hourly
```

**Intraday Prices**
```
Dataset: ElspotpricesIntraday  
Endpoint: https://api.energidataservice.dk/dataset/ElspotpricesIntraday
Columns: HourUTC, PriceArea, SpotPriceDKK, SpotPriceEUR
Coverage: Recent years, hourly
```

**Imbalance Prices**
```
Dataset: BalancingPowerPrices
Endpoint: https://api.energidataservice.dk/dataset/BalancingPowerPrices
Columns: HourUTC, PriceArea, ImbalancePriceEUR, ImbalancePriceDKK
Coverage: 2019-present, hourly
```

**Solar Production**
```
Dataset: PowerSystemRightNow
Endpoint: https://api.energidataservice.dk/dataset/PowerSystemRightNow
Columns: HourUTC, SolarPowerMWh, WindPowerMWh
Coverage: 2017-present, hourly
```

#### Example API Calls:

```python
import requests
import pandas as pd

# Day-ahead prices for DK1 in 2022
url = "https://api.energidataservice.dk/dataset/Elspotprices"
params = {
    'start': '2022-01-01T00:00',
    'end': '2022-12-31T23:00',
    'filter': '{"PriceArea":["DK1"]}',
    'columns': 'HourUTC,SpotPriceEUR',
    'timezone': 'UTC'
}

response = requests.get(url, params=params)
data = pd.DataFrame(response.json()['records'])
```

## 2. European Power Exchange Data

### Nord Pool
- **Website**: https://www.nordpoolgroup.com/Market-data1/
- **Historical Data**: Available for download
- **API**: Available for registered users
- **Coverage**: All Nordic countries (DK1, DK2, NO1-NO5, SE1-SE4, FI)

### EPEX SPOT
- **Website**: https://www.epexspot.com/en/market-data
- **Data**: German, French, Austrian markets
- **API**: Available for registered users
- **Intraday**: Continuous trading data available

## 3. Weather and Solar Data

### ECMWF ERA5 Reanalysis
- **Website**: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
- **API**: Copernicus Climate Data Store (CDS) API
- **Coverage**: Global, 1979-present, hourly
- **Variables**: Solar radiation, temperature, wind speed, cloud cover

#### Installation and Setup:
```bash
pip install cdsapi
```

```python
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'surface_solar_radiation_downwards',
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_cloud_cover'
        ],
        'year': '2022',
        'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'day': [f'{i:02d}' for i in range(1, 32)],
        'time': [f'{i:02d}:00' for i in range(24)],
        'area': [58, 8, 54, 15],  # Denmark bounding box [North, West, South, East]
        'format': 'netcdf',
    },
    'denmark_weather_2022.nc')
```

### PVLIB and NREL
- **NSRDB**: National Solar Radiation Database
- **PVLIB**: Python library for PV modeling
- **Coverage**: Global solar irradiance data

## 4. ENTSO-E Transparency Platform

### All European TSO Data
- **Website**: https://transparency.entsoe.eu/
- **API**: Free REST API
- **Documentation**: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

#### Key Data:
- Day-ahead prices
- Actual generation by fuel type
- Load forecasts and actual load
- Cross-border flows
- Imbalance prices (where available)

#### Example Usage:
```python
from entsoe import EntsoePandasClient

client = EntsoePandasClient(api_key='your-api-key')

# Day-ahead prices for Denmark
start = pd.Timestamp('20220101', tz='Europe/Copenhagen')
end = pd.Timestamp('20221231', tz='Europe/Copenhagen')
country_code = 'DK_1'  # DK1 bidding zone

prices = client.query_day_ahead_prices(country_code, start=start, end=end)
```

## 5. Recommended Data Collection Strategy

### For Your IEEE TSG Paper:

1. **Primary Dataset**: Use Energinet API for Danish data (2022-2023)
   - Day-ahead prices (DK1)
   - Imbalance prices
   - Actual solar production
   
2. **Weather Data**: ERA5 reanalysis for Denmark
   - Solar irradiance
   - Temperature
   - Wind speed
   - Cloud cover

3. **Validation Dataset**: Use ENTSO-E data for comparison
   - Cross-validate prices
   - Additional European markets for robustness testing

### Data Quality Considerations:

1. **Missing Data**: Handle gaps in intraday price data
2. **Time Zones**: Ensure consistent UTC timestamps
3. **Daylight Saving**: Account for DST transitions
4. **Market Holidays**: Handle non-trading days
5. **Data Validation**: Cross-check with multiple sources

## 6. Implementation Example

See `src/data_processing.py` for a complete implementation that:
- Fetches real Danish market data from Energinet API
- Handles API errors gracefully with fallback to synthetic data
- Processes and aligns different data sources
- Creates realistic PV generation profiles
- Generates comprehensive feature sets for ML models

## 7. Data Usage Guidelines

### For Academic Research:
- Most APIs are free for research use
- Cite data sources appropriately
- Check terms of service for publication rights
- Consider data sharing policies for reproducibility

### Recommended Citation Format:
```
Energinet. (2023). Danish Energy Data Service. Retrieved from https://www.energidataservice.dk/
ECMWF. (2023). ERA5 hourly data on single levels from 1979 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
```

This approach will provide your IEEE TSG paper with credible, real-world data that demonstrates the practical applicability of your PV trading methodology.
