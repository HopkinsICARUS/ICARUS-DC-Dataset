import pandas as pd
from functools import lru_cache
from typing import Dict, Tuple
import warnings

# Global cache for year data
_YEAR_DATA_CACHE: Dict[int, pd.DataFrame] = {}

def _load_year_data(year: int) -> pd.DataFrame:
    """Load and cache year data with parsed datetime."""
    if year not in _YEAR_DATA_CACHE:
        filename = f"raw_data/load/load_frcstd_hist_{year}.csv"
        try:
            zdf = pd.read_csv(filename)
            zdf['forecast_hour_beginning_ept'] = pd.to_datetime(
                zdf['forecast_hour_beginning_ept'], 
                format='%m/%d/%Y %I:%M:%S %p'
            )
            _YEAR_DATA_CACHE[year] = zdf
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find load data file: {filename}")
        except Exception as e:
            raise ValueError(f"Error loading data for year {year}: {e}")
    return _YEAR_DATA_CACHE[year]

@lru_cache(maxsize=10000)
def _get_day_weights(year: int, month: int, day: int) -> Tuple[Dict[str, float], float]:
    """
    Calculate and cache all zone weights for a specific day.
    
    Returns:
        Tuple of (zone_to_weight_dict, total_load)
    """
    zdf = _load_year_data(year)
    
    # Filter for the specific day/month/year
    mask = ((zdf['forecast_hour_beginning_ept'].dt.day == day) & 
            (zdf['forecast_hour_beginning_ept'].dt.month == month) & 
            (zdf['forecast_hour_beginning_ept'].dt.year == year))
    day_data = zdf[mask]
    
    if day_data.empty:
        warnings.warn(f"No data found for {year}-{month:02d}-{day:02d}")
        return {}, 0.0
    
    # Calculate average forecast_load_mw for each zone on that day
    zone_loads = day_data.groupby('forecast_area')['forecast_load_mw'].mean()
    
    # Calculate total load across all zones
    load_total = zone_loads.sum()
    
    if load_total == 0:
        warnings.warn(f"Total load is zero for {year}-{month:02d}-{day:02d}")
        return {}, 0.0
    
    # Calculate weights for all zones
    zone_weights = (zone_loads / load_total).to_dict()
    
    return zone_weights, load_total

def get_zonal_weight(zone_name: str, day: int, month: int, year: int) -> float:
    """
    Get the zonal weight for a specific zone on a specific day.
    
    Uses caching to avoid recalculating weights for the same day.
    
    Args:
        zone_name: Name of the zone (e.g., 'AEP')
        day: Day of month (1-31)
        month: Month (1-12)
        year: Year (e.g., 2025)
    
    Returns:
        Weight as a float (zone's load / total load for that day)
    """
    zone_weights, _ = _get_day_weights(year, month, day)
    return zone_weights.get(zone_name, 0.0)

def clear_cache():
    """Clear both the year data cache and the day weights cache."""
    global _YEAR_DATA_CACHE
    _YEAR_DATA_CACHE.clear()
    _get_day_weights.cache_clear()

def get_cache_info():
    """Get information about cache usage."""
    return {
        'year_data_cached': list(_YEAR_DATA_CACHE.keys()),
        'day_weights_cache_info': _get_day_weights.cache_info()
    }

if __name__ == "__main__":
    # Test correct aggregation of weights
    
    # Setup
    zdf = pd.read_csv("raw_data/load/load_frcstd_hist_2025.csv")
    zones = zdf['forecast_area'].unique().tolist()
    
    # Calculate
    weights = 0
    for zone in zones:
        weight = get_zonal_weight(zone, 1, 1, 2025)
        weights += weight
    
    print(f"Total weight: {weights}")
    assert abs(weights - 1.0) < 1e-10, f"Weights sum to {weights}, expected 1.0"
    
    # Show cache statistics
    print("\nCache info:")
    info = get_cache_info()
    print(f"Years cached: {info['year_data_cached']}")
    print(f"Day weights cache: {info['day_weights_cache_info']}")
    
    # Test cache efficiency - second call should be instant
    print("\nTesting cache efficiency...")
    import time
    
    # Clear cache to get accurate first call timing
    clear_cache()
    
    start = time.perf_counter()
    w1 = get_zonal_weight('AEP', 1, 1, 2025)
    elapsed1 = time.perf_counter() - start
    
    start = time.perf_counter()
    w2 = get_zonal_weight('AEP', 1, 1, 2025)
    elapsed2 = time.perf_counter() - start
    
    print(f"First call: {elapsed1*1000:.4f}ms")
    print(f"Cached call: {elapsed2*1000:.4f}ms")
    if elapsed2 > 0:
        print(f"Speedup: {elapsed1/elapsed2:.1f}x")
    else:
        print(f"Cached call too fast to measure accurately (< {elapsed2*1000:.6f}ms)")