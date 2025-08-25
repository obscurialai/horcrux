from .feature import Feature
import pandas as pd
import numpy as np
import numba
from typing import List
from .ohlcv import OHLCV

@numba.njit(cache=True)
def fast_linreg_slope(y, window):
    """
    Fast numba-compiled function to calculate rolling linear regression slope.
    
    Args:
        y: 1D numpy array of values
        window: Rolling window size
        
    Returns:
        1D numpy array of slope values
    """
    # Length of data
    N = len(y)
    
    # Initialize the slope array
    slope = np.full(N, np.nan)
    
    # Initialize the window x values
    x_window = np.arange(window)
    
    # Calculate the denominator of the beta formula, it does not depend on x and so is just a constant we divide by
    denom = np.sum((x_window - x_window.mean()) ** 2)
    
    # Calculate the array of x_i rolling mean values. Since x_i = i we have a an analytical expression for the mean
    x_mean = np.arange(N) - (window-1)/2
    
    # We initialize all of the variables for the first value of beta that we calculate
    # Mean of the first window values
    y_i_mean = y[:window].mean()
    c_i = ((x_window - x_mean[window-1])*(y[:window]- y_i_mean)).sum()
    slope[window-1] = c_i/denom
    
    for i in range(window-1, N-1):
        dy_i = y[i+1]-y[i-window+1]
        # Iterate y_i+1_mean
        y_i_mean = y_i_mean + dy_i/window
        # Iterate c_i+1
        c_i = c_i +dy_i+((i+1)-x_mean[i+1])*(y[i+1]-y_i_mean)-((i+1-window)-x_mean[i+1])*(y[i+1-window]-y_i_mean)
        slope[i+1] = c_i/denom
    return slope


def rolling_linear_regression_slope_fast(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the rolling slope (beta) of linear regression for a rolling window.
    
    Args:
        data: DataFrame with time series data
        window: Rolling window size (default: 14)
        
    Returns:
        DataFrame with rolling linear regression slopes
    """
    slope_series_list = []
    
    for col in data.columns:
        slope_values = fast_linreg_slope(data[col].values.astype(np.float64), window)
        slope_series = pd.Series(slope_values, index=data.index, name=col)
        slope_series_list.append(slope_series)
    
    slope_df = pd.concat(slope_series_list, axis=1)
    
    return slope_df


class RollingLinRegSlope(Feature):
    """
    Calculates the rolling linear regression slope for time series data.
    
    The linear regression slope represents the rate of change (trend) over a rolling window.
    Positive values indicate upward trends, negative values indicate downward trends.
    
    This implementation uses a numba-optimized algorithm for fast computation of rolling
    linear regression slopes, making it suitable for large datasets.
    """
    
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], 
                     base_feature: Feature, window: int = 14) -> pd.DataFrame:
        """
        Compute rolling linear regression slope for the given parameters.
        
        Args:
            start: Start timestamp for the computation period
            end: End timestamp for the computation period  
            pairs: List of trading pairs to compute the feature for
            base_feature: Feature to compute slope on
            window: Rolling window size for slope calculation (default: 14)
            
        Returns:
            pd.DataFrame: Rolling linear regression slopes with MultiIndex (pair, feature) columns
        """
        # Get extra history for rolling calculations
        extended_start = start - pd.Timedelta(minutes=window * 2)
        base_data = base_feature.compute(extended_start, end, pairs)
        
        # Create result DataFrame
        slope_data = pd.DataFrame(index=base_data.index)
        
        # Process each column in base_data
        for col in base_data.columns:
            if isinstance(base_data.columns, pd.MultiIndex):
                pair, feature_name = col
                new_col_name = (pair, f"{feature_name}_slope")
            else:
                # Simple column case - treat column name as pair
                pair = col
                new_col_name = (pair, "slope")
            
            # Only process if pair is in requested pairs
            if pair in pairs:
                feature_series = base_data[col]
                slope_values = fast_linreg_slope(feature_series.values.astype(np.float64), window)
                slope_data[new_col_name] = slope_values
        
        # Ensure MultiIndex columns
        if not isinstance(slope_data.columns, pd.MultiIndex):
            slope_data.columns = pd.MultiIndex.from_tuples(
                slope_data.columns, names=['pair', 'feature']
            )
        
        # Return only the requested time range
        return slope_data.loc[start:end] 