from .feature import Feature
import pandas as pd
from typing import List

class ZScore(Feature):
    """
    Transforms any given feature into its Z-score (standardized score).
    
    The Z-score is calculated as: z = (x - μ) / σ
    where μ is the rolling mean and σ is the rolling standard deviation.
    
    This transformation:
    - Centers the data around zero (mean = 0)
    - Scales the data to unit variance (std = 1)
    - Makes features comparable across different scales
    - Helps identify outliers (values beyond ±2 or ±3 standard deviations)
    - Improves model performance by normalizing input features
    """
    
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], 
                     base_feature: Feature, window: int = 30*24*60, min_periods: int = None) -> pd.DataFrame:
        """
        Compute the Z-score of a given feature using rolling statistics.
        
        Args:
            start: Start timestamp for the computation period
            end: End timestamp for the computation period  
            pairs: List of trading pairs to compute the feature for
            base_feature: The feature object to standardize (e.g., RSI(), MACD(), etc.)
            window: Rolling window size in minutes for mean/std calculation 
                   (default: 30 days * 24 hours * 60 minutes = 43,200 minutes)
            min_periods: Minimum number of observations required to have a value
                        (default: window // 4 to ensure statistical significance)
            
        Returns:
            pd.DataFrame: Z-score normalized values of the base feature
                         - Single feature: columns are pairs
                         - Multi-feature: MultiIndex with (pair, feature_name)
        """
        # Set default min_periods to 1/4 of window for statistical robustness
        if min_periods is None:
            min_periods = max(1, window // 4)
        
        # Get the base feature data with extra history for rolling calculations
        # We need additional data points to calculate meaningful rolling statistics
        extended_start = start - pd.Timedelta(minutes=window)
        base_data = base_feature.compute(extended_start, end, pairs)
        
        # Handle both single and multi-feature cases
        if isinstance(base_data.columns, pd.MultiIndex):
            # Multi-feature case (e.g., MACD with macd_line and macd_histogram)
            zscore_data = pd.DataFrame(index=base_data.index)
            
            # Process each pair and feature combination
            for pair in pairs:
                if pair in base_data.columns.get_level_values(0):
                    pair_data = base_data[pair]
                    
                    for feature_name in pair_data.columns:
                        feature_series = pair_data[feature_name]
                        
                        # Calculate rolling mean and standard deviation
                        rolling_mean = feature_series.rolling(
                            window=window, 
                            min_periods=min_periods
                        ).mean()
                        
                        rolling_std = feature_series.rolling(
                            window=window, 
                            min_periods=min_periods
                        ).std()
                        
                        # Compute Z-score: (value - mean) / std
                        # Handle division by zero by setting zscore to 0 when std is 0
                        zscore = (feature_series - rolling_mean) / rolling_std
                        zscore = zscore.fillna(0)  # Handle NaN values
                        zscore = zscore.replace([float('inf'), float('-inf')], 0)  # Handle infinite values
                        
                        # Store with MultiIndex column structure
                        zscore_data[(pair, f"{feature_name}_zscore")] = zscore
            
            # Create proper MultiIndex for columns
            zscore_data.columns = pd.MultiIndex.from_tuples(
                zscore_data.columns, 
                names=['pair', 'feature']
            )
            
        else:
            # Single feature case (e.g., RSI, simple moving average)
            zscore_data = pd.DataFrame(index=base_data.index, columns=base_data.columns)
            
            for pair in pairs:
                if pair in base_data.columns:
                    feature_series = base_data[pair]
                    
                    # Calculate rolling statistics
                    rolling_mean = feature_series.rolling(
                        window=window, 
                        min_periods=min_periods
                    ).mean()
                    
                    rolling_std = feature_series.rolling(
                        window=window, 
                        min_periods=min_periods
                    ).std()
                    
                    # Compute Z-score with robust error handling
                    zscore = (feature_series - rolling_mean) / rolling_std
                    zscore = zscore.fillna(0)  # Replace NaN with 0
                    zscore = zscore.replace([float('inf'), float('-inf')], 0)  # Replace inf with 0
                    
                    zscore_data[pair] = zscore
        
        # Return only the requested time range to match the interface specification
        return zscore_data.loc[start:end]
    
class AdaptiveZScore(Feature):
    """
    An adaptive Z-score that adjusts the window size based on market volatility.
    Uses a shorter window during volatile periods and longer window during stable periods.
    """
    
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str],
                     base_feature: Feature, base_window: int = 30*24*60, 
                     volatility_lookback: int = 7*24*60, adaptation_factor: float = 0.5) -> pd.DataFrame:
        """
        Compute adaptive Z-score that adjusts window size based on market volatility.
        
        Args:
            start: Start timestamp
            end: End timestamp
            pairs: List of trading pairs
            base_feature: Feature to standardize
            base_window: Base window size in minutes (default: 30 days)
            volatility_lookback: Period to measure volatility in minutes (default: 7 days)
            adaptation_factor: How much to adjust window (0-1, default: 0.5)
            
        Returns:
            pd.DataFrame: Adaptive Z-score values
        """
        # Get extended data for calculations
        max_window = int(base_window * 2)  # Maximum possible window
        extended_start = start - pd.Timedelta(minutes=max_window)
        
        # Get base feature data
        base_data = base_feature.compute(extended_start, end, pairs)
        
        # Get OHLCV data to measure volatility
        from .ohlcv import OHLCV
        ohlcv = OHLCV().compute(extended_start, end, pairs)
        close_prices = ohlcv.xs("close", axis=1, level=1)
        
        # Calculate rolling volatility (standard deviation of returns)
        returns = close_prices.pct_change()
        volatility = returns.rolling(window=volatility_lookback).std()
        
        # Normalize volatility to [0, 1] range for each pair
        vol_normalized = volatility.div(volatility.rolling(window=base_window).quantile(0.95), axis=1)
        vol_normalized = vol_normalized.clip(0, 1).fillna(0.5)
        
        # Calculate adaptive window sizes
        # High volatility -> shorter window, Low volatility -> longer window
        adaptive_windows = base_window * (1 - adaptation_factor * vol_normalized)
        adaptive_windows = adaptive_windows.round().astype(int)
        
        # Compute adaptive Z-scores
        if isinstance(base_data.columns, pd.MultiIndex):
            zscore_data = pd.DataFrame(index=base_data.index)
            
            for pair in pairs:
                if pair in base_data.columns.get_level_values(0):
                    pair_data = base_data[pair]
                    pair_windows = adaptive_windows[pair]
                    
                    for feature_name in pair_data.columns:
                        feature_series = pair_data[feature_name]
                        zscore_series = pd.Series(index=feature_series.index, dtype=float)
                        
                        # Calculate adaptive Z-score for each timestamp
                        for timestamp in feature_series.index:
                            if timestamp in pair_windows.index:
                                window_size = max(pair_windows[timestamp], volatility_lookback)
                                end_idx = feature_series.index.get_loc(timestamp)
                                start_idx = max(0, end_idx - window_size + 1)
                                
                                window_data = feature_series.iloc[start_idx:end_idx+1]
                                if len(window_data) > 1:
                                    mean_val = window_data.mean()
                                    std_val = window_data.std()
                                    if std_val > 0:
                                        zscore_series[timestamp] = (feature_series[timestamp] - mean_val) / std_val
                                    else:
                                        zscore_series[timestamp] = 0
                        
                        zscore_data[(pair, f"{feature_name}_adaptive_zscore")] = zscore_series
            
            zscore_data.columns = pd.MultiIndex.from_tuples(zscore_data.columns, names=['pair', 'feature'])
            
        else:
            zscore_data = pd.DataFrame(index=base_data.index, columns=base_data.columns)
            
            for pair in pairs:
                if pair in base_data.columns and pair in adaptive_windows.columns:
                    feature_series = base_data[pair]
                    pair_windows = adaptive_windows[pair]
                    zscore_series = pd.Series(index=feature_series.index, dtype=float)
                    
                    for timestamp in feature_series.index:
                        if timestamp in pair_windows.index:
                            window_size = max(pair_windows[timestamp], volatility_lookback)
                            end_idx = feature_series.index.get_loc(timestamp)
                            start_idx = max(0, end_idx - window_size + 1)
                            
                            window_data = feature_series.iloc[start_idx:end_idx+1]
                            if len(window_data) > 1:
                                mean_val = window_data.mean()
                                std_val = window_data.std()
                                if std_val > 0:
                                    zscore_series[timestamp] = (feature_series[timestamp] - mean_val) / std_val
                                else:
                                    zscore_series[timestamp] = 0
                    
                    zscore_data[pair] = zscore_series
        
        return zscore_data.loc[start:end]