from horcrux import Feature
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