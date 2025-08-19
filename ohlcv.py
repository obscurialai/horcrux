import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import toml
from pathlib import Path
from typing import List, Union
from pydantic import BaseModel, validator
from feature import BaseFeature

class OHLCV(BaseFeature):
    def calculate(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str]) -> pd.DataFrame:
        """
        Calculate OHLCV data for given time range and pairs.
        
        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive) 
            pairs: List of trading pairs to retrieve
            
        Returns:
            MultiIndex DataFrame with (pair, ohlcv) as columns and timestamp as index
        """
        # Validate inputs using pydantic-style validation
        if not isinstance(start, pd.Timestamp):
            raise ValueError("start must be a pandas Timestamp")
        if not isinstance(end, pd.Timestamp):
            raise ValueError("end must be a pandas Timestamp")
        if not isinstance(pairs, list) or not all(isinstance(p, str) for p in pairs):
            raise ValueError("pairs must be a list of strings")
        if start > end:
            raise ValueError("start timestamp must be before or equal to end timestamp")
        
        # Load config to get ohlcv_path
        config_path = Path(__file__).parent / "horcrux_config.toml"
        config = toml.load(config_path)
        ohlcv_path = Path(config["ohlcv_path"]).expanduser()
        
        # Use PyArrow filters for efficient time-based filtering
        # This pushes the filter down to the parquet reader, only loading relevant row groups
        filters = [
            ('timestamp', '>=', start),
            ('timestamp', '<=', end)
        ]
        
        # Add pair filtering if specific pairs are requested
        if pairs:
            pair_filter = ('pair', 'in', pairs)
            filters.append(pair_filter)
        
        try:
            # Read parquet with filters - this is the most efficient method
            # Only loads the filtered data into memory
            df = pd.read_parquet(
                ohlcv_path,
                engine='pyarrow',
                filters=filters
            )
            
            # Ensure the result is properly sorted by timestamp
            if not df.empty:
                df = df.sort_index()
                
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"OHLCV parquet file not found at {ohlcv_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading OHLCV data: {str(e)}")