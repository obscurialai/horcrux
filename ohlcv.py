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
        
        try:
            # Use PyArrow Dataset API for more robust timestamp filtering
            # This avoids timestamp precision issues while maintaining efficiency
            import pyarrow.dataset as ds
            import pyarrow.compute as pc
            
            # Create dataset from parquet file
            dataset = ds.dataset(ohlcv_path, format="parquet")
            
            # Convert timestamps to PyArrow timestamps with proper timezone
            start_arrow = pa.scalar(pd.to_datetime(start).tz_localize('UTC' if start.tz is None else None).tz_convert('UTC'))
            end_arrow = pa.scalar(pd.to_datetime(end).tz_localize('UTC' if end.tz is None else None).tz_convert('UTC'))
            
            # Create filter expression using PyArrow compute functions
            filter_expr = (
                (pc.field('datetime') >= start_arrow) & 
                (pc.field('datetime') <= end_arrow)
            )
            
            # Scan and convert to pandas
            table = dataset.to_table(filter=filter_expr)
            df = table.to_pandas()
            
            # Set datetime as index if it's not already
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            
            # Filter by pairs if specific pairs are requested
            # The pairs are encoded in the column multiindex, not as a separate column
            if pairs and not df.empty:
                # Get columns that match the requested pairs
                matching_columns = []
                for pair in pairs:
                    pair_columns = [col for col in df.columns if col[0] == pair]
                    matching_columns.extend(pair_columns)
                
                if matching_columns:
                    df = df[matching_columns]
                else:
                    # Return empty dataframe with same structure if no pairs match
                    df = df.iloc[0:0]
            
            # Ensure the result is properly sorted by datetime index
            if not df.empty:
                df = df.sort_index()
                
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"OHLCV parquet file not found at {ohlcv_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading OHLCV data: {str(e)}")