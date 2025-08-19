import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import toml
from pathlib import Path
from typing import List, Union
from pydantic import BaseModel, validator
from feature import BaseFeature

class OHLCV(BaseFeature):
    # Class variable to cache the entire OHLCV dataset
    ohlcv_data = None
    
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
            
        #Handle incorrect start/end timestamps
        if start > end:
            raise ValueError("start timestamp must be before or equal to end timestamp")
        
        # Load data into class variable if not already loaded
        if OHLCV.ohlcv_data is None:
            self._load_ohlcv_data()
        
        return OHLCV.ohlcv_data.loc[start:end][pairs]
    
    def _load_ohlcv_data(self):
        """
        Load the entire OHLCV parquet file into the class variable.
        This is called only once when the data is first needed.
        """
        try:
            # Load config to get ohlcv_path
            config_path = Path(__file__).parent / "horcrux_config.toml"
            config = toml.load(config_path)
            ohlcv_path = Path(config["ohlcv_path"]).expanduser()
            
            # Load the entire parquet file into memory
            print(f"Loading OHLCV data from {ohlcv_path}...")
            OHLCV.ohlcv_data = pd.read_parquet(ohlcv_path, engine='pyarrow')
            
            print(f"OHLCV data loaded successfully. Shape: {OHLCV.ohlcv_data.shape}")   
        except FileNotFoundError:
            raise FileNotFoundError(f"OHLCV parquet file not found at {ohlcv_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading OHLCV data: {str(e)}")
    
    @classmethod
    def clear_cache(cls):
        """
        Clear the cached OHLCV data to free up memory.
        Useful for testing or when you want to reload the data.
        """
        cls.ohlcv_data = None
        print("OHLCV data cache cleared.")