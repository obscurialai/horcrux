from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union
from pydantic import BaseModel
import inspect
import json
import base64, hashlib

class Feature:
    def __init__(self, pairs: Union[str, List[str]], **kwargs):
        self.kwargs = kwargs
        # Convert string pairs to list if needed
        if isinstance(pairs, str):
            self.pairs = [pairs]
        else:
            self.pairs = pairs
            
        #For now for convenience we will only use the first 10 characters of the hash
        self.hash = self.__compute_hash()[:10]
    
    def _ensure_multiindex_columns(self, output: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame has MultiIndex columns. If not, convert it by using
        pairs as the first level and the feature class name as the second level.
        """
        if not isinstance(output.columns, pd.MultiIndex):
            # Get the feature class name
            feature_name = self.__class__.__name__
            pairs = output.columns
            
            # Create MultiIndex columns with pairs as first level and feature name as second level
            new_columns = []
            for pair in pairs:
                for col in output.columns:
                    new_columns.append((pair, feature_name))
            
            output.columns = pd.MultiIndex.from_tuples(new_columns)
        
        return output
    
    def compute(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], add_hash: bool = False, convert_to_multiindex = False):
        # Convert string inputs to pd.Timestamp if needed
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)
        
        # Convert timezone-naive timestamps to UTC
        if start.tz is None:
            start = start.tz_localize('UTC')
        if end.tz is None:
            end = end.tz_localize('UTC')
        
        output = self._compute_impl(start, end, self.pairs, **self.kwargs).loc[start:end]
        
        # Ensure the DataFrame has MultiIndex columns
        if convert_to_multiindex:
            output = self._ensure_multiindex_columns(output)
        
        #If add_hash = True add hashes to the columnnames
        if add_hash:
            output = self.add_hash_to_output_columns(output)
        
        return output
    
    def add_hash_to_output_columns(self, output: pd.DataFrame) -> pd.DataFrame:
        # Get the current column names
        new_columns = []
        
        for pair, feature_name in output.columns:
            # Check if the feature name already has a hash appended
            # Format is FEATURENAME$HASH where hash is 10 characters
            # So we check if the 11th character from the right is '$'
            if len(feature_name) >= 11 and feature_name[-11] == '$':
                # Hash already present, keep the original name
                new_columns.append((pair, feature_name))
            else:
                # No hash present, append our hash
                new_feature_name = f"{feature_name}${self.hash}"
                new_columns.append((pair, new_feature_name))
        
        # Update the column names
        output.columns = pd.MultiIndex.from_tuples(new_columns)
        
        return output
    
    def save_to(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], file_location: str) -> pd.DataFrame:
        """
        Compute the feature and save it to a parquet file optimized for time-based queries.
        
        Args:
            start: Start timestamp
            end: End timestamp  
            pairs: List of pairs to compute
            file_location: Path to the parquet file
            
        Returns:
            pd.DataFrame: The computed feature dataframe
        """
        import os
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Compute the feature
        output = self.compute(start, end, add_hash=True, convert_to_multiindex=True)
        
        # Ensure the output is sorted by index (timestamp) for better query performance
        output = output.sort_index()
        
        # Check if the parquet file exists
        if os.path.exists(file_location):
            # Load existing data
            existing_df = pd.read_parquet(file_location)
            
            # Merge with new data
            # We'll use pd.concat and remove duplicates, keeping the newer data
            combined = pd.concat([existing_df, output])
            
            # Remove duplicate rows based on index (timestamp) and columns
            # Keep last occurrence (the newer data)
            combined = combined[~combined.index.duplicated(keep='last')]
            
            # Sort by index
            combined = combined.sort_index()
            
            # Save the combined data with optimization
            table = pa.Table.from_pandas(combined)
            pq.write_table(
                table, 
                file_location,
                compression='snappy',  # Fast compression for read performance
                row_group_size=10000,  # Smaller row groups for efficient month-by-month reading
                use_dictionary=True,   # Dictionary encoding for repeated values
                data_page_size=1024*1024,  # 1MB data pages
                version='2.6'  # Latest parquet version for better performance
            )
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_location), exist_ok=True)
            
            # Save the output directly with optimization
            table = pa.Table.from_pandas(output)
            pq.write_table(
                table, 
                file_location,
                compression='snappy',  # Fast compression for read performance
                row_group_size=10000,  # Smaller row groups for efficient month-by-month reading
                use_dictionary=True,   # Dictionary encoding for repeated values
                data_page_size=1024*1024,  # 1MB data pages
                version='2.6'  # Latest parquet version for better performance
            )
        
        return output
    
    @abstractmethod
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], **kwargs):
        raise NotImplementedError
    
    #TODO hashing does not support other features yet, only valid json objects like str, float and bool.
    #Need to write a custom serialization function that correctly serializes feature objects and also things like pd.DateTime etc
    def __compute_hash(self):
        identifier = {
            "code": inspect.getsource(self.__class__),
            "pairs": self.pairs,
            "kwargs": self.kwargs
        }
        identifier_json = json.dumps(identifier, sort_keys = True, default = str)
        hash_bytes = hashlib.sha256(identifier_json.encode()).digest()
        compact_hash_encoding = base64.urlsafe_b64encode(hash_bytes).decode()
        return compact_hash_encoding
    
    def test_leak(self):
        full_start = pd.Timestamp("2024-01-01", tz="UTC")
        step = pd.Timedelta(days=30)
        n = 10
        chunks = []
        for i in range(0, n):
            start = full_start + i*step
            end = full_start + (i+1)*step
            chunk = self.compute(start, end)
            chunks.append(chunk)
        
        chunks_df = pd.concat(chunks)
        full_df = self.compute(full_start, full_start + n*step)
        return full_df - chunks_df