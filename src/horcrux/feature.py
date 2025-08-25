from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union
from pydantic import BaseModel
import inspect
import json
import base64, hashlib

class Feature:
    def __init__(self, *args, fields: Union[None, List[str]] = None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.fields = fields
        self.hash = self.__compute_hash()
    
    def compute(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], pairs: Union[str, List[str]]):
        # Convert string inputs to pd.Timestamp if needed
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)
        # Convert string pairs to list if needed
        if isinstance(pairs, str):
            pairs = [pairs]
        
        # Convert timezone-naive timestamps to UTC
        if start.tz is None:
            start = start.tz_localize('UTC')
        if end.tz is None:
            end = end.tz_localize('UTC')
        
        output = self._compute_impl(start, end, pairs, *self.args, **self.kwargs)
        if self.fields != None:
            output = output.loc[:, pd.IndexSlice[pairs, self.fields]]
        
        output = self.normalize_output(output)
        
        return output
    
    def normalize_output(self, output: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the output DataFrame format based on its current structure.
        
        - If single index (columns are pairs), converts to MultiIndex with format:
          (pair, CLASSNAME-{first 6 chars of hash})
        - If already MultiIndex, converts column names to format:
          CLASSNAME-COLUMNNAME-{first 6 chars of hash}
          
        Args:
            output: The DataFrame to normalize
            
        Returns:
            pd.DataFrame: Normalized DataFrame with consistent column naming
        """
        class_name = self.__class__.__name__
        hash_prefix = self.hash[:6]
        
        if isinstance(output.columns, pd.MultiIndex):
            # Multi-index case: convert column names to CLASSNAME-COLUMNNAME-HASH format
            new_columns = []
            for pair, feature_name in output.columns:
                new_feature_name = f"{class_name}-{feature_name}-{hash_prefix}"
                new_columns.append((pair, new_feature_name))
            
            output.columns = pd.MultiIndex.from_tuples(
                new_columns, 
                names=output.columns.names
            )
        else:
            # Single index case: convert to MultiIndex with CLASSNAME-HASH format
            feature_name = f"{class_name}-{hash_prefix}"
            new_columns = pd.MultiIndex.from_product(
                [output.columns, [feature_name]],
                names=['pair', 'feature']
            )
            
            # Reshape the dataframe to match MultiIndex structure
            output_normalized = output.copy()
            output_normalized.columns = new_columns
            output = output_normalized
            
        return output
    
    @abstractmethod
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], *args, **kwargs):
        raise NotImplementedError
    
    #TODO hashing does not support other features yet, only valid json objects like str, float and bool.
    #Need to write a custom serialization function that correctly serializes feature objects and also things like pd.DateTime etc
    def __compute_hash(self):
        identifier = {
            "code": inspect.getsource(self.__class__),
            "args": self.args,
            "kwargs": self.kwargs
        }
        identifier_json = json.dumps(identifier, sort_keys = True, default = str)
        hash_bytes = hashlib.sha256(identifier_json.encode()).digest()
        compact_hash_encoding = base64.b85encode(hash_bytes).decode()
        return compact_hash_encoding
    