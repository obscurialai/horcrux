from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union
from pydantic import BaseModel

class Feature:
    def __init__(self, *args, fields: Union[None, List[str]] = None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.fields = fields
    
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
            return output.loc[:, pd.IndexSlice[pairs, self.fields]]
        return output
    
    @abstractmethod
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], *args, **kwargs):
        raise NotImplementedError