from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union
from pydantic import BaseModel, validator

class BaseFeature(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    
    @abstractmethod
    def calculate(self):
        pass
    
    def get_feature(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], pairs: Union[str, List[str]]):
        # Convert string inputs to pd.Timestamp if needed
        if isinstance(start, str):
            start = pd.Timestamp(start)
        if isinstance(end, str):
            end = pd.Timestamp(end)
        # Convert string pairs to list if needed
        if isinstance(pairs, str):
            pairs = [pairs]
        
        output = self.calculate(start, end, pairs, *self.args, **self.kwargs)
        return output