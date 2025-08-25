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
            return output.loc[:, pd.IndexSlice[pairs, self.fields]]
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