from .feature import Feature
import pandas as pd
from typing import List

class FMultiParam(Feature):
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], base_feature: Feature, params_list: List[dict]):
        multiparam_features_computed = [base_feature(**params).compute(start, end, pairs, add_hash = True) for params in params_list]
        
        return pd.concat(multiparam_features_computed, axis = 1)
    
    def get_features(self):
        """
        Create and return a list of feature instances instead of computing them.
        
        Returns:
            List[Feature]: List of feature instances created from base_feature with different parameters
        """
        base_feature = self.kwargs.get('base_feature')
        params_list = self.kwargs.get('params_list', [])
        
        if base_feature is None:
            raise ValueError("base_feature must be provided in kwargs")
        
        return [base_feature(**params) for params in params_list]