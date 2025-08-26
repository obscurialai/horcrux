from .feature import Feature
import pandas as pd
from typing import List

class FMultiParam(Feature):
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], base_feature: Feature, params_list: List[dict]):
        multiparam_features_computed = [base_feature(**params).compute(start, end, pairs, add_hash = True) for params in params_list]
        
        return pd.concat(multiparam_features_computed, axis = 1)
        