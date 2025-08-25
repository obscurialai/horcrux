from .feature import Feature
import pandas as pd
from typing import List

class FUnion(Feature):
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], features: List[Feature], add_hash_to_features = True):
        computed_features = [feature.compute(start, end, pairs, add_hash = add_hash_to_features) for feature in features]
        
        return pd.concat(computed_features, axis = 1)
        