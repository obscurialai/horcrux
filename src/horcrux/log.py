from .feature import Feature
import pandas as pd
from typing import List
import numpy as np

class FLog(Feature):
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], base_feature: Feature):
        return np.log(base_feature.compute(start, end, pairs))
        