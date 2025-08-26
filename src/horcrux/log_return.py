from .feature import Feature
import pandas as pd
from typing import List
import numpy as np
from .log import FLog
from .ohlcv import OHLCV

class FLogReturns(Feature):
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], offset = 15):
        abs_offset = np.abs(offset)
        offset_timedelta = pd.Timedelta(minutes = abs_offset)
        result = None
        #if offset is positive then that means we are looking into the past so we are not data leaking
        if offset >= 0:
            log_price = FLog(OHLCV(fields = ["close"])).compute(start- offset_timedelta, end, pairs)
            result = (log_price - log_price.shift(abs_offset))
        #if offset is negative then that means we are looking into the future so we are data leaking
        if offset < 0:
            log_price = FLog(OHLCV(fields = ["close"])).compute(start, end + offset_timedelta, pairs)
            result = (log_price.shift(-abs_offset) - log_price)
        
        # Create multiindex columns with pair as first level and "log_return" as second level
        result.columns = pd.MultiIndex.from_product([result.columns, ["log_return"]], names=['pair', 'feature'])
        return result
        