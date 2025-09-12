from .feature import Feature
import pandas as pd
import numpy as np
import numba
from typing import List
from .ohlcv import OHLCV
import numba

@numba.njit(cache=True)
def calculate_single_exit_index_log(entry_index, close_log, high_log_bt, low_log_bt, tp_log, sl_log):
    #Tolerance value needed because otherwise sometimes the sl/tp doesn't trigger because of numerical issues
    epsilon = 1e-12
    
    tp = close_log[entry_index] + tp_log
    sl = close_log[entry_index] + sl_log
    #binary tree midpoint to get start of the single step data
    bt_midpoint = len(high_log_bt) >> 1
    #We take +1 as the starting point
    current_index = entry_index + bt_midpoint + 1
    #Exit not detected yet
    while True:
        high = high_log_bt[current_index]   
        low = low_log_bt[current_index]
        #We detected exit so we leave the loop
        if high > tp - epsilon or low < sl + epsilon:
            break
        else:
            #This condition means that we have reached the right edge and are going to the right out of bounds therefore there is no trigger for the sl/tp in the whole data
            if current_index & (current_index + 1) == 0:
                return bt_midpoint 
            #If we are in the right cell we can't go up so we step to the right. If we are in the left cell we go up the binary tree
            current_index = current_index + 1 if (current_index & 1) else current_index >> 1
    #Exit detected
    while True:
        #If we are at the bottom of the tree we have reached the exit
        if current_index > bt_midpoint:
            return current_index - bt_midpoint  # Return the index instead of close price
        #We descend the tree and then check both of the leaves
        current_index = current_index << 1
        
        high = high_log_bt[current_index]   
        low = low_log_bt[current_index]
        
        #If the exit is triggered in the first leaf we go there, if not it is triggered in the second leaf. It must be triggered somewhere so if not the first then the second
        current_index = current_index if (high > tp - epsilon or low < sl + epsilon) else current_index + 1
        
@numba.njit(cache=True)
def calculate_exit_log_return(entries, close, high, low, tp_frac, sl_frac):
    high_log = np.log(high)
    low_log = np.log(low)
    close_log = np.log(close)
    
    #Init the binary tree arrays
    high_log_bt = np.full(len(high_log) * 2, np.nan)
    low_log_bt = np.full(len(low_log) * 2, np.nan)
    
    #The bottom level of the BT is just the values of the high and the low
    high_log_bt[len(high_log): 2*len(high_log)] = high_log
    low_log_bt[len(low_log): 2*len(low_log)] = low_log

    current_length = len(high)
    
    #Construct binary tree for high and low by resampling pairs
    while current_length != 0:
        current_length = current_length // 2
        for i in range(current_length, 2*current_length):
            high_log_bt[i] = max(high_log_bt[2*i], high_log_bt[2*i + 1])
            low_log_bt[i] = min(low_log_bt[2*i], low_log_bt[2*i + 1])
             
    result = np.full(len(entries), np.nan, dtype=np.float64)  # Use -1 for no exit, int32 for indices
    for i in range(len(entries)):
        if entries[i]:
            exit_index = calculate_single_exit_index_log(i, close_log, high_log_bt, low_log_bt, np.log(1+tp_frac), np.log(1-sl_frac))
            exit_index = min(exit_index, len(entries)-1)
            log_return = close_log[exit_index] - close_log[i]
            result[i] = log_return
    return result

def fast_exit(entries, ohlcv, tp_frac, sl_frac):
    log_return = pd.DataFrame(-1, index=entries.index, columns=entries.columns, dtype=np.float64)
    log2 = np.ceil(np.log2(len(entries)))
    power_of_2 = int(2 ** log2)
    
    for pair in entries.columns.get_level_values(0).unique():
        entries_for_pair = entries[pair].to_numpy(dtype=np.bool_)
        close_for_pair   = ohlcv[pair]['close'].to_numpy()
        high_for_pair    = ohlcv[pair]['high'].to_numpy()
        low_for_pair     = ohlcv[pair]['low'].to_numpy()

        #We pad the data to a power of 2 so that we can construct a binary tree
        close_for_pair = np.pad(close_for_pair, (0, power_of_2 - len(close_for_pair)), mode='edge')
        high_for_pair = np.pad(high_for_pair, (0, power_of_2 - len(high_for_pair)), mode='edge')
        low_for_pair = np.pad(low_for_pair, (0, power_of_2 - len(low_for_pair)), mode='edge')
        log_return[pair] = calculate_exit_log_return(entries_for_pair, close_for_pair, high_for_pair, low_for_pair, tp_frac, sl_frac)[:len(entries_for_pair)]
        
    return log_return


class TPSL_LogReturn(Feature):
    
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], tp_frac = 0.05, sl_frac = 0.05) -> pd.DataFrame:
        # Get extra history for rolling calculations
        ohlcv = OHLCV().compute(start, end, pairs)
        entries = pd.DataFrame(True, index=ohlcv.index, columns=ohlcv.columns.get_level_values(0).unique())
        tp_sl_log_return = fast_exit(entries, ohlcv, tp_frac, sl_frac)
        
        # Create MultiIndex DataFrame with (pair, feature) structure
        result = pd.DataFrame(index=tp_sl_log_return.index)
        for pair in pairs:
            if pair in tp_sl_log_return.columns:
                result[(pair, 'tpsl_logreturns')] = tp_sl_log_return[pair]
        
        # Set proper MultiIndex column names
        result.columns = pd.MultiIndex.from_tuples(result.columns, names=['pair', 'feature'])
        
        return result 