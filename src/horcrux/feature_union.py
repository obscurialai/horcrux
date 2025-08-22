from .feature import Feature
import pandas as pd
from typing import List

class FUnion(Feature):
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], features: List[Feature]):
        computed_features = [feature.compute(start, end, pairs) for feature in features]
        
        # Convert non-MultiIndex dataframes to MultiIndex format
        multiindex_features = []
        for i, df in enumerate(computed_features):
            if isinstance(df.columns, pd.MultiIndex):
                # Already MultiIndex, use as is
                multiindex_features.append(df)
            else:
                # Convert to MultiIndex: (pair, class_name)
                class_name = features[i].__class__.__name__
                
                # Create MultiIndex columns where first level is pair, second level is class name
                new_columns = pd.MultiIndex.from_product(
                    [df.columns, [class_name]], 
                    names=['pair', 'feature']
                )
                
                # Reshape the dataframe to match MultiIndex structure
                df_multiindex = df.copy()
                df_multiindex.columns = new_columns
                multiindex_features.append(df_multiindex)
        
        return pd.concat(multiindex_features, axis=1)