from .feature import Feature
import pandas as pd
from typing import List, Union
import os
import logging
import time
import traceback
from datetime import datetime

class FUnion(Feature):
    def _compute_impl(self, start: pd.Timestamp, end: pd.Timestamp, pairs: List[str], features: List[Feature], add_hash_to_features = True):
        computed_features = [feature.compute(start, end, pairs, add_hash = add_hash_to_features, convert_to_multiindex = True) for feature in features]
        
        return pd.concat(computed_features, axis = 1)
    
    def save_to(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], 
                pairs: Union[str, List[str]], file_directory: str, 
                log_dir: Union[str, None] = None) -> None:
        """
        Save each feature to its own parquet file with comprehensive logging.
        
        Args:
            start: Start timestamp
            end: End timestamp
            pairs: List of pairs to compute
            file_directory: Directory where feature parquet files will be saved
            log_dir: Directory for log files (defaults to file_directory)
        """
        # Setup logging
        if log_dir is None:
            log_dir = file_directory
        
        # Create directories if they don't exist
        os.makedirs(file_directory, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique log filename with timestamp
        log_filename = os.path.join(log_dir, f"feature_union_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Configure logger
        logger = logging.getLogger(f"FUnion_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicate logs
        logger.handlers = []
        
        # Create file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Log start of process
        logger.info(f"Starting FUnion save_to process")
        logger.info(f"Start: {start}, End: {end}, Pairs: {pairs}")
        logger.info(f"File directory: {file_directory}")
        logger.info(f"Number of features: {len(self.kwargs.get('features', []))}")
        logger.info("-" * 80)
        
        # Get features from kwargs
        features = self.kwargs.get('features', [])
        
        # Process each feature
        for i, feature in enumerate(features):
            feature_start_time = time.time()
            
            try:
                # Log feature information
                logger.info(f"Processing feature {i+1}/{len(features)}")
                logger.info(f"Feature class: {feature.__class__.__name__}")
                logger.info(f"Feature args: {feature.args}")
                logger.info(f"Feature kwargs: {feature.kwargs}")
                logger.info(f"Feature hash: {feature.hash}")
                
                # Create feature-specific file location with just the hash
                feature_file_location = os.path.join(file_directory, f"{feature.hash}.parquet")
                
                logger.info(f"Saving to: {feature_file_location}")
                
                # Call save_to on the feature
                output_df = feature.save_to(start, end, pairs, feature_file_location)
                
                # Calculate duration
                duration = time.time() - feature_start_time
                logger.info(f"Feature computation took {duration:.2f} seconds")
                
                # Check for NaN values
                nan_count = output_df.isna().sum().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in output!")
                    
                    # Log detailed NaN information
                    nan_summary = output_df.isna().sum()
                    nan_cols = nan_summary[nan_summary > 0]
                    if len(nan_cols) > 0:
                        logger.warning("NaN values by column:")
                        for col, count in nan_cols.items():
                            logger.warning(f"  {col}: {count} NaN values")
                else:
                    logger.info("No NaN values found in output")
                
                # Log output shape
                logger.info(f"Output shape: {output_df.shape}")
                logger.info(f"Successfully saved feature {i+1}/{len(features)}")
                
            except Exception as e:
                # Log the error
                duration = time.time() - feature_start_time
                logger.error(f"Error processing feature {i+1}/{len(features)} after {duration:.2f} seconds")
                logger.error(f"Feature class: {feature.__class__.__name__}")
                logger.error(f"Feature hash: {feature.hash}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                
                # Continue with next feature
                logger.info("Continuing with next feature...")
            
            logger.info("-" * 80)
        
        logger.info("FUnion save_to process completed")
        
        # Close the file handler
        file_handler.close()
        logger.removeHandler(file_handler)
        