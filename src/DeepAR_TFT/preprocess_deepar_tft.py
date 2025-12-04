import config
import pandas as pd
import numpy as np
import logging
from typing import Tuple
import torch
from pytorch_forecasting import TimeSeriesDataSet

from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder, TorchNormalizer, EncoderNormalizer
from sklearn.preprocessing import StandardScaler


# Set up Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define global variables ---
# Columns
STATIC_CAT_COLS = config.STATIC_CAT_COLS
STATIC_REAL_COLS = config.STATIC_REAL_COLS
DYNAMIC_KNOWN_CAT_COLS = config.DYNAMIC_KNOWN_CAT_COLS
DYNAMIC_KNOWN_REAL_COLS_DENORM = config.DYNAMIC_KNOWN_REAL_COLS_DENORM
DYNAMIC_KNOWN_REAL_COLS_NORM = config.DYNAMIC_KNOWN_REAL_COLS_NORM
DYNAMIC_UNKNOWN_REAL_COLS = config.DYNAMIC_UNKNOWN_REAL_COLS
IS_ACTIVE = config.IS_ACTIVE
WEIGHT = config.WEIGHT
GROUP_ID = config.GROUP_ID
TARGET = config.TARGET
TIME_COL = config.TIME_COL
# Directories
DATA_DIR = config.DATA_DIR
# Parameters
ENCODER_LENGTH = config.ENCODER_LENGTH
HORIZON = config.HORIZON

def load_splits() -> dict | None:
    """
    Function that loads base preprocessing.

    Returns:
        dict: A dictionary containing all the dataframe splits.
    """
    # --- 1. Setup paths and check for existing segmented splits ---
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    segmented_splits_path = processed_dir / "segmented_splits.parquet"

    if segmented_splits_path.exists():
        logger.info(f"--- Found existing segmented splits. Loading from {segmented_splits_path} ---")
        df = pd.read_parquet(segmented_splits_path)
        # Ensure categorical types
        df['fold'] = df['fold'].astype('category')
        df['split_type'] = df['split_type'].astype('category')

        # --- 2. Recreate splits dictionary from parquet ---
        splits = {}
        for fold_name in df['fold'].cat.categories:
            splits[fold_name] = {}
            fold_df = df[df['fold'] == fold_name]
            for split_type in fold_df['split_type'].cat.categories:
                # Drop the helper columns before storing
                split_df = fold_df[fold_df['split_type'] == split_type].drop(columns=['fold', 'split_type']).copy()
                splits[fold_name][split_type] = split_df
        logger.info("--- Reconstructed splits dictionary from Parquet file. ---")
        return splits
    else:
        logger.error("--- Existing segmented splits not found. Please run the main environment first.")
        return None

def preprocess_deepar_tft_fold(splits: dict, fold: str) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Conducts full preprocessing for DeepAR and TFT fold.

    Args:
        splits (dict): The dictionary containing all pre-defined data splits.
        fold (str): The fold to prepare data for ('fold1', 'fold2', 'hold_out').

    Returns:
        tuple: A tuple containing the training, validation, and test TimeSeriesDatasets.
    """
    # --- 1. Define Paths and Check for Final Output ---
    processed_dir = DATA_DIR / "processed" / "DeepAR_TFT" / fold
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_dataset_path = processed_dir / "train_dataset.pt"
    val_dataset_path = processed_dir / "val_dataset.pt"
    test_dataset_path = processed_dir / "test_dataset.pt"

    if train_dataset_path.exists() and val_dataset_path.exists() and test_dataset_path.exists():
        logger.info(f"--- Found existing datasets for {fold}. Loading... ---")
        train_data = torch.load(train_dataset_path, weights_only=False)
        val_data = torch.load(val_dataset_path, weights_only=False)
        test_data = torch.load(test_dataset_path, weights_only=False)
        return train_data, val_data, test_data

    # --- 2. Get the correct dataframes for the fold and combine for further processing ---
    train_df = splits[fold]['train'].copy()
    validation_df = splits[fold]['validation'].copy()
    test_df = splits[fold]['test'].copy()
    combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

    # --- 3. Create the TimeSeriesDataSet for training ---
    logger.info(f"--- Creating TimeSeriesDataSet for {fold} ---")

    # Ensure all categorical features are strings
    all_categoricals = STATIC_CAT_COLS + DYNAMIC_KNOWN_CAT_COLS
    for col in all_categoricals:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(str)
    
    # Ensure all real-valued features are floats
    all_reals = STATIC_REAL_COLS + DYNAMIC_KNOWN_REAL_COLS_DENORM + DYNAMIC_KNOWN_REAL_COLS_NORM + DYNAMIC_UNKNOWN_REAL_COLS + [IS_ACTIVE]
    for col in all_reals:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype(np.float32)

    # The training_cutoff tells the dataset where the training data ends.
    training_cutoff = train_df['time_idx'].max()

    # Scalers
    # scalers = {}

    scalers = {
    col: EncoderNormalizer(method="identity", center=False)
    for col in DYNAMIC_KNOWN_REAL_COLS_NORM + [IS_ACTIVE]}

    for col in DYNAMIC_KNOWN_REAL_COLS_DENORM + STATIC_REAL_COLS:
        scalers[col] = EncoderNormalizer(method="standard", transformation="log1p")
    logger.info(f"Used scalers for {len(scalers)} real features: {DYNAMIC_KNOWN_REAL_COLS_NORM + [IS_ACTIVE]} remain unscaled, {DYNAMIC_KNOWN_REAL_COLS_DENORM + STATIC_REAL_COLS} Standard Scaled")

    # To allow NaN categories
    categorical_encoders = {}
    for col in all_categoricals:
        categorical_encoders[col] = NaNLabelEncoder(add_nan=True)

    training_dataset = TimeSeriesDataSet(
        combined_df.loc[combined_df.time_idx <= training_cutoff],
        time_idx=TIME_COL,
        target=TARGET,
        group_ids=[GROUP_ID],
        
        static_categoricals=STATIC_CAT_COLS,
        static_reals=STATIC_REAL_COLS,
        
        time_varying_known_reals=DYNAMIC_KNOWN_REAL_COLS_DENORM + DYNAMIC_KNOWN_REAL_COLS_NORM + [IS_ACTIVE],
        time_varying_known_categoricals=DYNAMIC_KNOWN_CAT_COLS,
        time_varying_unknown_reals=DYNAMIC_UNKNOWN_REAL_COLS,
        
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=HORIZON,
        weight = WEIGHT, # We specify masking from first sold
        target_normalizer=TorchNormalizer(method='identity', center=False),
        scalers=scalers,
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True
    )

    # --- 4. Create Validation Set ---
    validation_cutoff = validation_df['time_idx'].max()
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, combined_df.loc[combined_df.time_idx <= validation_cutoff], 
        min_prediction_idx=training_cutoff + 1, 
        predict=True, 
        stop_randomization=True
    )
    
    # --- 5. Create Test Set ---
    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, 
        combined_df, 
        min_prediction_idx=validation_cutoff + 1, 
        predict=True, 
        stop_randomization=True
    )

    # --- 6. Save the results ---
    training_dataset.save(train_dataset_path)
    validation_dataset.save(val_dataset_path)
    test_dataset.save(test_dataset_path)
    logger.info(f"--- Saved datasets for {fold} to {processed_dir} ---")

    return training_dataset, validation_dataset, test_dataset

if __name__ == "__main__":
    pass