import config
from pathlib import Path
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import joblib

# GluonTS Imports
from gluonts.dataset.common import ListDataset


# Set up Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define global variables ---
# Columns
STATIC_CAT_COLS = config.DEEPAR_STATIC_CAT_COLS
STATIC_REAL_COLS = config.DEEPAR_STATIC_REAL_COLS
DYNAMIC_REAL_COLS = config.DEEPAR_DYNAMIC_REAL_COLS 
TARGET_COLUMN = config.DEEPAR_TARGET_COLUMN
GROUP_ID = config.DEEPAR_GROUP_ID
# Directories
DATA_DIR = config.DATA_DIR
# Parameters
ENCODER_LENGTH = config.ENCODER_LENGTH
FREQ = config.FREQ


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


def _get_categorical_mappings(df_full: pd.DataFrame) -> dict:
    """
    Computes categorical mappings for categrical columns.
    Support function for preprocess_deepar_fold

    Args:
        df_full (pd.DataFrame): full dataset (based of concatenated hold-out fold)
    
    Returns:
        dict: dictionary containing factorized categories for each categorical column.
    """
    mappings = {}
    for col in STATIC_CAT_COLS: 
        if col in df_full.columns:
             codes, uniques = pd.factorize(df_full[col].astype(str))
             mappings[col] = {
                 'codes': dict(zip(df_full[col].astype(str), codes)), 
                 'categories': uniques.tolist() 
             }
    return mappings


def _convert_to_gluonts_format(df: pd.DataFrame, cat_mappings: dict) -> ListDataset:
    """
    Converts a pandas DataFrame into the GluonTS ListDataset format.
    Support function for preprocess_deepar_fold.

    Args:
        df (pd.DataFrame): Data to convert.
        cat_mappings (dict): Factorized categoreis of categorical columns.

    Returns:
        ListDataset: Gluonts object for data.
    """  
    # Set up list for ListDataset
    data = []

    # --- 1. Loop through the product ids ---
    for group_values, group_data in df.groupby(GROUP_ID):
        # Define start for the series (should be same for each in a fold)
        start_ts = group_data['Week'].min()
        target_array = group_data[TARGET_COLUMN].values.astype(np.float32)
        
        # --- 2. Process Static Categorical Features ---
        static_cat_list = []
        for col in STATIC_CAT_COLS:
            if col in group_data.columns and col in cat_mappings:
                cat_value = str(group_data[col].iloc[0]) 
                cat_code = cat_mappings[col]['codes'].get(cat_value, 0) 
                static_cat_list.append(cat_code)
            else:
                static_cat_list.append(0) 
        feat_static_cat = np.array(static_cat_list, dtype=np.int64)

        # --- 3. Process Static Real Features ---
        static_real_list = []
        for col in STATIC_REAL_COLS:
             if col in group_data.columns:
                 val = group_data[col].iloc[0]
                 static_real_list.append(val if pd.notna(val) else 0.0) 
             else:
                  static_real_list.append(0.0) 
        feat_static_real = np.array(static_real_list, dtype=np.float32)

        # --- 4. Process Dynamic Real Features ---
        # (Dynamic cat features are removed, handled by `time_features` in estimator)
        dynamic_real_list = []
        for col in DYNAMIC_REAL_COLS:
             if col in group_data.columns:
                 val_array = group_data[col].values.astype(np.float32)
                 dynamic_real_list.append(val_array)
             else:
                  logger.debug(f"Missing dynamic real feature {col} for group {group_values}")
                  dynamic_real_list.append(np.zeros_like(target_array, dtype=np.float32))
        
        # Transform into dynamic real arrays
        if dynamic_real_list:           
            final_dynamic_real_list = []
            for col in DYNAMIC_REAL_COLS:
                 if col in group_data.columns:
                     final_dynamic_real_list.append(group_data[col].values.astype(np.float32))
                 else:
                     final_dynamic_real_list.append(np.zeros(len(target_array), dtype=np.float32))

            try:
                feat_dynamic_real = np.vstack(final_dynamic_real_list)
            except ValueError as e:
                logger.error(f"Error stacking dynamic features for group {group_values}: {e}")
                logger.error(f"Array lengths: {[len(arr) for arr in final_dynamic_real_list]}")
                continue
        else:
            feat_dynamic_real = None

        # --- 5. Form dictionary with the processed values for columns ---
        item_data = {
            "start": start_ts,
            "target": target_array,
            "feat_static_cat": feat_static_cat,
            "feat_static_real": feat_static_real,
            "feat_dynamic_real": feat_dynamic_real,
        }
        item_data = {k: v for k, v in item_data.items() if v is not None} 
        data.append(item_data)

    return ListDataset(data, freq=FREQ)


def preproccess_deepar_fold(splits: dict, fold: str) -> Tuple[ListDataset, ListDataset, ListDataset, dict]:
    """
    Conducts full preprocessing for DeepAR fold.

    Args:
        splits (dict): Dictionary containin all segmented fold splits.
        fold (str): Name of the current fold.

    Returns:
        traing_data_path (ListDataset): ListDataset for the preprocessed training data.
        val_data_path (ListDataset): ListDataset for the preprocessed validation data.
        test_data_path (ListDataset): ListDataset for the preprocessed test data.
        cat_mappings_path (dict): Dictionary with the categorical mappings.
    """
    # --- 1. Define directories ---
    data_save_path = DATA_DIR / 'processed' / 'DeepAR' / fold
    data_save_path.mkdir(parents=True, exist_ok=True)
    train_data_path = data_save_path / f'train_{fold}.joblib'
    val_data_path = data_save_path / f'val_{fold}.joblib'
    test_data_path = data_save_path / f'test_{fold}.joblib'
    cat_mappins_path = data_save_path / f'cat_mappings_{fold}.joblib'

    if train_data_path.exists() and val_data_path.exists() and test_data_path.exists() and cat_mappins_path.exists():
        logger.info(f'{fold} preprocessing already exists. Loading from {data_save_path}')
        train_data_gluon = joblib.load(train_data_path)
        val_data_gluon = joblib.load(val_data_path)
        test_data_gluon = joblib.load(test_data_path)
        cat_mappings = joblib.load(cat_mappins_path)
        return train_data_gluon, val_data_gluon, test_data_gluon, cat_mappings

    # --- 2. Prepare data splits ---
    logger.info(f"--- Preparing GluonTS data for {fold} ---")
    train_df = splits[fold]['train'].copy()
    validation_df = splits[fold]['validation'].copy()
    test_df = splits[fold]['test'].copy()

    # --- 3. Compute categorical mappings ---
    logger.info("Computing categorical mappings...")
    full_dataset_df = pd.concat([splits['hold_out']['train'], splits['hold_out']['validation'], splits['hold_out']['test']], ignore_index=True)
    cat_mappings = _get_categorical_mappings(full_dataset_df)

    # --- 4. Transform training data to GluonTS format ---
    logger.info("Converting training data...")
    train_data_gluon = _convert_to_gluonts_format(train_df, cat_mappings)

    # --- 5. Transform validation data to GluonTS format ---
    logger.info("Converting validation data...")
    full_val_df = pd.concat([train_df.groupby('StockCode').tail(ENCODER_LENGTH), validation_df], ignore_index=True)
    val_data_gluon = _convert_to_gluonts_format(full_val_df, cat_mappings)

    # --- 6. Transform test data to GluonTS format ---
    logger.info("Converting test data...")
    # Get full history for prediction
    full_history_df = pd.concat([train_df, validation_df], ignore_index=True)
    full_test_df = pd.concat([full_history_df.groupby('StockCode').tail(ENCODER_LENGTH), test_df], ignore_index=True)
    test_data_gluon = _convert_to_gluonts_format(full_test_df, cat_mappings)

    # --- 7. Save ListDatasets ---
    joblib.dump(train_data_gluon, train_data_path)
    joblib.dump(val_data_gluon, val_data_path)
    joblib.dump(test_data_gluon, test_data_path)
    joblib.dump(cat_mappings, cat_mappins_path)

    return train_data_gluon, val_data_gluon, test_data_gluon, cat_mappings







