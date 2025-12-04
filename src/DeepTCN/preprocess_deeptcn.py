import config
import pandas as pd
import numpy as np
import joblib
import pickle
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from typing import Tuple
from typing import Optional


# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- DEFINE GLOBAL VARIAABLES ---
# Define directories
DATA_DIR = config.DATA_DIR
# Define configrations
INPUT_SIZE = config.ENCODER_LENGTH  # e.g., 6 months of history
OUTPUT_SIZE = config.HORIZON # e.g., 3 months of forecast
SAMPLE_LEN = INPUT_SIZE + OUTPUT_SIZE # Full window length
# Define columns
TARGET_COL = config.TARGET
ID_COL = config.GROUP_ID
TIME_COL = config.TIME_COL
WEIGHT = config.WEIGHT
IS_ACTIVE = config.IS_ACTIVE
STATIC_CAT_COLS = config.STATIC_CAT_COLS
DYNAMIC_CAT_COLS = config.DYNAMIC_KNOWN_CAT_COLS
STATIC_REAL_COLS = config.STATIC_REAL_COLS
DYNAMIC_REAL_COLS_DENORM = config.DYNAMIC_KNOWN_REAL_COLS_DENORM
DYNAMIC_REAL_COLS_NORM = config.DYNAMIC_KNOWN_REAL_COLS_NORM

# Define which cols to scale vs. encode
COLS_TO_SCALE = STATIC_REAL_COLS + DYNAMIC_REAL_COLS_DENORM
COLS_NOT_SCALE = DYNAMIC_REAL_COLS_NORM + [IS_ACTIVE]
ALL_REAL_COLS = COLS_TO_SCALE + COLS_NOT_SCALE


def load_base_preprocessing() -> Optional[dict]:
    """
    Loads pre-saved base dataset preprocessing.

    Returns:
        dict: A dictionary containing fold splits
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
        logger.error("--- Pre-saved base dataset preprocessing not found. Please run the main script from the main environment first. ---")
        return None

def _fit_preprocessors(train_df: pd.DataFrame) -> dict:
    """
    Fits scalers and encoders on the training data.

    Args:
        train_df (pd.DataFrame): training data.

    Returns:
        dict: A dictionary containing preprocessing.
    """
    logger.info("Fitting preprocessors on training data...")
    preprocessors = {'scalers': {}, 'encoders': {}, 'cat_features_info': {}}

    # --- 1. Fit ID Encoder ---
    id_encoder = LabelEncoder()
    # Add a placeholder for unknown IDs during inference
    all_ids = np.append(train_df[ID_COL].unique(), 'UNKNOWN_ID')
    id_encoder.fit(all_ids)
    preprocessors['encoders'][ID_COL] = id_encoder
    # Save cardinality (num unique + 1 for unknown)
    preprocessors['cat_features_info'][f'{ID_COL}_encoded'] = len(id_encoder.classes_)

    # --- 2. Fit Covariate Scaler (for all real-valued covariates) ---
    covariate_scaler = StandardScaler()
    preprocessors['scalers']['covariates'] = covariate_scaler.fit(np.log1p(train_df[COLS_TO_SCALE]))

    # --- 3. Fit LabelEncoders for all categorical covariates ---
    for col in STATIC_CAT_COLS + DYNAMIC_CAT_COLS:
        le = LabelEncoder()
        # Add a placeholder for unknown categories
        all_cats = np.append(train_df[col].unique(), 'UNKNOWN_CAT')
        le.fit(all_cats)
        preprocessors['encoders'][col] = le
        # Save cardinality
        preprocessors['cat_features_info'][col] = len(le.classes_)
        
    logger.info("Preprocessors fitted successfully.")
    logger.info(f"Categorical Feature Info: {preprocessors['cat_features_info']}")
    return preprocessors


def _preprocess_dataframe(df: pd.DataFrame, preprocessors: dict) -> Tuple[pd.DataFrame, list]:
    """
    Applies fitted preprocessors to a new dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame to be processed
        preprocessors (dict): Dictionary of preprocessing to be done.

    Returns:
        pd.DataFrame: Scaled DataFrame.
        list: Names of time series.
    """
    logger.debug(f"Preprocessing dataframe with shape {df.shape}")
    df_processed = df.copy()

    # --- 1. Apply Target Scaler ---
    df_processed[TARGET_COL] = df_processed[TARGET_COL].astype(np.float32)

    # --- 2. Apply ID Encoder ---
    id_encoder = preprocessors['encoders'][ID_COL]
    df_processed[f'{ID_COL}_encoded'] = df_processed[ID_COL].map(
        lambda x: id_encoder.transform([x])[0] if x in id_encoder.classes_ else id_encoder.transform(['UNKNOWN_ID'])[0]
    )

    # --- 3. Apply Covariate Scaler (Real features) ---
    scaled_real_features = preprocessors['scalers']['covariates'].transform(np.log1p(df_processed[COLS_TO_SCALE]))
    scaled_real_df = pd.DataFrame(scaled_real_features, columns=COLS_TO_SCALE, index=df_processed.index)

    # --- 4. Apply LabelEncoders (Categorical features) ---
    encoded_cat_cols_list = [] # Just the names
    encoded_cat_df_list = []   # The actual data
    
    # Handle the ID col first (which is already encoded)
    encoded_cat_cols_list.append(f'{ID_COL}_encoded')
    encoded_cat_df_list.append(df_processed[[f'{ID_COL}_encoded']])

    # Handle all other categorical cols
    for col in STATIC_CAT_COLS + DYNAMIC_CAT_COLS:
        col_name_encoded = col # Use original col name
        encoded_cat_cols_list.append(col_name_encoded)
        
        le = preprocessors['encoders'][col]
        known_classes_set = set(le.classes_)
        encoded_series = df_processed[col].map(
            lambda x: le.transform([x])[0] if x in known_classes_set else le.transform(['UNKNOWN_CAT'])[0]
        )
        encoded_cat_df_list.append(pd.Series(encoded_series, name=col_name_encoded, index=df_processed.index))

    # Final dataframe of categorical features
    final_cat_df = pd.concat(encoded_cat_df_list, axis=1)

    # --- 5. Combine all processed features back together ---
    non_scaled_real_df = df_processed[COLS_NOT_SCALE + [WEIGHT]].copy()
    final_df = pd.concat([
        df_processed[[ID_COL, TIME_COL, TARGET_COL]],
        scaled_real_df,
        non_scaled_real_df,
        final_cat_df
    ], axis=1)

    logger.debug("Dataframe preprocessed successfully.")

    return final_df, encoded_cat_cols_list


def _create_windows(df: pd.DataFrame, real_cols: list, cat_cols: list) -> dict:
    """
    Generates sliding windows from the preprocessed dataframe.
    Creates separate arrays for target, real covariates, and categorical covariates.
    
    Args:
        df (pd.DataFrame): DataFrame to be processed.
        real_cols (list): List of countable covariates.
        cat_cols (list): List of categorical covariates.

    Returns:
        dict: Dictionary containing windowed historical target, future target, future mask, real covariates, and categorical covariates.
    """
    logger.info(f"Generating sliding windows (input_size={INPUT_SIZE}, output_size={OUTPUT_SIZE})...")
    
    # These lists will hold arrays, one for each series
    x_hist_target_list, y_future_target_list = [], []
    x_full_cat_list, x_full_real_list = [], []
    mask_future_list = []
    index_list = []

    grouped = df.groupby(ID_COL)
    
    # --- 1. Loop through the time series ---
    for stockcode, series in tqdm(grouped, desc="Processing series"):
        series = series.sort_values(TIME_COL)
        
        target = series[TARGET_COL].values
        real_covariates = series[real_cols].values
        cat_covariates = series[cat_cols].values
        time_indices = series[TIME_COL].values
        mask_arr = series[WEIGHT].values 
        
        total_len = len(series)
        
        # --- 2. Loop through all possible windows ---
        for i in range(total_len - SAMPLE_LEN + 1):
            # Define window indices
            hist_start = i
            hist_end = i + INPUT_SIZE
            future_end = i + SAMPLE_LEN # hist_end + OUTPUT_SIZE
            
            # --- 2.1. Historical Target (for Encoder) ---
            x_hist_target_list.append(target[hist_start:hist_end].reshape(-1, 1))

            # --- 2.2. Future Target ---
            y_future_target_list.append(target[hist_end:future_end].reshape(-1, 1))
            mask_future_list.append(mask_arr[hist_end:future_end].reshape(-1, 1))

            # --- 2.3. Full Covariates (Historical + Future) ---
            x_full_real_list.append(real_covariates[hist_start:future_end])
            x_full_cat_list.append(cat_covariates[hist_start:future_end])

            # --- 2.4 Index ---
            forecast_start_time_idx = time_indices[hist_end] # Get the time_idx for the first forecast step
            index_list.append((stockcode, forecast_start_time_idx))

    # --- 3. Create arrays for DeepTCN ---
    x_target_arr = np.array(x_hist_target_list)
    y_target_arr = np.array(y_future_target_list)
    mask_future_arr = np.array(mask_future_list)
    x_real_arr = np.array(x_full_real_list)
    x_cat_arr = np.array(x_full_cat_list)
    index_arr = np.array(index_list, dtype=[('StockCode', 'O'), ('time_idx', '<i8')])
    
    logger.info(f"Generated {x_target_arr.shape[0]} total windows.")
    
    return {
        'x_hist_target': x_target_arr,     # (N, L_in)
        'y_future_target': y_target_arr,   # (N, L_out)
        'mask_future': mask_future_arr,    # (N, L_out)
        'x_full_real_cov': x_real_arr,     # (N, L_in + L_out, n_real)
        'x_full_cat_cov': x_cat_arr,       # (N, L_in + L_out, n_cat)
        'index': index_arr
    }


def _filter_windows_by_time(df_full: pd.DataFrame, df_period: pd.DataFrame, real_cols: list, cat_cols: list) -> dict:
    """
    Window generation function for val/test sets that ensures windows *start* in the correct period.

    Args:
        df_full (pd.DataFrame): Full data containing train+val or train+val+test data.
        df_period (pd.DataFrame): Data containing only validation or test data to determine starting period.
        real_cols (list): List of countable covariates.
        cat_cols (list): List of categorical covariates.

    Returns:
        dict: Dictionary containing windowed historical target, future target, future mask, real covariates, and categorical covariates.
    """
    logger.info(f"Generating sliding windows for specific period...")
    
    x_hist_target_list, y_future_target_list = [], []
    x_full_cat_list, x_full_real_list = [], []
    mask_future_list = []
    index_list = []
    
    # --- 1. Filter by the start of the *future* window ---
    valid_future_start_indices = set(df_period[TIME_COL].unique())
    
    grouped = df_full.groupby(ID_COL)
    
    # --- 2. Loop through the time series ---
    for stockcode, series in tqdm(grouped, desc="Processing series"):
        series = series.sort_values(TIME_COL)
        
        target = series[TARGET_COL].values
        real_covariates = series[real_cols].values
        cat_covariates = series[cat_cols].values
        time_idx = series[TIME_COL].values
        mask_arr = series[WEIGHT].values 

        
        total_len = len(series)
        
        # --- 3. Loop through all possible windows ---
        for i in range(total_len - SAMPLE_LEN + 1):
            future_start_time = time_idx[i + INPUT_SIZE]
            
            # Filters by the start date
            if future_start_time in valid_future_start_indices:
                hist_start = i
                hist_end = i + INPUT_SIZE
                future_end = i + SAMPLE_LEN
                
                x_hist_target_list.append(target[hist_start:hist_end].reshape(-1, 1))
                y_future_target_list.append(target[hist_end:future_end].reshape(-1, 1))
                mask_future_list.append(mask_arr[hist_end:future_end].reshape(-1, 1))
                x_full_real_list.append(real_covariates[hist_start:future_end])
                x_full_cat_list.append(cat_covariates[hist_start:future_end])
                index_list.append((stockcode, future_start_time))

    # --- 4. Create arrays for DeepTCN ---
    x_target_arr = np.array(x_hist_target_list)
    y_target_arr = np.array(y_future_target_list)
    mask_future_arr = np.array(mask_future_list)
    x_real_arr = np.array(x_full_real_list)
    x_cat_arr = np.array(x_full_cat_list)
    index_arr = np.array(index_list, dtype=[('StockCode', 'O'), ('time_idx', '<i8')])
    
    logger.info(f"Generated {x_target_arr.shape[0]} total windows.")
    
    return {
        'x_hist_target': x_target_arr,
        'y_future_target': y_target_arr,
        'mask_future': mask_future_arr,
        'x_full_real_cov': x_real_arr,
        'x_full_cat_cov': x_cat_arr,
        'index': index_arr
    }


def save_as_pickle(data_dict: dict, file_path: Path):
    """
    Saves the final formatted data as a .pkl file.
    
    Args:
        data_dict (dict): Dictionary containing arrays.
        file_path (Path): Path where to save the data.
    """
    # --- 1. Define data to save ---
    repo_formatted_data = {
        'x_target': data_dict['x_hist_target'],      # xNum
        'y_target': data_dict['y_future_target'],  # y_true
        'x_cat': data_dict['x_full_cat_cov'],      # xCat
        'x_real': data_dict['x_full_real_cov'],    # xReal
        'mask_future': data_dict['mask_future'],
        'index': data_dict['index']
    }
    
    # --- 2. Save the data ---
    with open(file_path, 'wb') as f:
        pickle.dump(repo_formatted_data, f, protocol=4)
    logger.info(f"Successfully saved formatted data to {file_path}")


def prepare_data_deeptcn(splits, fold='fold1'):
    """
    Main orchestration script to load, preprocess, and format the data.

    Args:
        splits (dict): dictionary containing the data for folds.
        fold (str): Current fold name.

    Returns:
        (Path): Path to the training processed data.
        (Path): Path to the validation processed data.
        (Path): Path to the test processed data.
        (Path): Path to the preprocessor dictionary.
    """
    # --- 1. Define Paths ---
    output_dir = DATA_DIR / 'processed' / 'DeepTCN' / fold
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_file = output_dir / 'preprocessors.joblib'

    if preprocessor_file.exists():
        logger.info(f'{fold} preprocessing already exists. Saving paths...')
        return (output_dir / f'train_{fold}.pkl',
            output_dir / f'validation_{fold}.pkl',
            output_dir / f'test_{fold}.pkl',
            preprocessor_file)

    # --- 2. Define data for the fold ---
    train_raw = splits[fold]['train']
    val_raw = splits[fold]['validation']
    test_raw = splits[fold]['test']
    
    logger.info(f"Loaded {fold}: {len(train_raw)} train rows, {len(val_raw)} val rows, {len(test_raw)} test rows.")

    # --- 3. Fit and Save Preprocessors ---
    preprocessors = _fit_preprocessors(train_raw)
    joblib.dump(preprocessors, preprocessor_file)
    logger.info(f"Preprocessors saved to {preprocessor_file}")

    # --- 4. Process All Sets ---
    logger.info("--- Processing Training Set ---")
    train_processed, cat_cols = _preprocess_dataframe(train_raw, preprocessors)
    logger.info(f"Total number of real covariates: {len(ALL_REAL_COLS)}")
    logger.info(f"Total number of categorical covariates: {len(cat_cols)}")
    
    logger.info("--- Processing Validation Set ---")
    val_processed, _ = _preprocess_dataframe(val_raw, preprocessors)

    logger.info("--- Processing Test Set ---")
    test_processed, _ = _preprocess_dataframe(test_raw, preprocessors)

    # --- 5. Create Training Windows ---
    logger.info("--- Creating Training Windows ---")
    train_windows = _create_windows(train_processed, ALL_REAL_COLS, cat_cols)
    save_as_pickle(train_windows, output_dir / f'train_{fold}.pkl')
    
    # --- 6. Create Validation Windows ---
    # We need historical context to build the *first* validation window
    df_for_val = pd.concat([train_processed, val_processed]).sort_values([ID_COL, TIME_COL]).drop_duplicates()
    val_windows = _filter_windows_by_time(df_for_val, val_processed, ALL_REAL_COLS, cat_cols)
    if val_windows['x_hist_target'].shape[0] > 0:
        save_as_pickle(val_windows, output_dir / f'validation_{fold}.pkl')
    else:
        logger.warning("No validation windows were created. Check window size and data.")

    # --- 7. Create Test Windows ---
    # Need history from train + val sets
    df_for_test = pd.concat([train_processed, val_processed, test_processed]).sort_values([ID_COL, TIME_COL]).drop_duplicates()
    test_windows = _filter_windows_by_time(df_for_test, test_processed, ALL_REAL_COLS, cat_cols)
    if test_windows['x_hist_target'].shape[0] > 0:
        save_as_pickle(test_windows, output_dir / f'test_{fold}.pkl')
    else:
        logger.warning("No test windows were created. Check window size and data.")

    logger.info("--- All data successfully formatted for DeepTCN ---")

    return (output_dir / f'train_{fold}.pkl',
            output_dir / f'validation_{fold}.pkl',
            output_dir / f'test_{fold}.pkl',
            preprocessor_file)

def preprocess_deeptcn():
    """
    Full pipeline to preprocess data for DeepTCN.

    Returns:
        data_path: Dictionary containing paths to data files for each fold.
    """
    # --- 1. Load pre-constructed base preprocessing ---
    data_splits = load_base_preprocessing()
    if data_splits is None:
        return None

    # --- 2. Loop through the folds to preprocess for DeepTCN ---
    data_path = {}

    for fold in data_splits:
        tr_path, val_path, test_path, preproc_path = prepare_data_deeptcn(data_splits, fold)
        data_path[fold] = {'train_path': tr_path, 'validation_path': val_path, 'test_path': test_path, 'preprocessors': preproc_path}
    
    return data_path

if __name__ == "__main__":
    pass