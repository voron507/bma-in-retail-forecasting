import config
import pandas as pd
import numpy as np
import logging
import time
import json
import gc
from typing import Tuple
from pathlib import Path
from warnings import simplefilter
import joblib
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
from scipy.stats import norm
from codecarbon import OfflineEmissionsTracker
from utils.evaluate_preds import calculate_crps, calculate_wape, calculate_rmse, compare_statistics

# --- Set up logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Suppress specific warnings from statsforecast ---
simplefilter(action="ignore", category=RuntimeWarning)
simplefilter(action="ignore", category=UserWarning)
simplefilter(action="ignore", category=FutureWarning)

# --- Define global variables ---
# Directories
FORECASTS_DIR = config.FORECASTS_DIR
MODELS_DIR = config.MODELS_DIR
METRICS_DIR = config.METRICS_DIR
# Columns
GROUP_ID = config.GROUP_ID
TIME_COL = config.TIME_COL
TARGET = config.TARGET
IS_ACTIVE = config.IS_ACTIVE
# Configuration
HORIZON = config.HORIZON
N_SAMPLES = config.N_SAMPLES
BASELINE_LEVEL = config.BASELINE_LEVEL


def _prepare_data(splits: dict, fold: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.array]:
    """
    Prepares data for the baseline prediction.

    Args:
        splits (dict): Dictionary containing fold data.
        fold (str): Current fold to process.

    Returns:
        pd.Dataframe: Training data.
        pd.Dataframe: Target data.
        np.array: Target mask.
    """
    logger.info(f"Preparing data for {fold}...")

    # --- 1. Extract Training and Test Data ---
    train_ds = splits[fold]['train']
    val_ds = splits[fold]['validation']
    df_full = pd.concat([train_ds, val_ds], ignore_index=True)
    test_ds = splits[fold]['test']

    # --- 2. Filter for Established products ---
    all_prods = df_full[GROUP_ID].unique()
    est_prods = df_full[df_full['ProductStatus'] == 'Established'][GROUP_ID].unique()
    logger.info(f"Filtering established products: {len(est_prods)} out of {len(all_prods)} kept.")
    df_full = df_full[df_full[GROUP_ID].isin(est_prods)].copy()
    test_ds = test_ds[test_ds[GROUP_ID].isin(est_prods)].copy()

    # --- 3. Apply mask to inactive products ---
    # Product may have history in training that didn't start from day 1 - we mask before 1st sale 0 volume
    logger.info(f"Applying mask: filtering for {IS_ACTIVE} > 0")
    df_full = df_full[df_full[IS_ACTIVE] > 0].copy()

    # --- 4. Prepare dataframes ---
    cols_to_keep = [GROUP_ID, TIME_COL, TARGET, 'AvgPrice']

    df_train = df_full[cols_to_keep].rename(columns={
        GROUP_ID: 'unique_id',
        TIME_COL: 'ds',
        TARGET: 'y'
    })
    df_train['unique_id'] = df_train['unique_id'].astype(str)

    y_mask = test_ds[[IS_ACTIVE]].to_numpy()
    df_test = test_ds[cols_to_keep].rename(columns={
        GROUP_ID: 'unique_id',
        TIME_COL: 'ds',
        TARGET: 'y'
    })
    df_test['unique_id'] = df_test['unique_id'].astype(str)

    # Sort for consistency
    df_train = df_train.sort_values(by=['unique_id', 'ds'])
    df_test = df_test.sort_values(by=['unique_id', 'ds'])

    del df_full
    gc.collect()

    return df_train, df_test, y_mask


def _statsforecast_to_samples(forecast_df: pd.DataFrame, model_name: str, n_samples: int = N_SAMPLES, level: int = BASELINE_LEVEL) -> Tuple[np.array, np.recarray, np.array]:
    """
    Converts StatsForecast intervals (Gaussian approximation) into probabilistic samples.
    
    Args:
        forecast_df (pd.DataFrame): Results from StatsForecast prediction.
        model_name (str): Name of the model used for prediction.
        n_samples (int): Number of samples to generate.
        level (int): Prediction interval chosen.

    Returns:
        np.array: Generated samples.
        np.recarray: Structured index to be passed.
        np.array: unique ids.
    """
    # --- 1. Sort to ensure consistent order ---
    df = forecast_df.sort_values(by=['unique_id', 'ds']).copy()
    
    # --- 2. Extract Columns ---
    mean_col = model_name
    lo_col = f'{model_name}-lo-{level}'
    hi_col = f'{model_name}-hi-{level}'
    
    if lo_col not in df.columns:
        raise ValueError(f"Interval columns for {model_name} not found. Check if level=[{level}] was passed.")

    mu = df[mean_col].values.astype(np.float32)
    hi = df[hi_col].values.astype(np.float32)
    lo = df[lo_col].values.astype(np.float32)

    # --- 3. Calculate Sigma from CI Width ---
    alpha = 1 - (level / 100.0)
    z_score = norm.ppf(1 - alpha / 2)
    sigma = (hi - lo) / (2 * z_score)
    
    # --- 4. Generate Samples (Vectorized) ---
    # Shape: (N_rows, n_samples)
    noise = np.random.normal(0, 1, size=(len(df), n_samples)).astype(np.float32)
    samples_flat = mu[:, None] + sigma[:, None] * noise
    
    # --- 5. Clip at Zero (Enforce non-negativity) ---
    samples_flat = np.maximum(samples_flat, 0)
    
    # --- 6. Reshape to (N_series, Horizon, n_samples) ---
    unique_ids = df['unique_id'].unique()
    n_series = len(unique_ids)
    horizon = len(df) // n_series
    
    if len(df) % n_series != 0:
        logger.warning(f"Forecast length ({len(df)}) not perfectly divisible by N_series ({n_series}). Reshape might fail.")
    
    samples_3d = samples_flat.reshape(n_series, horizon, n_samples)

    # --- 7. Create Standardized Structured Index ---
    stock_codes = df['unique_id'].values
    time_indices = df['ds'].values

    structured_index = np.core.records.fromarrays(
        [stock_codes, time_indices], 
        names='StockCode, time_idx'
    )
    
    return samples_3d, structured_index, unique_ids


def _run_single_model(fold: str, model_name: str, model_obj, df_train: pd.DataFrame, df_test: pd.DataFrame, y_mask: np.array):
    """
    Executes training, prediction, and cost tracking for a single baseline model.
    
    Args:
        fold (str): Current fold to process.
        model_name (str): Model to use in prediction.
        model_obj (obj): Pre-confidured model to be used in prediction.
        df_train (pd. DataFrame): Training set.
        df_test (pd.DataFrame): Testing set.
    """
    # Setup names
    if model_name == 'SeasonalNaive':
        save_name = 'naive'
    elif model_name == 'AutoARIMA':
        save_name = 'sarimax'
    else:
        save_name = 'etsx'

    # --- 1. Setup Paths ---
    metrics_path = METRICS_DIR / "Baselines"
    metrics_path.mkdir(parents=True, exist_ok=True)
    cost_metrics_path = metrics_path / f"baseline_costs_{save_name}_{fold}.json"

    pred_dir = FORECASTS_DIR / "Baselines"
    pred_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = pred_dir / f"test_predictions_{save_name}_{fold}.npz"

    if predictions_path.exists():
        logger.info(f"Predictions for {save_name} model, {fold} already exist. Skipping...")
        return None
    
    # --- 2. Start Tracking ---
    logger.info(f"--- Starting execution for {save_name} on {fold} ---")
    if not cost_metrics_path.exists():
        tracker = OfflineEmissionsTracker(
        country_iso_code="NLD", 
        output_dir=METRICS_DIR, 
        project_name=f"{save_name}_{fold}", 
        log_level='error'
        )
        tracker.start()
        start_time = time.time()

    try:
        # --- 3. Fit & Predict
        # Apart from exogenous must also include Groups and time periods
        future_exog = df_test[['unique_id', 'ds', 'AvgPrice']].reset_index(drop=True)

        sf = StatsForecast(
            models=[model_obj],
            freq=1, 
            n_jobs=-1,
            fallback_model=None
        )

        forecasts_df = sf.forecast(
            df=df_train, 
            h=HORIZON, 
            X_df=future_exog,
            level=[BASELINE_LEVEL]
        )
        forecasts_df = forecasts_df.reset_index()

        del future_exog, sf
        gc.collect()

        # --- 4. Generate Samples ---
        samples, structured_index, ids = _statsforecast_to_samples(forecasts_df, model_name, N_SAMPLES, BASELINE_LEVEL)

        # --- 5. Debug ---
        n_series = len(ids)
        test_merged = forecasts_df.merge(df_test, on=['unique_id', 'ds'], how='inner')
        y_pred = test_merged[model_name].to_numpy().round().reshape(n_series, HORIZON)
        y_true = test_merged['y'].to_numpy().reshape(n_series, HORIZON)
        y_mask = y_mask.reshape(n_series, HORIZON)

        del test_merged, forecasts_df
        gc.collect()

        rmse = calculate_rmse(y_pred, y_true, y_mask)
        logger.info(f"--- Test Set RMSE (Mean vs y_true): {rmse:.4f} ---")
        wape = calculate_wape(y_pred, y_true, y_mask)
        logger.info(f"--- Test Set WAPE (p50 vs y_true): {wape:.4f} ---")
        # crps = calculate_crps(y_true, samples, y_mask)
        # logger.info(f"--- Test Set CRPS: {crps:.4f} ---")
        avg_mean, avg_std, emp_zero_rate_p50, y_true_mean, y_true_std, emp_zero_rate_y = compare_statistics(y_pred, y_pred, y_true, y_mask)
        logger.debug(f"    -> P50 Predictions: Mean={avg_mean:.4f}, Std={avg_std:.4f}, Empirical zero rate={emp_zero_rate_p50:.4f}")
        logger.debug(f"    -> Y_True Values:   Mean={y_true_mean:.4f}, Std={y_true_std:.4f}, Empirical zero rate={emp_zero_rate_y:.4f}")

        # --- 6. Save Predictions ---
        np.savez_compressed(
            predictions_path,
            samples=samples,
            point_forecasts=y_pred,
            test_index=structured_index
        )
    
    except Exception as e:
        logger.error(f"{save_name} failed: {e}", exc_info=True)
        raise e
    
    finally:
        if not cost_metrics_path.exists():
            # --- 7. Stop Tracking ---
            end_time = time.time()
            total_time_seconds = end_time - start_time
            emissions_data = tracker.stop()
            
            cost_metrics = {
                "fold": fold,
                "model": save_name,
                "total_time_seconds": round(total_time_seconds, 2),
                "co2_emissions_kg": emissions_data if emissions_data is not None else 0
            }
            
            with open(cost_metrics_path, 'w') as f:
                json.dump(cost_metrics, f, indent=4)
            logger.info(f"--- Costs saved to {cost_metrics_path} ---")
        
        gc.collect()


def run_baseline_pipeline(fold: str, splits: dict):
    """
    Runs Seasonal Naive, SARIMAX, and ETSX models on a given fold.

    Args:
        fold (str): Current fold to be proccessed.
        splits (dict): Dictionary containing fold data
    """
    # --- 1. Prepare Data ---
    df_train, df_test, y_mask = _prepare_data(fold=fold, splits=splits)

    # --- 2. Run Seasonal Naive ---
    _run_single_model(
        fold=fold,
        model_name="SeasonalNaive",
        model_obj=SeasonalNaive(season_length=52),
        df_train=df_train,
        df_test=df_test,
        y_mask=y_mask
    )

    # --- 3. Run SARIMAX ---
    _run_single_model(
        fold=fold,
        model_name="AutoARIMA",
        model_obj=AutoARIMA(season_length=52),
        df_train=df_train,
        df_test=df_test,
        y_mask=y_mask
    )

    # --- 4. Run ETSX ---
    _run_single_model(
        fold=fold,
        model_name="AutoETS",
        model_obj=AutoETS(season_length=52, model='ZZZ'),
        df_train=df_train,
        df_test=df_test,
        y_mask=y_mask
    )

if __name__ == "__main__":
    pass
