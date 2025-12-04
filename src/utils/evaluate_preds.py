import cupy as cp
import numpy as np
import logging
import config
import pickle


# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Define global variables ---
FORECASTS_DIR = config.FORECASTS_DIR


def calculate_rmse(mean_preds, y_test, mask_test):
    """
    Calculates RMSE.

    Args:
        mean_preds (np.array): Contains averaged samples for each prediction.
        y_test (np.array): Contains true test values.
        mask_test (np.array): Contains mask for the valid predictions.
    
    Returns:
        rmse (float): Calculated RMSE.
    """
    # Check Prediction Existence (Baselines may not have)
    if mean_preds.shape[0] == 0:
        return None
    
    rmse = np.sqrt(np.sum(((mean_preds - y_test) ** 2) * mask_test) / (np.sum(mask_test) + 1e-8))
    return rmse


def calculate_wape(p50_preds, y_test, mask_test):
    """
    Calculates WAPE.

    Args:
        p50_preds (np.array): Contains median of samples for each prediction.
        y_test (np.array): Contains true test values.
        mask_test (np.array): Contains mask for the valid predictions.

    Returns:
        wape (float): Calculated WAPE.
    """
    # Check Prediction Existence (Baselines may not have)
    if p50_preds.shape[0] == 0:
        return None

    abs_errors = np.abs(p50_preds - y_test) * mask_test
    wape = abs_errors.sum() / (np.abs(y_test * mask_test).sum() + 1e-8)
    return wape


def compare_statistics(mean_preds, p50_preds, y_test, mask_test):
    """
    Provides basic statistics to give additional comparative angle.

    Args:
        mean_preds (np.array): Contains averaged samples for each prediction.
        p50_preds (np.array): Contains median of samples for each prediction.
        y_test (np.array): Contains true test values.
        mask_test (np.array): Contains mask for the valid predictions.

    Returns:
        avg_mean (float), avg_std (float), emp_zero_rate_p50 (float), y_true_mean (float), y_true_std (float), emp_zero_rate_y (float).
    """
    # Check Prediction Existence (Baselines may not have)
    if mean_preds.shape[0] == 0:
        return None, None, None, None, None, None
    
    avg_mean = (mean_preds * mask_test).sum() / (mask_test.sum() + 1e-8)
    avg_var = ((mean_preds - avg_mean) ** 2 * mask_test).sum() / (mask_test.sum() + 1e-8)
    avg_std = np.sqrt(avg_var)
    p50_masked = p50_preds * mask_test
    zero_mask_p50 = (p50_masked == 0) & (mask_test == 1)
    emp_zero_rate_p50 = zero_mask_p50.sum() / mask_test.sum()
    y_true_mean = (y_test * mask_test).sum() / (mask_test.sum() + 1e-8)
    y_true_var = ((y_test - y_true_mean) ** 2 * mask_test).sum() / (mask_test.sum() + 1e-8)
    y_true_std = np.sqrt(y_true_var)
    y_true_masked = y_test * mask_test
    zero_mask_y = (y_true_masked == 0) & (mask_test == 1)
    emp_zero_rate_y = zero_mask_y.sum() / mask_test.sum()
    return avg_mean, avg_std, emp_zero_rate_p50, y_true_mean, y_true_std, emp_zero_rate_y


def calculate_pointwise_crps(y_true, samples, mask=None, batch_size=1024):
    """
    The Core Engine: Calculates CRPS for each time series individually on GPU.

    Args:
        y_true (np.array): Shape (N, H)
        samples (np.array): Shape (N, H, n_samples)
        mask (np.array): Shape (N, H)
        batch_size (int): Processes time-series in chunks to save GPU memory.
        
    Returns:
        (np.array): Array of shape (N_series,) containing CRPS per series.
    """
    # --- 1. Check Prediction Existence ---
    if samples.shape[0] == 0:
        return np.array([])

    if mask is None:
        mask = np.ones_like(y_true)
        
    n_series = y_true.shape[0]
    pointwise_crps = []
    
    # --- 2. Process in Batches to save GPU memory ---
    for i in range(0, n_series, batch_size):
        end = min(i + batch_size, n_series)
        
        # Transfer to GPU
        y_batch = cp.asarray(y_true[i:end])        # (B, H)
        samples_batch = cp.asarray(samples[i:end]) # (B, H, S)
        mask_batch = cp.asarray(mask[i:end])       # (B, H)
        
        # Expand y: (B, H, 1)
        y_expanded = y_batch[..., None]
        
        # --- 2.1 CRPS Calculation ---
        # Term 1: E|X - y|
        term1 = cp.abs(samples_batch - y_expanded).mean(axis=-1) 
        
        # Term 2: 0.5 * E|X - X'|
        term2 = 0.5 * cp.abs(
            samples_batch[..., None] - samples_batch[..., None, :]
        ).mean(axis=(-1, -2)) 
        
        batch_crps_t = term1 - term2 # (B, H)
        
        # --- 2.2 Reduction over Time Axis ---
        sum_mask = cp.sum(mask_batch, axis=1)
        series_crps = cp.sum(batch_crps_t * mask_batch, axis=1) / (sum_mask + 1e-8)
        series_crps = cp.where(sum_mask == 0, cp.nan, series_crps)
        
        pointwise_crps.append(cp.asnumpy(series_crps))
        
        # Cleanup
        del y_batch, samples_batch, mask_batch, term1, term2, batch_crps_t
        cp.get_default_memory_pool().free_all_blocks()

    return np.concatenate(pointwise_crps)


def calculate_crps(y_true, samples, mask=None, batch_size=1024):
    """
    Wrapper: Calculates the Mean CRPS across all series.
    Uses calculate_pointwise_crps as the backend.
    
    Args:
        y_true (np.array): Shape (N, H)
        samples (np.array): Shape (N, H, n_samples)
        mask (np.array): Shape (N, H)
        batch_size (int): Processes time-series in chunks to save GPU memory.
        
    Returns:
        float: Scalar CRPS value.
    """
    # --- 1. Get the vector of CRPS values ---
    pointwise_scores = calculate_pointwise_crps(y_true, samples, mask, batch_size)
    
    # --- 2. Handle Edge Cases (Empty predictions) ---
    if len(pointwise_scores) == 0:
        return None
        
    # --- 3. Aggregate ---
    mean_crps = np.nanmean(pointwise_scores) # ignores series that were fully masked
    
    return float(mean_crps)


def evaluate_fold(fold, all_test, all_samples, point_base_preds):
    """
    Evaluates the fold predictions across all models.

    Args:
        fold (str): Current fold being processed.
        all_test (dict): Dictionary containing y test true values and mask.
        all_samples (dict): Dictionary containing the predictions for each model for the fold and across Volatility labels.
        point_base_preds (dict): Dictionary containing the point predictions for the baselines.

    Returns:
        metrics_dict (dict): Dictionary containing all calculated metrices for a fold.
    """
    # Define save path
    metrics_path = FORECASTS_DIR / f"all_metrics_{fold}.json"
    
    # Check if exists
    if metrics_path.exists():
        logger.info('Found saved evaluation. Loading...')
        with open(metrics_path, 'rb') as f:
            data = pickle.load(f)
        return data['metrics']
    
    logger.info(f"Evaluating predictions for {fold}...")

    # --- 1. Define function to recursively evaluate ---
    def _recursive_eval(all_test, all_samples, point_base_preds, metrics_dict):
        """Loops through dictionary structure recursively to evaluate each model."""
        if 'y_test' in all_test:
            # --- 1. Leaf Node (Evaluate samples) ---
            y_true = all_test['y_test']
            y_mask = all_test['mask_test']

            for model, samples in all_samples.items():
                logger.info("\n-------------------------------------------------------------------------")
                logger.info(f"Evaluating {model} predictions...")
                
                # Ensure baseline utilize point predictions
                if model in ['SARIMAX', 'ETSX', 'Naive']:
                    p50_predictions = point_base_preds[model].round()
                    mean_predictions = point_base_preds[model].round()
                else:
                    p50_predictions = np.percentile(samples, 50, axis=2)
                    mean_predictions = np.mean(samples, axis=2).round()
                
                metrics_dict[model] = {}
                rmse = calculate_rmse(mean_predictions, y_true, y_mask)
                metrics_dict[model]['rmse'] = rmse
                wape = calculate_wape(p50_predictions, y_true, y_mask)
                metrics_dict[model]['wape'] = wape
                crps = calculate_crps(y_true, samples, y_mask)
                metrics_dict[model]['crps'] = crps
                if rmse is not None and wape is not None and crps is not None:
                    logger.info(f"--- Test Set RMSE (Mean vs y_true): {rmse:.4f} ---")
                    logger.info(f"--- Test Set WAPE (p50 vs y_true): {wape:.4f} ---")
                    logger.info(f"--- Test Set CRPS: {crps:.4f} ---")
                    avg_mean, avg_std, emp_zero_rate_p50, y_true_mean, y_true_std, emp_zero_rate_y = compare_statistics(mean_predictions, p50_predictions, y_true, y_mask)
                    logger.info(f"    -> P50 Predictions: Mean={avg_mean:.4f}, Std={avg_std:.4f}, Empirical zero rate={emp_zero_rate_p50:.4f}")
                    logger.info(f"    -> Y_True Values:   Mean={y_true_mean:.4f}, Std={y_true_std:.4f}, Empirical zero rate={emp_zero_rate_y:.4f}")
        else:
            # --- 2. Branch Node (Recursive find)
            for key in all_test.keys():
                logger.info(f"Processing group: {key}...")
                next_base = point_base_preds[key] if (point_base_preds and key in point_base_preds) else {}

                metrics_dict[key] = {}
                _recursive_eval(
                    all_test[key],
                    all_samples[key],
                    next_base,
                    metrics_dict[key]
                )
    
    metrics_dict = {}

    # --- 2. Evaluate the predictions ---
    _recursive_eval(all_test, all_samples, point_base_preds, metrics_dict)

    # --- 3. Save the predictions ---
    logger.info(f"Saving predictions to {metrics_path}...")
    save_data = {
        'metrics': metrics_dict
    }

    with open(metrics_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=4)
    
    logger.debug(f"--- Successfully saved evaluations for {fold} ---")

    return metrics_dict
            





if __name__ == "__main__":
    pass
