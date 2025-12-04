import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import pickle
import logging
import config
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize

from utils.evaluate_preds import calculate_rmse, calculate_wape, compare_statistics



# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration ---
# Directories
MODELS = ["DeepAR", "TFT", "DeepTCN"]
RESULTS_DIR = config.RESULTS_DIR
FORECASTS_DIR = config.FORECASTS_DIR
# Columns
TARGET = config.TARGET
IS_ACTIVE = config.IS_ACTIVE
# Parameters
HORIZON = config.HORIZON
N_SAMPLES = config.N_SAMPLES

def _calculate_pointwise_log_likelihood(y_val, mask_val, parameters, softZero=1e-6):
    """
    Calculates the pointwise Log-Probability of the true data given 5 ensemble seeds.
    
    Args:
        y_val (np.array): True target values. Shape (N, H)
        mask_val (np.array): Mask for valid values. Shape (N, H)
        parameters (np.array): Parameters (Mu, Alpha, Gate). Shape (N, H, n_seeds, 3)
        softZero (float): Epsilon for numerical stability.
    
    Returns: 
        valid_log_lik (np.array): Flattened array of valid log-likelihoods.
    """
    # Temporal fix for DeepTCN bug
    if parameters.ndim == 5 and parameters.shape[-2] == 1:
        parameters = parameters.squeeze(axis=-2)
    
    # --- 1. Ensure Inputs are Numpy Arrays & Handle Dimensions ---
    # Expand y to (N, H, 1)
    y = np.array(y_val, dtype=np.float64)[..., None]
    mask = np.array(mask_val, dtype=np.float64) # Keep mask as (N, H) for final indexing
    
    # Extract to (N, H, n_seeds)
    mu = np.array(parameters[..., 0], dtype=np.float64)
    alpha = np.array(parameters[..., 1], dtype=np.float64)
    gate = np.array(parameters[..., 2], dtype=np.float64)

    # --- 2. ZINB Parameters (Computed per seed) ---
    r = 1.0 / (alpha + softZero)
    theta = 1.0 / (1.0 + alpha * mu)
    
    log_theta = np.log(theta + softZero)
    log_1m_theta = np.log(1.0 - theta + softZero)

    # Ensure y >= 0
    y = np.maximum(y, 0.0)

    # --- 3. NB log pmf for general y ---
    p1 = gammaln(y + r + softZero)
    p2 = gammaln(r + softZero)
    p3 = gammaln(y + 1.0)
    p4 = r * log_theta
    p5 = y * log_1m_theta
    
    log_nb_y = p1 - p2 - p3 + p4 + p5

    # --- 4. NB log pmf for y = 0 ---
    # y0 = np.zeros_like(y)
    # p1_0 = gammaln(y0 + r + softZero)
    log_nb_0 = p4  # p1_0 - p2 + p4, where p1_0 = p2 -> p2 - p2 + p4 = p4

    # --- 5. Zero-Inflation Mixture Logic ---
    is_zero = (y == 0)

    # Case A: y = 0
    prob_zero_mixture = gate + (1.0 - gate) * np.exp(log_nb_0)
    log_prob_zero = np.log(prob_zero_mixture + softZero)

    # Case B: y > 0
    log_prob_nonzero = np.log(1.0 - gate + softZero) + log_nb_y

    # Combine: result is (N, H, n_seeds)
    log_prob_seeds = np.where(is_zero, log_prob_zero, log_prob_nonzero)

    # --- 6. Aggregate Seeds (Mixture) ---
    # Log Likelihood = LogSumExp(log_probs) - Log(N_seeds)
    n_seeds = parameters.shape[-2]
    log_prob_mixture = logsumexp(log_prob_seeds, axis=-1) - np.log(n_seeds) # (N, H, 5) -> (N, H)

    # --- 7. Masking and Flattening ---
    log_prob_flat = log_prob_mixture.flatten()
    mask_flat = mask.flatten()
    
    # Return only valid log-likelihoods where mask == 1
    valid_log_lik = log_prob_flat[mask_flat == 1]

    return valid_log_lik


def _get_bma_weights(y_val, mask_val, model_params):
    """
    Calculates Bayesian Stacking weights using Scipy Optimization.
    Maximizes sum(log(sum(w_k * P(y|M_k)))).

    Args:
        y_val (np.array): Target true values for the validation set.
        mask_val (np.array): Mask for the true target validation values.
        model_params (dict): Contains parameters for Mu, Alpha, Gate for each model.

    Returns: 
        weights (dict): Dictionary containing weights of each model.
    """
    logger.info("--- Calculating Log-Likelihoods for BMA ---")
    model_names = []
    log_lik_vectors = []
    val_ll_scores = {}
    
    for model_name, params in model_params.items():
        # --- 1. Calculate Log-Likelihood for the models ---
        log_lik = _calculate_pointwise_log_likelihood(y_val, mask_val, params)
        model_names.append(model_name)
        log_lik_vectors.append(log_lik)
        val_ll_scores[model_name] = float(np.mean(log_lik))
        logger.debug(f"{model_name} log-likelihood calculated (Size: {len(log_lik)}).")

    # Transpose so rows are samples, cols are models
    log_lik_matrix = np.stack(log_lik_vectors, axis=1) 
    n_models = len(model_names)
    
    # --- 2. Define Optimization Objective (Negative Log Score) ---
    def negative_log_score(weights):
        # log(sum(w * exp(LL))) = logsumexp(log(w) + LL)
        w_safe = np.clip(weights, 1e-10, 1.0) # to avoid log(0)
        log_w = np.log(w_safe)
        
        # Broadcast: (1, N_models) + (N_samples, N_models)
        weighted_ll = log_w + log_lik_matrix
        
        # Sum over models (axis 1) for each sample
        sample_scores = logsumexp(weighted_ll, axis=1)
        
        # Minimize Negative Sum
        return -np.sum(sample_scores)

    # --- 3. Constraints & Bounds ---
    # Constraint: Sum(weights) = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    # Bounds: 0 <= w <= 1
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    # Initial Guess: Equal weights
    initial_weights = np.ones(n_models) / n_models

    # --- 4. Optimize ---
    logger.debug("Running Stacking Optimization...")
    result = minimize(
        negative_log_score, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9}
    )
    if not result.success:
        logger.warning(f"BMA Optimization warning: {result.message}")

    # Normalize result
    final_weights = np.maximum(result.x, 0)
    final_weights /= np.sum(final_weights)

    # Create dictionary
    weights_dict = dict(zip(model_names, final_weights)) # {'DeepAR': 0.2, 'TFT': 0.7, ...}
    
    return weights_dict, val_ll_scores


def _ensemble_samples(weights, samples):
    """
    Combines samples from different models based on weights.

    Args:
        weights (dict): Dictionary containing weights for each model.
        samples (dict): Dictionary containing samples from each of the models.
        n_output_samples (int): Number of samples to use for evaluation

    Returns:
        final_ensemble (np.array): Array containing ensemble samples.
    """
    logger.info(f"--- Ensembling Test Samples (Total: {N_SAMPLES}) ---")
    
    # --- 1. Determine how many samples to take from each model ---
    counts = {}
    for model, w in weights.items():
        counts[model] = int(np.round(w * N_SAMPLES))
    
    # Fix rounding errors (ensure sum is exactly n_output_samples)
    current_sum = sum(counts.values())
    diff = N_SAMPLES - current_sum
    if diff != 0:
        # Add/subtract remainder from the best model
        best_model = max(weights, key=weights.get)
        counts[best_model] += diff
        
    logger.info(f"Sample Counts: {counts}")

    # --- 2. Load Test Samples and Mix ---
    ensembled_samples = []
    
    for model in MODELS:
        count = counts[model]
        if count == 0:
            continue
            
        # Load samples
        logger.info(f"Loading {model} test samples...")
        full_samples = samples[model] # (N, H, 200)
        
        # Randomly select 'count' samples from the 200 available
        indices = np.random.choice(full_samples.shape[2], count, replace=False)
        selected_samples = full_samples[:, :, indices]
        
        ensembled_samples.append(selected_samples)
    
    # --- 3. Concatenate along the sample axis ---
    final_ensemble = np.concatenate(ensembled_samples, axis=2) # (N_series, Horizon, 200)
    
    return final_ensemble


def _recursive_ensemble(val_dict, params_dict, test_dict, samples_dict, weights_dict, bma_preds, val_ll_dict):
    """
    Recursively traverses the nested data dictionary to calculate BMA weights and predictions.
    
    Args:
        val_dict, params_dict, test_dict, samples_dict: Current level dictionaries.
        weights (dict): Output dictionary to store weights.
        bma_preds (dict): Output dictionary to store BMA predictions.
        val_ll_dict (dict): Dictionary containing log likelihood for DL models.
    """
    
    # --- 1. Base Case: Check if this Node contains Data ---
    if 'y_val' in val_dict:
        # --- 1.1 Calculate Weights ---
        y_val = val_dict['y_val']
        mask_val = val_dict['mask_val']
        weights, ll_scores = _get_bma_weights(y_val, mask_val, params_dict)
        logger.debug(f"Weights calculated: {weights}")
        
        # Here stored directly
        weights_dict.update(weights)
        val_ll_dict.update(ll_scores)

        # --- 1.2 Ensemble Samples ---
        bma_samples = _ensemble_samples(weights, samples_dict)
        # Stored with key to integrate with other samples
        bma_preds['BMA'] = bma_samples
        logger.info(f"Ensemble Shape: {bma_samples.shape}")

        # --- 1.3 Assess for Debug ---
        y_test = test_dict['y_test']
        mask_test = test_dict['mask_test']
        p50_predictions = np.percentile(bma_samples, 50, axis=2)
        mean_predictions = np.mean(bma_samples, axis=2).round()
        rmse = calculate_rmse(mean_predictions, y_test, mask_test)
        logger.debug(f"--- Test Set RMSE (p50 vs y_true): {rmse:.4f} ---")
        wape = calculate_wape(p50_predictions, y_test, mask_test)
        logger.debug(f"--- Test Set WAPE (p50 vs y_true): {wape:.4f} ---")
        avg_mean, avg_std, emp_zero_rate_p50, y_true_mean, y_true_std, emp_zero_rate_y = compare_statistics(mean_predictions, p50_predictions, y_test, mask_test)
        logger.debug(f"    -> P50 Predictions: Mean={avg_mean:.4f}, Std={avg_std:.4f}, Empirical zero rate={emp_zero_rate_p50:.4f}")
        logger.debug(f"    -> Y_True Values:   Mean={y_true_mean:.4f}, Std={y_true_std:.4f}, Empirical zero rate={emp_zero_rate_y:.4f}")
        
        return

    # --- 2. Recursive Step: Iterate through keys (ProductStatus or VolatilityLabel) ---
    for key in val_dict.keys():
        logger.info(f"Processing group: {key}...")
        
        # Initialize sub-dictionaries in output to maintain structure
        weights_dict[key] = {}
        bma_preds[key] = {}
        val_ll_dict[key] = {}
        
        # Recursive Call
        _recursive_ensemble(
            val_dict[key], 
            params_dict[key], 
            test_dict[key], 
            samples_dict[key],
            weights_dict[key],
            bma_preds[key],
            val_ll_dict[key]
        )


def ensemble_on_fold(fold, val_split, params_split, test_split, samples_split):
    """
    Ensembles 3 DL models on a single fold.

    Args:
        fold (str): Fold to process.
        val_split (dict): Dictionary containing validation true values and masks split across Product Status and Volatility segments.
        params_split (dict): Dictionary containing validation parameters split across Product Status and Volatility segments.
        test_split (dict): Dictionary containing test true values and masks split across Product Status and Volatility segments.
        samples_split (dict): Dictionary containing test samples split across Product Status and Volatility segments for each model.

    Returns:
        weights_dict (dict), bma_preds (dict): Dictionaries with the weights assigned and predictions generated.
    """
    # --- 1. Create Directory ---
    pred_dir = FORECASTS_DIR / "BMA"
    pred_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = pred_dir / f"test_predictions_bma_{fold}.pkl"

    if predictions_path.exists():
        logger.info("Found saved predictions for BMA. Loading...")
        with open(predictions_path, 'rb') as f:
            data = pickle.load(f)
        return data['weights'], data['predictions'], data['log_lik']
    
    weights_dict = {}
    bma_preds = {}
    val_ll_dict = {}

    # --- 2. Ensemble predictions ---
    _recursive_ensemble(val_split, params_split, test_split, samples_split, weights_dict, bma_preds, val_ll_dict)

    # --- 3. Save all predictions ---
    logger.info(f"Saving predictions to {predictions_path}...")
    save_data_pkl = {
        'weights': weights_dict,
        'predictions': bma_preds,
        'log_lik': val_ll_dict
    }

    with open(predictions_path, 'wb') as f:
        pickle.dump(save_data_pkl, f, protocol=4)
    
    logger.debug(f"--- Successfully saved predictions for {fold} ---")
    return weights_dict, bma_preds, val_ll_dict

if __name__ == "__main__":
    pass

