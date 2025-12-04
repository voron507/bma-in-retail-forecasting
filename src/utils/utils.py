import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging
import config
import json



# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration ---
# Directories
MODELS = ["DeepAR", "TFT", "DeepTCN"]
FORECASTS_DIR = config.FORECASTS_DIR
METRICS_DIR = config.METRICS_DIR
# Columns
TARGET = config.TARGET
IS_ACTIVE = config.IS_ACTIVE
# Parameters
HORIZON = config.HORIZON


def _sort_data_by_index(data, index):
    """
    Sorts data and index arrays based on 'StockCode' and 'time_idx'.
    
    Args:
        data (np.ndarray): The data to sort. Can be shape (N_total, ...) or (N_series, Horizon, ...).
        index (np.recarray): The index array. Shape (N_total,).
    
    Returns:
        tuple: (sorted_data, sorted_index)
    """
    # --- 1. Capture original shape to restore it later ---
    original_shape = data.shape
    n_total = index.shape[0]
    
    # --- 2. Flatten data to (N_total, ...) ---
    if data.shape[0] != n_total:
        data_flat = data.reshape(n_total, *original_shape[2:])
    else:
        data_flat = data

    # --- 3. Determine the sort order ---
    sort_indices = np.argsort(index, order=['StockCode', 'time_idx'])
    
    # --- 4. Apply sorting ---
    sorted_index = index[sort_indices]
    sorted_data_flat = data_flat[sort_indices]
    
    # --- 5. Restore original shape structure ---
    sorted_data = sorted_data_flat.reshape(original_shape)
    
    return sorted_data, sorted_index


def _load_data(fold, splits):
    """
    Function to load targets, masks, indexes, parameters, and samples.

    Args:
        fold (str): Name of the fold.
        splits (dict): Contains dataset across folds.
    
    Returns:
        y_val (np.array), y_val_mask (np.array), val_index_array (np.array), 
        y_test (np.array), y_test_mask(np.array), test_index_array (np.array), 
        params (dict), samples (dict), samples_base (dict), point_preds_base(dict): Tuple containing targets, masks, indexes, parameters, and samples.
    """
    logger.info("Loading validation targets, model parameters, test targets and test samples...")
    # --- 1. Define paths ---
    dl_pred_paths = {
    "DeepAR": FORECASTS_DIR / 'DeepAR' / f'test_predictions_deepar_{fold}.npz',
    "TFT": FORECASTS_DIR / 'TFT' / f'test_predictions_tft_{fold}.npz',
    "DeepTCN": FORECASTS_DIR / 'DeepTCN' / f'test_predictions_deeptcn_{fold}.npz'
    }

    base_pred_paths = {
    "SARIMAX": FORECASTS_DIR / 'Baselines' / f'test_predictions_sarimax_{fold}.npz',
    "ETSX": FORECASTS_DIR / 'Baselines' / f'test_predictions_etsx_{fold}.npz',
    "Naive": FORECASTS_DIR / 'Baselines' / f'test_predictions_naive_{fold}.npz'
    }

    # --- 2. Check if exist ---
    for model, path in dl_pred_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{model} doesn't have saved predictions {path}, run the full pipeline in appropriate environment first.")
    
    # --- 3. Extract validation targets ---
    val_df = splits[fold]['validation'].sort_values(['StockCode', 'time_idx'])
    n_series = val_df['StockCode'].nunique()

    # Val Index
    val_index_df = val_df[['StockCode', 'time_idx', 'VolatilityLabel', 'ProductStatus']].reset_index(drop=True)
    val_index_array = val_index_df.to_records(index=False) # (N_series * HORIZON, )

    # Val Target
    y_val = val_df[TARGET].to_numpy().reshape(n_series, HORIZON)
    y_val, val_index_array = _sort_data_by_index(y_val, val_index_array) # For safety

    y_val_mask = val_df[IS_ACTIVE].to_numpy().reshape(n_series, HORIZON)
    y_val_mask, _ = _sort_data_by_index(y_val_mask, val_index_array) # For safety

    # --- 4. Extract test targets ---
    test_df = splits[fold]['test'].sort_values(['StockCode', 'time_idx'])
    n_series = test_df['StockCode'].nunique()

    # Test Index
    test_index_df = test_df[['StockCode', 'time_idx', 'VolatilityLabel', 'ProductStatus']].reset_index(drop=True)
    test_index_array = test_index_df.to_records(index=False) # (N_series * HORIZON, )

    # Test Target
    y_test = test_df[TARGET].to_numpy().reshape(n_series, HORIZON)
    y_test, test_index_array = _sort_data_by_index(y_test, test_index_array)

    y_test_mask = test_df[IS_ACTIVE].to_numpy().reshape(n_series, HORIZON)
    y_test_mask, _ = _sort_data_by_index(y_test_mask, test_index_array)
    
    # --- 5. Extract params and samples ---
    params = {}
    samples = {}
    point_preds_base = {}
    samples_base = {}
    
    for model in dl_pred_paths:
        # Load model data
        p_raw = np.load(dl_pred_paths[model], allow_pickle=True)['val_params']
        idx_val_model = np.load(dl_pred_paths[model], allow_pickle=True)['val_index']
        s_raw = np.load(dl_pred_paths[model], allow_pickle=True)['predictions']
        idx_test_model = np.load(dl_pred_paths[model], allow_pickle=True)['test_index']

        # Sort by index
        params[model], _ = _sort_data_by_index(p_raw, idx_val_model) # {Model: Params (N, H, 5, 3)}
        samples[model], _ = _sort_data_by_index(s_raw, idx_test_model) # {Model: Samples (N, H, 200)}

    for model in base_pred_paths:
        p_base_raw = np.load(base_pred_paths[model], allow_pickle=True)['point_forecasts']
        s_base_raw = np.load(base_pred_paths[model], allow_pickle=True)['samples']
        idx_test_model = np.load(base_pred_paths[model], allow_pickle=True)['test_index']

        # Sort by index
        point_preds_base[model], _ = _sort_data_by_index(p_base_raw, idx_test_model) # {Model: Preds (N, H)}
        samples_base[model], _ = _sort_data_by_index(s_base_raw, idx_test_model) # {Model: Samples (N, H, 200)}
    
    logger.info("--- All data is successfuly loaded and sorted ---")
        
    return y_val, y_val_mask, val_index_array, y_test, y_test_mask, test_index_array, params, samples, samples_base, point_preds_base


def _split_dicts_by_segments(data_dict, index):
    """
    Splits a dictionary of arrays into subsets based on 'ProductStatus' and 'VolatilityLabel'.
    
    Structure:
    {
        'Cold-Start': {model: data},
        'Established Full': {model: data},
        'Established': {
            'High Volatility': {model: data},
            'Moderate Volatility': {model: data},
            'Low Volatility': {model: data}
        }
    }

    Args:
        data_dict (dict): Dictionary with data of shape (N_series, Horizon, ...).
        index (np.recarray): The aligned index array of shape (N_series * Horizon, ...).

    Returns:
        dict: Hierarchical dictionary containing the split data.
    """
    # --- 1. Setup and Validation --- 
    first_model = next(iter(data_dict))
    sample_data = data_dict[first_model]
    
    n_series_data = sample_data.shape[0] # Number of series in the data provided
    horizon = sample_data.shape[1]
    n_total_points = index.shape[0]      # Total points in the provided index
    
    # Calculate implicit number of series in the Index
    n_series_index = n_total_points // horizon

    # --- 2. Handle Baseline Mismatch (Data < Index) ---
    if n_series_data * horizon != n_total_points:
        
        # Reshape index to check metadata
        try:
            index_reshaped_full = index.reshape(n_series_index, horizon)
        except ValueError:
             raise ValueError(f"Index size {n_total_points} is not divisible by horizon {horizon}.")

        # Check counts
        status_full = index_reshaped_full[:, 0]['ProductStatus']
        n_established = np.sum(status_full == 'Established')
        
        if n_series_data == n_established:
            logger.info(f"Shape mismatch detected ({n_series_data} data vs {n_series_index} index). "
                        f"Data matches exactly 'Established' count. Filtering index...")
            
            # Create mask for Established items on the full index
            mask_established_flat = np.repeat((status_full == 'Established'), horizon)
            
            # Filter the index to match the data
            index = index[mask_established_flat]
            
            # Update dimensions for the rest of the function
            n_total_points = index.shape[0]
            n_series_index = n_established
            
        else:
            # It's a real shape error, not just a missing Cold-Start issue
            raise ValueError(f"Shape mismatch: Index has {n_series_index} series, Data has {n_series_data}. "
                             f"Even filtering for Established ({n_established}) does not match.")

    # --- 3. Extract Labels per Series ---
    # Reshape aligned index to (N_series, Horizon)
    index_reshaped = index.reshape(n_series_index, horizon)
    series_status = index_reshaped[:, 0]['ProductStatus']
    series_volatility = index_reshaped[:, 0]['VolatilityLabel']

    # --- 4. Define Helper for Masking ---
    def _subset_dict(d, mask):
        """Filters all arrays in the dictionary using the boolean mask."""
        subset = {}
        for k, v in d.items():
            if v.shape[0] != len(mask):
                 raise ValueError(f"Model {k} data shape {v.shape[0]} mismatch with mask length {len(mask)}.")
            subset[k] = v[mask]
        return subset

    # --- 4. Construct the Hierarchy ---
    split_results = {
        'Cold-Start': {},
        'Established Full': {},
        'Established Segments': {}
    }
    
    logger.info("Splitting data into ProductStatus hierarchy...")

    # --- A. Handle Cold-Start ---
    # 'Unknown' volatility is implicitly covered
    mask_cold_start = (series_status == 'Cold-Start')
    split_results['Cold-Start'] = _subset_dict(data_dict, mask_cold_start)
    logger.info(f"  -> 'Cold-Start': {np.sum(mask_cold_start)} series")

    # --- B. Handle Established Full ---
    mask_established = (series_status == 'Established')
    split_results['Established Full'] = _subset_dict(data_dict, mask_established)
    logger.info(f"  -> 'Established Full': {np.sum(mask_established)} series")

    # --- C. Handle Established Splits (High/Moderate/Low) ---
    target_volatilities = ['High Volatility', 'Moderate Volatility', 'Low Volatility']
    
    for vol_label in target_volatilities:
        # Create combined mask
        mask_sub = mask_established & (series_volatility == vol_label)  
        split_results['Established Segments'][vol_label] = _subset_dict(data_dict, mask_sub)
        logger.info(f"     -> Established Segments / '{vol_label}': {np.sum(mask_sub)} series")
            
    return split_results


def _calculate_segment_costs(test_split_dict, fold):
    """
    Calculates and distributes computational costs (Time & CO2) across all segments and models.
    Loads cost files, calculates prediction volume (based on sum of masks) for each segment, 
    distributes costs proportional to volume and aggregates DL costs to form BMA costs.
    
    Args:
        test_split_dict (dict): The hierarchical dictionary containing 'mask_test' for segmentation.
        fold (str): Current fold name.
        
    Returns:
        dict: Nested dictionary with costs per segment per model. {{Segment}: {Model: {'time': X, 'co2': Y, 'volume': Z}}}
    """
    logger.info(f"Calculating cost distribution for {fold}...")
    
    # --- 1. Define path to cost files ---
    dl_metrics_paths = {
        "DeepAR": METRICS_DIR / 'DeepAR' / f'deepar_costs_{fold}.json',
        "TFT": METRICS_DIR / 'TFT' / f'tft_costs_{fold}.json',
        "DeepTCN": METRICS_DIR / 'DeepTCN' / f'deeptcn_costs_{fold}.json'
    }

    base_metrics_paths = {
        "SARIMAX": METRICS_DIR / 'Baselines' / f'baseline_costs_sarimax_{fold}.json',
        "ETSX": METRICS_DIR / 'Baselines' / f'baseline_costs_etsx_{fold}.json',
        "Naive": METRICS_DIR / 'Baselines' / f'baseline_costs_naive_{fold}.json'
    }

    tune_metrics_paths = {
        "DeepAR": METRICS_DIR / 'DeepAR' / f"deepar_tuning_{fold}.json",
        "TFT": METRICS_DIR / 'TFT' / f"tft_tuning_{fold}.json",
        "DeepTCN": METRICS_DIR / 'DeepTCN' / f"deeptcn_tuning_{fold}.json"
    }
    
    # --- 2. Load Raw Total Costs ---
    raw_costs = {}
    tune_costs = {}
    skip_tuning_costs = False
    
    # Load DL Training Costs
    for model, path in dl_metrics_paths.items():
        with open(path, 'r') as f:
            data = json.load(f)
        if data:
            raw_costs[model] = {
                'time': data['total_time_seconds'],
                'co2': data['co2_emissions_kg']
            }
            
    # Load Baseline Costs
    for model, path in base_metrics_paths.items():
        with open(path, 'r') as f:
            data = json.load(f)
        if data:
            raw_costs[model] = {
                'time': data['total_time_seconds'],
                'co2': data['co2_emissions_kg']
            }

    # Load DL Tuning Costs
    try:
        for model, path in tune_metrics_paths.items():
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                tune_costs[model] = {
                    'time': data['total_time_seconds'],
                    'co2': data['co2_emissions_kg']
                }
            else:
                pass
        if not tune_costs:
            raise ValueError("No tuning files found")
    except:
        logger.info(f'No tuning costs found for {fold}. Filling structure with NaN.')
        skip_tuning_costs = True
            
    # --- 2. Calculate Reference Volumes (Denominators) ---
    # Find volume based on masks for cold start
    test_split_cold = test_split_dict['Cold-Start']
    vol_cold_start = float(np.sum(test_split_cold['mask_test']))
    
    # Find volume based on masks for established
    test_split_test = test_split_dict['Established Full']
    vol_established_full = float(np.sum(test_split_test['mask_test']))
        
    # Denominators
    total_dl_volume = vol_cold_start + vol_established_full
    total_baseline_volume = vol_established_full # Baselines ignore Cold-Start

    # --- 3. Recursive Cost Distribution ---
    def _process_node(node, is_cold_start_branch=False):
        """
        Recursively traverses the dictionary. 
        If Leaf: Calculates costs.
        If Branch: Recurses deeper.
        """
        # --- 3.1 Base Case: Leaf Node (Contains Data) ---
        if 'mask_test' in node:
            mask = node['mask_test']
            volume = float(np.sum(mask))
            
            node_costs = {}
            tune_node_costs = {}
            
            # --- 3.1.1 DL Models (Fraction of Global Total) ---
            dl_fraction = volume / total_dl_volume
            
            bma_train_t, bma_train_c = 0, 0
            bma_tune_t, bma_tune_c = 0, 0
            
            for model in ["DeepAR", "TFT", "DeepTCN"]:
                t = raw_costs[model]['time'] * dl_fraction
                c = raw_costs[model]['co2'] * dl_fraction
                
                node_costs[model] = {
                    'time': round(t, 2),
                    'co2': round(c, 6),
                    'volume': int(volume)
                }
                
                # Aggregate BMA
                bma_train_t += t
                bma_train_c += c

                if not skip_tuning_costs and model in tune_costs:
                    t_tune = tune_costs[model]['time'] * dl_fraction
                    c_tune = tune_costs[model]['co2'] * dl_fraction

                    tune_node_costs[model] = {
                    'time': round(t_tune, 2),
                    'co2': round(c_tune, 6),
                    'volume': int(volume)
                    }
                    
                    bma_tune_t += t_tune
                    bma_tune_c += c_tune
                else:
                    tune_node_costs[model] = {'time': np.nan, 'co2': np.nan, 'volume': int(volume)}
            
            # Add BMA
            node_costs["BMA"] = {
                'time': round(bma_train_t, 2),
                'co2': round(bma_train_c, 6),
                'volume': int(volume)
            }

            if not skip_tuning_costs:
                tune_node_costs["BMA"] = {
                'time': round(bma_tune_t, 2),
                'co2': round(bma_tune_c, 6),
                'volume': int(volume)
                }    
            else:
                tune_node_costs["BMA"] = {'time': np.nan, 'co2': np.nan, 'volume': int(volume)}
            
            # --- 3.1.2 Baseline Models (Fraction of Established Total) ---
            if is_cold_start_branch:
                base_fraction = 0.0
            else:
                base_fraction = volume / total_baseline_volume
                
            for model in ["SARIMAX", "ETSX", "Naive"]:
                if model in raw_costs:
                    # If cold start, cost is explicitly 0
                    t = 0.0 if is_cold_start_branch else raw_costs[model]['time'] * base_fraction
                    c = 0.0 if is_cold_start_branch else raw_costs[model]['co2'] * base_fraction
                    
                    node_costs[model] = {
                        'time': round(t, 2),
                        'co2': round(c, 6),
                        'volume': int(volume)
                    }
                    val_tune = np.nan if skip_tuning_costs else 0.0
                    tune_node_costs[model] = {'time': val_tune, 'co2': val_tune, 'volume': int(volume)}
            
            return node_costs, tune_node_costs

        # --- 3.2 Recursive Step: Branch Node ---
        train_branch_costs = {} # Holds the existing structure
        tune_branch_costs = {}
        for key, value in node.items():
            # Flag if we are entering the Cold-Start branch
            new_flag = is_cold_start_branch or (key == 'Cold-Start')
            
            t_branch, tu_branch = _process_node(value, is_cold_start_branch=new_flag)
            train_branch_costs[key] = t_branch
            tune_branch_costs[key] = tu_branch

        return train_branch_costs, tune_branch_costs
    
    return _process_node(test_split_dict)


def _recursive_merge(dict_a, dict_b):
    """
    Recursively merges dict_b into dict_a.
    If keys match and values are dicts, it merges them.
    If keys match and values are NOT dicts (e.g. arrays), dict_b overwrites dict_a.
    If keys don't match, it adds the new key.

    Args:
        dict_a (dict): Dictionary to merge in.
        dict_b (dict): Dictionary to be merged in.

    Returns:
        dict_a (dict): Merged dictionary.
    """
    for key, value in dict_b.items():
        if key in dict_a and isinstance(dict_a[key], dict) and isinstance(value, dict):
            # If both are dictionaries, go deeper (Recurse)
            _recursive_merge(dict_a[key], value)
        else:
            # If leaf node (data), add/overwrite the key
            dict_a[key] = value
    return dict_a


def _inject_ll(metric_node, ll_node):
                for k, v in ll_node.items():
                    if isinstance(v, dict) and k in metric_node: 
                        # If we are at Branch level, recurse
                        _inject_ll(metric_node[k], v)
                    elif not isinstance(v, dict):
                        # We are at Leaf level: k is Model Name, v is Float
                        if k in metric_node:
                            metric_node[k]['val_log_lik'] = v


def prepare_preds_bma(fold, splits):
    """
    Full pipeline to prepare data for BMA and evaluation after model training.

    Args:
        fold (str): Fold to be proccessed.
        splits (dict): Dictionary containing fold data.

    Returns:
        val_split(dict), params_split(dict), test_split(dict), samples_split(dict), samples_base_split (dict), point_preds_base_split (dict)
    """
    # --- 1. Load and Align Order ---
    y_val, y_val_mask, val_idx, y_test, y_test_mask, test_idx, params, samples, samples_base, point_preds_base = _load_data(fold, splits)
    
    # --- 2. Create full dictionaries ---
    val_full = {'y_val': y_val, 'mask_val': y_val_mask}
    test_full = {'y_test': y_test, 'mask_test': y_test_mask}

    # --- 3. Split Validation ---
    logger.info("Spliting data by volatility labels for validation...")
    # {'Established Full' / 'Cold-Start': {'y_val'/'mask_val': data (N(sub), H)}, 'Established Segments': {VolLabel: {'y_val'/'mask_val': data (N(sub), H)}}}
    val_split = _split_dicts_by_segments(val_full, val_idx)
    # {'Established Full' / 'Cold-Start': {VolLabel: {Model: params (N(sub), H, 5, 3)}, 'Established Segments': {VolLabel: {Model: params (N(sub), H, 5, 3)}}}
    params_split = _split_dicts_by_segments(params, val_idx) 
    
    # --- 4. Split Test ---
    logger.info("Spliting data by volatility labels for test...")
    # {'Established Full' / 'Cold-Start': {'y_test'/'mask_test': data (N(sub), H)}, 'Established Segments': {VolLabel: {'y_test'/'mask_test': data (N(sub), H)}}}
    test_split = _split_dicts_by_segments(test_full, test_idx)
    # {'Established Full' / 'Cold-Start': {Model: samples (N(sub), H, 200)}, 'Established Segments': {VolLabel: {Model: samples (N(sub), H, 200)}}}
    samples_split = _split_dicts_by_segments(samples, test_idx)

    # Split base predictions
    # {'Established Full': {Model: samples (N(sub), H, 200)}, 'Established Segments': {VolLabel: {Model: samples (N(sub), H, 200)}}}
    samples_base_split = _split_dicts_by_segments(samples_base, test_idx)
    # {'Established Full': {Model: point-preds (N(sub), H)}, 'Established Segments': {VolLabel: {Model: point-preds (N(sub), H)}}}
    point_preds_base_split = _split_dicts_by_segments(point_preds_base, test_idx)

    return val_split, params_split, test_split, samples_split, samples_base_split, point_preds_base_split


def prepare_preds_eval(fold, test_split, samples_split, bma_preds, samples_base_split):
    """
    Prepares data for the final evaluation.

    Args:
        fold (str): Current fold to process.
        test_split (dict): Contains data on test true values and masks split across Product Status and Volatility segments. {...(N(sub), H)}
        samples_split (dict): Contains test samples split across Product Status and Volatility segments for each DL model. {...(N(sub), H, 200)}
        bma_preds (dict): Contains BMA samples split across Product Status and Volatility segments.. {...(N(sub), H, 200)}
        samples_base_split (dict): Contains Base samples split across Product Status and Volatility segments. {...(N(sub), H, 200)}

    Returns:
        all_test (dict), all_samples (dict): Test data, and samples across Product Status and Volatility segments.
    """
    logger.info("Preparing data for evaluation...")
    # --- 1. Unite the preds ---
    all_dl_samples = _recursive_merge(samples_split, bma_preds)
    all_samples = _recursive_merge(all_dl_samples, samples_base_split)

    # --- 2. Derive costs for each group ---
    train_costs, tune_costs = _calculate_segment_costs(test_split, fold)

    # --- 3. Save the costs ---
    costs_path = FORECASTS_DIR / f"all_costs_{fold}.json"
    logger.info(f"Saving ccosts to {costs_path}...")
    save_data = {
        'train_costs': train_costs,
        'tune_costs': tune_costs
    }

    with open(costs_path, 'w') as f:
        json.dump(save_data, f, indent=4)

    return all_samples, train_costs, tune_costs
    
    
if __name__ == "__main__":
    pass 
