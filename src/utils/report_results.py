import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import config
from utils.evaluate_preds import calculate_pointwise_crps


# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Define global variables ---
FORECASTS_DIR = config.FORECASTS_DIR
RESULTS_DIR = config.RESULTS_DIR
MODEL_ORDER = ["DeepAR", "TFT", "DeepTCN", "BMA", "SARIMAX", "ETSX", "Naive"]


def _recursive_concat(list_of_dicts):
    """
    Helper: Stacks arrays from a list of nested dictionaries.
    Used to merge [Fold1_Dict, Fold2_Dict] -> Aggregated_Dict
    """
    if not list_of_dicts: return None
    first = list_of_dicts[0]

    # Base Case: Array -> Concatenate
    if isinstance(first, (np.ndarray)):
        return np.concatenate(list_of_dicts, axis=0)

    # Recursive Case: Dictionary -> Recurse on Keys
    if isinstance(first, dict):
        agg = {}
        for k in first.keys():
            # Collects key from all dicts in the list
            vals = [d[k] for d in list_of_dicts if k in d]
            agg[k] = _recursive_concat(vals)
        return agg
    return None


def _recursive_flatten_metrics(d, parent_key='', sep='/'):
    """
    Flattens a nested dictionary of metrics into a list of records for further reports.
    Example: {'Established': {'DeepAR': {'rmse': 10}}} 
    Becomes: [{'Segment': 'Established', 'Model': 'DeepAR', 'Metric': 'rmse', 'Value': 10}]

    Args:
        d (dict): Dictionary to flatten.
        parent_key (str): String to save hierarchical order.
        sep (str): String to separate hierarchical order.

    Returns:
        (dict): Dictionary containing data for pivoting.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # If value is a dictionary - keep recursing
            items.extend(_recursive_flatten_metrics(v, new_key, sep=sep))
        else:
            # Path looks like: "Established/High Volatility/DeepAR/rmse"
            parts = new_key.split(sep)
            
            # This assumes depth: Segment -> [SubSegment] -> Model -> Metric
            metric = parts[-1]
            model = parts[-2]
            segment = " / ".join(parts[:-2]) # Join everything before Model as Segment
            
            items.append({
                'Segment': segment,
                'Model': model,
                'Metric': metric,
                'Value': v
            })
    return items


def _recursive_flatten_weight(d, parent_key='', sep='/'):
    """
    Flattens a BMA weights dictionary.
    Structure: Segment -> [SubSegment] -> Model -> Weight (float)

    Args:
        d (dict): Dictionary to flatten.
        parent_key (str): String to save hierarchical order.
        sep (str): String to separate hierarchical order.

    Returns:
        (dict): Dictionary containing data for pivoting.
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            # If value is a dictionary - keep recursing
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(_recursive_flatten_weight(v, new_key, sep=sep))
        else:
            # If value is not a dictionary - it's a float
            segment = parent_key
            model = k
            weight = v
            
            items.append({
                'Segment': segment,
                'Model': model,
                'Weight': weight
            })
    return items


def _apply_custom_sort(df, segment_col='Display_Segment'):
    """
    Applies strict sorting to Rows:
    1. Segments (Total -> High -> Med -> Low -> Cold)
    2. Models (DeepAR -> TFT -> DeepTCN -> BMA -> Baselines)
    """
    df = df.copy()
    
    # --- 1. Define Segment Order ---
    sort_order = {
        'Established (Total)': 0,
        '   L High Volatility': 1,
        '   L Moderate Volatility': 2,
        '   L Medium Volatility': 2, 
        '   L Low Volatility': 3,
        'Cold-Start': 4
    }
    df['sort_key'] = df[segment_col].map(lambda x: sort_order.get(x, 99))
    
    # --- 2. Define Model Order ---
    sort_cols = ['sort_key']
    if 'Model' in df.columns:
        df['Model'] = pd.Categorical(df['Model'], categories=MODEL_ORDER, ordered=True)
        sort_cols.append('Model')
        
    # --- 3. Sort ---
    df = df.sort_values(sort_cols)
    df = df.drop(columns=['sort_key'])
    return df


def _format_segment_names(df):
    """
    Cleans segment names to create a visual hierarchy and enforces a logical sort order.
    
    Transforms:
        'Established Full' -> 'Established (Total)'
        'Established / High Volatility' -> '  L High Volatility'
    
    Returns:
        df (pd.DataFrame): The dataframe with a new 'Display_Segment' column and sorted rows.
    """
    df = df.copy()
    
    # --- 1. Define the Mappings for "Elegant" Display ---
    def _clean_name(row):
        seg = row['Segment']
        
        # Handle the "Full" parent group
        if seg == 'Established Full':
            return 'Established (Total)'
        
        # Handle the segments
        elif 'Established Segments' in seg and '/' in seg:
            # Extract the volatility part
            sub_segment = seg.split('/')[-1].strip() 
            return f"   L {sub_segment}"
            
        # Handle Cold Start
        elif seg == 'Cold-Start':
            return 'Cold-Start'
            
        return seg 

    df['Display_Segment'] = df.apply(_clean_name, axis=1)

    # --- 2. Enforce Logical Sort Order ---
    df = _apply_custom_sort(df)
    
    return df


def _format_mean_range(row, col_name, precision=2):
    """Standard Mean (Min-Max)"""
    mean = row[(col_name, 'mean')]
    min_v = row[(col_name, 'min')]
    max_v = row[(col_name, 'max')]
    if pd.isna(mean): return "N/A"
    return f"{mean:.{precision}f} ({min_v:.{precision}f}-{max_v:.{precision}f})"

def _format_int_range(row, col_name):
    """Volume formatting (Whole numbers)"""
    mean = row[(col_name, 'mean')]
    min_v = row[(col_name, 'min')]
    max_v = row[(col_name, 'max')]
    if pd.isna(mean): return "N/A"
    return f"{int(mean)} ({int(min_v)}-{int(max_v)})"

def _format_time(seconds):
    """Converts seconds to decimal hours (e.g., 2.50 h)."""
    if pd.isna(seconds) or seconds == 0:
        return "0.00 h"
    hours = seconds / 3600
    return f"{hours:.2f} h"

def _format_val(val, precision):
    """Helper to format a single value safely."""
    if pd.isna(val): return "N/A"
    return f"{val:.{precision}f}"

def _format_percent_range(row, col_name):
    """Weights formatting (Percentages)"""
    mean = row[(col_name, 'mean')] * 100
    min_v = row[(col_name, 'min')] * 100
    max_v = row[(col_name, 'max')] * 100
    if pd.isna(mean): return "-"
    return f"{mean:.2f}% ({min_v:.2f}-{max_v:.2f}%)"

def _clean_table(df, table_type='metrics'):
    """
    Applies formatting, sorting, and cleaning.
    """
    df = df.copy()
    
    # --- 1. Enforce Model Order (Categorical) ---
    if 'Model' in df.columns:
        df['Model'] = pd.Categorical(df['Model'], categories=MODEL_ORDER, ordered=True)
    
    # --- 2. Enforce Row Order (Custom Segment Sort) ---
    if 'Display_Segment' in df.columns:
        df = _apply_custom_sort(df, 'Display_Segment')

    # --- 3. Column Sorting for Weights (Wide Format) ---
    if table_type == 'weights':
        cols = ['Display_Segment']
        cols += [m for m in MODEL_ORDER if m in df.columns]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]

    # --- 4. Format Metrics ---
    metric_cols = ['rmse', 'wape', 'crps', 'val_log_lik']
    for col in metric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

    # --- 5. Format Costs (Seconds -> Hours) ---
    time_cols = [c for c in df.columns if 'Time' in c]
    for col in time_cols:
        df[col] = df[col].apply(lambda x: _format_time(x) if pd.notnull(x) else "N/A")
        new_name = col.replace('(s)', '(h)')
        df.rename(columns={col: new_name}, inplace=True)

    # --- 6. Format Weights (%) ---
    if table_type == 'weights':
        cols_to_format = [c for c in df.columns if c in MODEL_ORDER]
        for col in cols_to_format:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "-")

    return df


def generate_fold_report(fold, metrics_dict, train_costs, tune_costs, weights_dict):
    """
    Generates two tables:
    1. Comprehensive Metrics & Costs Table (RMSE, WAPE, CRPS, Train Cost, Tune Cost)
    2. BMA Weights Table

    Args:
        fold (str): Current fold to process.
        metrics_dict (dict): Dictionary containing all evaluation metrics.
        train_costs (dict): Dictionary containing all training costs.
        tune_costs (dict): Dictionary containing all tuning costs.
        weights_dict (dict): Dictionary containing all weights.

    Returns:
        (pd.DataFrame), (pd.DataFrame): Dataframes containing tables for metrices and weights.
    """
    save_dir = RESULTS_DIR / 'reports'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- TABLE 1: Metrics & Costs ---
    logger.info(f"Generating detailed report for {fold}...")
    
    # --- 1.1 Flatten all dictionaries ---
    flat_metrics = _recursive_flatten_metrics(metrics_dict)
    flat_train = _recursive_flatten_metrics(train_costs)
    flat_tune = _recursive_flatten_metrics(tune_costs)
    
    # --- 1.2 Convert to DataFrames ---
    df_metrics = pd.DataFrame(flat_metrics)
    df_train_costs = pd.DataFrame(flat_train)
    df_tune_costs = pd.DataFrame(flat_tune)
    
    # --- 1.3 Pivot the DataFrames ---
    # Metrics
    if not df_metrics.empty:
        df_metrics_pivot = df_metrics.pivot(index=['Segment', 'Model'], columns='Metric', values='Value').reset_index()
    else:
        df_metrics_pivot = pd.DataFrame()
    
    # Train Costs
    if not df_train_costs.empty:
        df_train_costs_pivot = df_train_costs.pivot(index=['Segment', 'Model'], columns='Metric', values='Value').reset_index()
        df_train_costs_pivot = df_train_costs_pivot.rename(columns={'time': 'Train_Time(s)', 'co2': 'Train_CO2(kg)'})
    else:
        df_train_costs_pivot = pd.DataFrame()

    # Tune Costs
    if not df_tune_costs.empty:
        df_tune_costs_pivot = df_tune_costs.pivot(index=['Segment', 'Model'], columns='Metric', values='Value').reset_index()
        df_tune_costs_pivot = df_tune_costs_pivot.rename(columns={'time': 'Tune_Time(s)', 'co2': 'Tune_CO2(kg)'})
    else:
        df_tune_costs_pivot = pd.DataFrame()

    # --- 1.4 Merge Everything ---
    full_df = pd.merge(df_metrics_pivot, df_train_costs_pivot, on=['Segment', 'Model'], how='outer')
    full_df = pd.merge(full_df, df_tune_costs_pivot, on=['Segment', 'Model'], how='outer')
    
    # --- 1.5 Format Segments ---
    full_df = _format_segment_names(full_df)
    metrics_df = _clean_table(full_df, table_type='metrics')
    
    # --- 1.6 Reorder Columns ---
    cols = ['Display_Segment', 'Model', 'rmse', 'wape', 'crps', 'val_log_lik', 
            'Train_Time(h)', 'Train_CO2(kg)', 'Tune_Time(h)', 'Tune_CO2(kg)']
    cols = [c for c in cols if c in metrics_df.columns]
    metrics_df = metrics_df[cols]
    
    # --- 1.7 Save Table 1 ---
    metrics_df.to_csv(save_dir / f"results_{fold}.csv", index=False)

    # LaTeX Export
    with open(save_dir / f"results_{fold}.tex", "w") as f:
        f.write(metrics_df.to_latex(index=False))
    
    # --- TABLE 2: BMA Weights ---
    # --- 2.1 Flatten the dictionaries ---
    flat_weights = _recursive_flatten_weight(weights_dict)
    df_weights = pd.DataFrame(flat_weights)

    # --- 2.2 Pivot: Segments as Rows, Models as Columns ---
    weights_pivot = df_weights.pivot(index='Segment', columns='Model', values='Weight').reset_index()
    
    # --- 2.3 Format Segments ---
    weights_df = _format_segment_names(weights_pivot)

    # --- 2.4 Clean up Segment column ---
    if 'Segment' in weights_df.columns and 'Display_Segment' in weights_df.columns:
        weights_df = weights_df.drop(columns=['Segment'])
        # Move Display_Segment to front
        cols = ['Display_Segment'] + [c for c in weights_df.columns if c != 'Display_Segment']
        weights_df = weights_df[cols]

    # --- 2.5 Format table ---
    report_weights = _clean_table(weights_df, table_type='weights')
            
    # --- 2.6 Save Table 2 ---
    report_weights.to_csv(save_dir / f"weights_{fold}.csv", index=False)

    # LaTeX Export
    with open(save_dir / f"weights_{fold}.tex", "w") as f:
        f.write(report_weights.to_latex(index=False))

    logger.info(f"Saved Metrices and Weights tables to {save_dir}")
        
    return full_df, weights_df, metrics_df


def _bootstrap_difference(metric_a, metric_b, n_boot=1000, confidence=0.95):
    """
    Bootstraps the difference between two metric vectors (Model A - Model B).

    Args:
        metric_a (np.array): Contains pointwise CRPS for Model A.
        metric_b (np.array): Contains pointwise CRPS for Model B.
        n_boot (int): Number of samples to bootstrap.
        confidence (float): Confidence level.

    Returns: 
        (float), (float), (float), (Bool): Mean difference, Lowe and Upper Confidence interval and True/False significance marker.
    """
    # --- 1. Filter for valid indices (remove NaNs from masking/Cold-Start mismatches) ---
    valid_mask = ~(np.isnan(metric_a) | np.isnan(metric_b))
    a = metric_a[valid_mask]
    b = metric_b[valid_mask]
    
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0, False
    
    # --- 2. Calculate Actual Difference ---
    diff_data = a - b 
    mean_diff = np.mean(diff_data)
    
    # --- 3. Bootstrap ---
    # --- 3.1 Resample indices with replacement ---
    indices = np.random.randint(0, n, size=(n_boot, n))

    # --- 3.2 Compute mean ---
    boot_means = np.mean(diff_data[indices], axis=1) 
    
    # --- 3.3 Compute Confidence Intervals ---
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    
    # --- 3.4 Check Significance: 0 is NOT in the interval
    is_significant = (lower > 0) or (upper < 0)
    
    return mean_diff, lower, upper, is_significant


def aggregate_results(fold_report_dfs, fold_weights_dfs):
    """
    Aggregates results across folds with specific formatting for each data type.
    """
    save_dir = RESULTS_DIR / 'reports'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================
    # PART 1: Metrics & Costs
    # =========================================================
    # --- 1.1 Combine Raw DataFrames ---
    combined = pd.concat(fold_report_dfs, ignore_index=True)
    
    # --- 1.2 Convert Time Seconds -> Hours ---
    time_cols = [c for c in combined.columns if 'Time' in c]
    for col in time_cols:
        combined[col] = combined[col] / 3600
        # Rename columns immediately to (h)
        new_name = col.replace('(s)', '(h)')
        combined = combined.rename(columns={col: new_name})
        
    # --- 1.3 Identify Columns to Aggregate ---
    nums = combined.select_dtypes(include=['number']).columns.tolist()
    if 'Fold' in nums: nums.remove('Fold')
    
    # --- 1.4 Group and Aggregate ---
    agg = combined.groupby(['Display_Segment', 'Model'])[nums].agg(['mean', 'min', 'max'])
    final_metrics = pd.DataFrame(index=agg.index)
    
    # --- 1.5 Apply Specific Formatters per Column Type ---
    for col in nums:
        if 'volume' in col.lower():
            # Integer formatting
            final_metrics[col] = agg.apply(lambda x: _format_int_range(x, col), axis=1)
        elif 'co2' in col.lower():
            # High precision for CO2 (6 decimals)
            final_metrics[col] = agg.apply(lambda x: _format_mean_range(x, col, precision=6), axis=1)
        else:
            # Standard Metrics & Time (2 decimals)
            final_metrics[col] = agg.apply(lambda x: _format_mean_range(x, col, precision=2), axis=1)
        
    # --- 1.6 Reset Index and Sort ---
    final_metrics = final_metrics.reset_index()
    
    # Sort Rows (Segment + Model)
    final_metrics['Model'] = pd.Categorical(final_metrics['Model'], categories=MODEL_ORDER, ordered=True)
    final_metrics = _apply_custom_sort(final_metrics)
    
    # --- 1.7 Reorder Columns ---
    desired_cols = ['Display_Segment', 'Model', 'rmse', 'wape', 'crps', 'val_log_lik', 
                    'Train_Time(h)', 'Train_CO2(kg)', 'Tune_Time(h)', 'Tune_CO2(kg)', 'volume']
    # Keep only columns that exist
    final_cols = [c for c in desired_cols if c in final_metrics.columns]
    final_metrics = final_metrics[final_cols]
    
    # --- 1.8 Save ---
    final_metrics.to_csv(save_dir / "aggregated_metrics.csv", index=False)
    with open(save_dir / "aggregated_metrics.tex", "w") as f: f.write(final_metrics.to_latex(index=False))
    logger.info("Saved Aggregated Metrics.")

    # =========================================================
    # PART 2: BMA Weights
    # =========================================================
    combined = pd.concat(fold_weights_dfs, ignore_index=True)
    
    # Models are columns here
    models = [c for c in combined.columns if c not in ['Display_Segment', 'Fold']]
    
    agg = combined.groupby(['Display_Segment'])[models].agg(['mean', 'min', 'max'])
    final_weights = pd.DataFrame(index=agg.index)
    
    # Apply Percentage Formatter
    for col in models:
        final_weights[col] = agg.apply(lambda x: _format_percent_range(x, col), axis=1)
        
    final_weights = final_weights.reset_index()
    final_weights = _apply_custom_sort(final_weights)
    
    # Reorder Columns (Model Order)
    cols = ['Display_Segment'] + [m for m in MODEL_ORDER if m in final_weights.columns]
    final_weights = final_weights[cols]
    
    final_weights.to_csv(save_dir / "aggregated_weights.csv", index=False)
    with open(save_dir / "aggregated_weights.tex", "w") as f: f.write(final_weights.to_latex(index=False))
    logger.info("Saved Aggregated Weights.")
    
    return final_metrics


def run_significance_tests(fold, bootstrap_data_container):
    """
    Aggregates data, runs bootstrap, and produces:
    1. A Clean Pivoted Table
    2. A Raw Table
    """
    save_dir = RESULTS_DIR / 'reports'
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("--- Starting Statistical Significance Testing (Per Segment) ---")
    
    # Storage: {SegmentPath: {Model: [Vector_Fold1, Vector_Fold2]}}
    segment_data_storage = {}

    # --- 1. Recursive Collector ---
    def _collect_by_segment(target_node, pred_node, current_path):
        if 'y_test' in target_node:
            y_true = target_node['y_test']
            mask = target_node['mask_test']
            
            if current_path not in segment_data_storage:
                segment_data_storage[current_path] = {}

            for model, samples in pred_node.items():
                if samples.shape[0] > 0:
                    loss_vec = calculate_pointwise_crps(y_true, samples, mask)
                    if model not in segment_data_storage[current_path]:
                        segment_data_storage[current_path][model] = []
                    segment_data_storage[current_path][model].append(loss_vec)
            return

        for key in target_node.keys():
            if key in pred_node:
                new_path = f"{current_path}/{key}" if current_path else key
                _collect_by_segment(target_node[key], pred_node[key], new_path)

    # --- 2. Populate Storage ---
    for fold_data in bootstrap_data_container:
        _collect_by_segment(fold_data['targets'], fold_data['predictions'], "")

    # --- 3. Run Tests Per Segment ---
    results = []
    
    for segment, models_dict in segment_data_storage.items():
        if "BMA" not in models_dict: continue
            
        bma_vec = np.concatenate(models_dict["BMA"])
        
        # Compare against every other model
        for model in models_dict.keys():
            if model == "BMA": continue
            
            other_vec = np.concatenate(models_dict[model])
            
            # Run Test (BMA - Model)
            diff, lower, upper, sig = _bootstrap_difference(bma_vec, other_vec)
            
            if not np.isnan(diff):
                results.append({
                    "Segment": segment,
                    "Opponent": model,
                    "Mean Diff": diff,
                    "CI Lower": lower,
                    "CI Upper": upper,
                    "Significant": sig,
                    "Winner": "BMA" if (sig and diff < 0) else (model if (sig and diff > 0) else "Tie")
                })

    # --- 4. Process Raw Results ---
    df_sig = pd.DataFrame(results)
    
    # --- 4.1 Enforce Model Order ---
    df_sig['Opponent'] = pd.Categorical(df_sig['Opponent'], categories=MODEL_ORDER, ordered=True)
    df_sig = df_sig.sort_values(['Segment', 'Opponent'])
    
    # --- 4.2 Round Raw Floats ---
    numeric_cols = ['Mean Diff', 'CI Lower', 'CI Upper']
    df_sig[numeric_cols] = df_sig[numeric_cols].round(2)
    
    # --- 4.3 Save Raw Table ---
    raw_path = save_dir / f"significance_testing_raw_{fold}.csv"
    df_sig.to_csv(raw_path, index=False)

    # --- 5. Create Clean Pivoted Table ---
    # --- 5.1 Format Cell Content ---
    def _format_sig_cell(row):
        sig_mark = "**" if row['Significant'] else ""
        return f"{row['Winner']}{sig_mark} ({row['Mean Diff']:.2f})"

    df_sig['Cell_Report'] = df_sig.apply(_format_sig_cell, axis=1)

    # --- 5.2 Pivot (Segment x Opponent) ---
    pivot_sig = df_sig.pivot(
        index='Segment', 
        columns='Opponent', 
        values='Cell_Report'
    ).reset_index()

    # --- 5.3 Format Segments & Sort Rows ---
    pivot_sig = _format_segment_names(pivot_sig)

    # --- 5.4 Final Cleanup ---
    if 'Segment' in pivot_sig.columns: 
        pivot_sig = pivot_sig.drop(columns=['Segment'])
        
    # Construct final column order
    final_cols = ['Display_Segment'] + [m for m in MODEL_ORDER if m in pivot_sig.columns]
    pivot_sig = pivot_sig[final_cols]

    # --- 5.5 Save Clean Table ---
    save_path = save_dir / f"significance_testing_pivoted_{fold}.csv"
    pivot_sig.to_csv(save_path, index=False)
    
    with open(save_dir / f"significance_table_{fold}.tex", "w") as f:
        f.write(pivot_sig.to_latex(index=False))
        
    logger.info(f"Significance tables saved to {RESULTS_DIR}")
    return pivot_sig


def plot_pareto_frontier(fold, aggregated_metrics_df):
    """
    Generates Pareto Frontier plots (CRPS vs. Compute Time) for each segment.
    
    Args:
        fold (str): Current fold proccessed.
        metrics_df (pd.DataFrame): DataFrame with metrics.
    """
    logger.info("Generating Pareto Frontier Plots...")
    save_dir = RESULTS_DIR / 'reports' / 'figures' / 'pareto'
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Preprocess Data ---
    df = aggregated_metrics_df.copy()
    
    # Extract numeric values from the formatted strings (e.g., "0.45 (0.40-0.50)" -> 0.45)
    def _extract_val(val_str):
        try:
            return float(val_str.split(' ')[0])
        except:
            return np.nan

    df['crps_val'] = df['crps'].apply(_extract_val)
    df['train_h'] = df['Train_Time(h)'].apply(_extract_val)
    df['tune_h'] = df['Tune_Time(h)'].apply(_extract_val)
    
    # Total Compute Cost = Train + Tune
    df['total_time_h'] = df['train_h'].fillna(0) + df['tune_h'].fillna(0)
    
    # --- 2. Iterate per Segment ---
    segments = df['Display_Segment'].unique()
    
    for seg in segments:
        seg_df = df[df['Display_Segment'] == seg].copy()
        
        if seg_df.empty: continue
        
        # --- 3. Identify Pareto Frontier ---
        models = seg_df['Model'].values
        costs = seg_df['total_time_h'].values
        errors = seg_df['crps_val'].values
        
        is_frontier = []
        for i in range(len(seg_df)):
            c_i = costs[i]
            e_i = errors[i]
            dominated = False
            for j in range(len(seg_df)):
                if i == j: continue
                # Check if model J dominates model I (J is faster and more accurate)
                if (costs[j] <= c_i and errors[j] < e_i) or (costs[j] < c_i and errors[j] <= e_i):
                    dominated = True
                    break
            is_frontier.append(not dominated) # If model False dominated, it is True frontier
            
        seg_df['is_frontier'] = is_frontier
        
        # --- 4. Plotting ---
        plt.figure(figsize=(10, 6))
        
        # Plot non-frontier points
        non_front = seg_df[~seg_df['is_frontier']]
        plt.scatter(non_front['total_time_h'], non_front['crps_val'], color='gray', alpha=0.5, s=100, label='Sub-optimal')
        
        # Plot frontier points
        front = seg_df[seg_df['is_frontier']].sort_values('total_time_h') # Sort for line drawing
        plt.plot(front['total_time_h'], front['crps_val'], 'b--', alpha=0.3) # Connect frontier
        plt.scatter(front['total_time_h'], front['crps_val'], color='red', s=150, edgecolors='black', label='Pareto Frontier')
        
        # Annotate Models
        for _, row in seg_df.iterrows():
            plt.text(row['total_time_h'], row['crps_val'] + (max(errors)*0.01), 
                     row['Model'], fontsize=9, ha='center', fontweight='bold')

        plt.title(f"Efficiency-Accuracy Trade-off: {seg}")
        plt.xlabel("Total Computational Cost (GPU/CPU Hours)")
        plt.ylabel("Probabilistic Error (CRPS)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        safe_name = seg.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(save_dir / f"{fold}_pareto_{safe_name}.png", dpi=300)
        plt.close()
        
    logger.info(f"Pareto plots saved to {save_dir}")


def plot_calibration_curves(fold, test_split, all_samples):
    """
    Generates plot per segment, containing lines for each model. 
 
    Args:
        fold (str): Current fold to process.
        test_split (dict): Dictionary containing y_true, and masks.
        all_samples (dict): Dictionary containing samples.
    """
    logger.info(f"Generating Calibration Plots for {fold}...")
    save_dir = RESULTS_DIR / 'reports' / 'figures' / 'calibration'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    quantiles = np.linspace(0.1, 0.9, 9)
    
    # --- 1. Identify Segments ---
    target_segments = [
        ('Established Full', test_split['Established Full']),
        ('Cold-Start', test_split['Cold-Start']),
        ('High Volatility', test_split['Established Segments']['High Volatility']),
        ('Moderate Volatility', test_split['Established Segments']['Moderate Volatility']),
        ('Low Volatility', test_split['Established Segments']['Low Volatility'])
    ]
    
    # --- 2. Loop the segments ---
    for seg_name, seg_data_test in target_segments:
        if seg_name == 'High Volatility':
            seg_samples = all_samples['Established Segments']['High Volatility']
        elif seg_name == 'Moderate Volatility':
            seg_samples = all_samples['Established Segments']['Moderate Volatility']
        elif seg_name == 'Low Volatility':
            seg_samples = all_samples['Established Segments']['Low Volatility']
        else:
            seg_samples = all_samples[seg_name]

        y_true = seg_data_test['y_test']
        mask = seg_data_test['mask_test']
        valid_mask = mask == 1
        y_flat = y_true[valid_mask]
        
        if len(y_flat) < 100: continue

        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label="Ideal")

        # --- 3. Loop the models ---
        for model, samples in seg_samples.items():
            # Skip baselines if they don't have samples (empty array)
            if samples.shape[0] == 0: continue
            
            # Check if model is BMA to style it differently (Bold/Red)
            style = 'r-' if model == 'BMA' else '--'
            width = 3 if model == 'BMA' else 1.5
            alpha = 1.0 if model == 'BMA' else 0.6
            
            empirical_coverage = []
            for q in quantiles:
                q_pred = np.percentile(samples, q * 100, axis=2)
                q_flat = q_pred[valid_mask]
                coverage = np.mean(y_flat <= q_flat)
                empirical_coverage.append(coverage)
            
            plt.plot(quantiles, empirical_coverage, style, linewidth=width, alpha=alpha, label=model)

        plt.xlabel("Theoretical Quantile")
        plt.ylabel("Empirical Coverage")
        plt.title(f"Calibration: {seg_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # --- 4. Save the plot for the segment ---
        safe_name = seg_name.replace(" ", "_")
        plt.savefig(save_dir / f"calibration_{fold}_{safe_name}.png", dpi=300)
        plt.close()


def plot_aggregated_calibration_curves(test_splits_list, samples_list):
    """
    Concatenates history from all folds and generates the aggregated plot.
    """
    logger.info("Aggregating data from all folds for Calibration...")
    
    # --- 1. Concatenate the lists ---
    agg_test_split = _recursive_concat(test_splits_list)
    agg_all_samples = _recursive_concat(samples_list)

    # --- 2. Plot calibration curves ---
    plot_calibration_curves("Aggregated", agg_test_split, agg_all_samples)


def plot_example_forecasts(fold, test_split, all_samples, n_examples=3):
    """
    Plots forecast examples: Truth (Black), BMA (Blue Shaded), DeepAR/TFT/DeepTCN (Red Dotted).

    Args:
        fold (str): Current plot to process.
        test_split (dict): Dictionary containing y_true and masks.
        all_samples (dict): Dictionary containing samples.
        n_examples (int): Number of predictions to compare per segment.
    """
    logger.info(f"Generating Example Forecast Plots for {fold}...")
    
    save_dir = RESULTS_DIR / 'reports' / 'figures' / 'examples'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Identify Segments ---
    target_segments = [
        ('Established Full', test_split['Established Full']),
        ('Cold-Start', test_split['Cold-Start']),
        ('High Volatility', test_split['Established Segments']['High Volatility']),
        ('Moderate Volatility', test_split['Established Segments']['Moderate Volatility']),
        ('Low Volatility', test_split['Established Segments']['Low Volatility'])
    ]
    
    # --- 2. Loop the segments ---
    for seg_name, seg_data_test in target_segments:
        if seg_name == 'High Volatility':
            seg_samples = all_samples['Established Segments']['High Volatility']
        elif seg_name == 'Moderate Volatility':
            seg_samples = all_samples['Established Segments']['Moderate Volatility']
        elif seg_name == 'Low Volatility':
            seg_samples = all_samples['Established Segments']['Low Volatility']
        else:
            seg_samples = all_samples[seg_name]

        y_true = seg_data_test['y_test']
        mask = seg_data_test['mask_test']
        valid_indices = [i for i in range(len(mask)) if np.sum(mask[i]) > 0]

        bma_samples = seg_samples['BMA']
        competitors = ['DeepAR', 'TFT', 'DeepTCN']
        
        
        if len(y_true) == 0: continue

        # Select random indices
        indices = np.random.choice(valid_indices, min(n_examples, len(y_true)), replace=False)
        
        # --- 3. Loop random forecasts ---
        for idx in indices:
            for model in competitors:
                comp_samples = seg_samples[model]

                # Extract series
                y_series = y_true[idx]

                # BMA
                b_s = bma_samples[idx]
                b_p10 = np.percentile(b_s, 10, axis=1)
                b_p50 = np.percentile(b_s, 50, axis=1)
                b_p90 = np.percentile(b_s, 90, axis=1)
                
                # DeepAR/TFT/DeepTCN
                d_s = comp_samples[idx]
                d_p10 = np.percentile(d_s, 10, axis=1)
                d_p50 = np.percentile(d_s, 50, axis=1)
                d_p90 = np.percentile(d_s, 90, axis=1)
                
                plt.figure(figsize=(12, 6))
                x = range(len(y_series))
                
                # --- 3.1 Plot Ground Truth ---
                plt.plot(x, y_series, 'k-o', label='Actual', linewidth=2, zorder=10)
                
                # --- 3.2 Plot BMA (Blue)
                plt.plot(x, b_p50, 'b--', label='BMA Median', linewidth=2, zorder=5)
                plt.fill_between(x, b_p10, b_p90, color='b', alpha=0.2, label='BMA 80% CI')
                
                # --- 3.3 Plot DeepAR/TFT/DeepTCN (Red) ---
                plt.plot(x, d_p50, 'r:', label=f'{model} Median', linewidth=2, zorder=6)
                plt.fill_between(x, d_p10, d_p90, color='r', alpha=0.1, label=f'{model} 80% CI')
                
                plt.title(f"Forecast: {seg_name} | Item {idx}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xlabel("Prediction Steps")
                plt.ylabel("Demand")
                
                safe_seg = seg_name.replace(" ", "_").replace("/", "_")
                plt.savefig(save_dir / f"{fold}_{model}_{safe_seg}_{idx}_example.png", dpi=300)
                plt.close()


if __name__ == "__main__":
    pass
