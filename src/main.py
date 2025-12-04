import logging
import config
import sys
import random
import numpy as np
import cupy as cp
import os

# --- 1. Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE), # Log to a file
        logging.StreamHandler(sys.stdout)  # Log to the console
    ]
)

logger = logging.getLogger(__name__)

# --- 2. Set up seed for reproducibility ---
def set_seed(seed=config.MAIN_SEED):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    print(f"Random seed set to {seed}")

# --- 3. Set up import statements for script ---
from utils.preprocess_dataset import preprocess_data
from train_baseline import run_baseline_pipeline
from utils.utils import prepare_preds_bma, prepare_preds_eval, _inject_ll
from ensemble_bma import ensemble_on_fold
from utils.evaluate_preds import evaluate_fold
from utils.report_results import generate_fold_report, run_significance_tests, aggregate_results, plot_pareto_frontier, plot_calibration_curves, plot_aggregated_calibration_curves, plot_example_forecasts

# --- 4. Pipeline ---
def main():
    logger.info("--- Pipeline starting ---")
    # --- 4.1 Load and Preprocess dataset ---
    splits = preprocess_data()

    # --- 4.2 Train the baselines ---
    for fold in splits:
        run_baseline_pipeline(fold, splits)

    # --- 4.2 Train the models ---
    # Done in separate environments

    # Initialize Containers for Aggregation
    fold_report_dfs = []
    fold_weights_dfs = []
    fold_test_splits = []
    fold_all_samples = []
    global_tune_costs = {}
    bootstrap_data_container = [] # for statistical significance
    try:
        # Iterate for each fold
        for i, fold in enumerate(splits):
            # Ensure reproducibility
            fold_seed = config.MAIN_SEED + i
            set_seed(fold_seed)
            logger.info(f"--- Processing {fold} (Seed reset to {fold_seed}) ---")

            # --- 4.3 Load and segment the models ---
            val_split, params_split, test_split, samples_split, samples_base_split, point_preds_base_split = prepare_preds_bma(fold, splits)

            # --- 4.4 Ensemble the models ---
            weights, bma_preds, val_ll_dict = ensemble_on_fold(fold, val_split, params_split, test_split, samples_split)

            # --- 4.5 Prepare data for the evaluation ---
            all_samples, train_costs, tune_costs = prepare_preds_eval(fold, test_split, samples_split, bma_preds, samples_base_split)
            if fold == 'fold1':
                global_tune_costs = tune_costs.copy()

            # --- 4.6 Evaluate the results ---
            metrics_dict = evaluate_fold(fold, test_split, all_samples, point_preds_base_split)
            # Inject log-likelihood for report
            _inject_ll(metrics_dict, val_ll_dict)

            if fold == 'hold_out':
                # --- 4.7 Report the fold ---
                _, _, metrics_df = generate_fold_report(fold, metrics_dict, train_costs, global_tune_costs, weights)
                # --- 4.8 Conduct Paretto analysis ---
                plot_pareto_frontier(fold, metrics_df)
                # --- 4.9 Perform significance testing ---
                bootstrap_holdout = {
                    'fold': fold,
                    'targets': test_split,
                    'predictions': all_samples
                }
                run_significance_tests(fold, [bootstrap_holdout])
                # --- 4.10 Conduct error analysis (Calibration + Examples) ---
                plot_calibration_curves(fold, test_split, all_samples)
                plot_example_forecasts(fold, test_split, all_samples)
            else:
                # --- 4.7 Report the fold ---
                metrics_df, weights_df, _ = generate_fold_report(fold, metrics_dict, train_costs, tune_costs, weights)
                # --- 4.8 Save the variables for aggregation ---
                # Save metrices and weights
                metrics_df['Fold'] = fold
                weights_df['Fold'] = fold
                fold_test_splits.append(test_split)
                fold_all_samples.append(all_samples)
                fold_report_dfs.append(metrics_df)
                fold_weights_dfs.append(weights_df)

                # Save raw files for bootstrap resampling
                bootstrap_data_container.append({
                    'fold': fold,
                    'targets': test_split,   # Contains y_true, mask per segment
                    'predictions': all_samples # Contains samples per segment per model
                })

        # --- 4.9 Aggregate across folds ---
        agg_data = aggregate_results(fold_report_dfs, fold_weights_dfs)

        # --- 4.10 Conduct Paretto Analysis on aggregated data ---
        plot_pareto_frontier("agg", agg_data)

        # --- 4.11 Perform significance testing on aggregated data ---
        run_significance_tests("agg", bootstrap_data_container)

        # --- 4.12 Conduct error analysis on aggregated data (Calibration) ---
        plot_aggregated_calibration_curves(fold_test_splits, fold_all_samples)

        logger.info("--- Pipeline has successfuly finished ---")

    except Exception as e:
        logger.error(f"Error: {e}")

    

if __name__ == "__main__":
    set_seed()
    main()