import config
import json
import time
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import random
import gc
import warnings
from pathlib import Path
import optuna
from codecarbon import OfflineEmissionsTracker

# PyTorch and PyTorch Lightning for model training
import torch
import torch.nn as nn
import lightning.pytorch as pl 
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl


from DeepAR_TFT.zinb_loss import ZINBLoss, crps_from_quantiles, crps_from_samples, calculate_pointwise_log_likelihood
from pytorch_forecasting.metrics.distributions import NegativeBinomialDistributionLoss
from pytorch_forecasting import DeepAR
from pytorch_forecasting.data import TimeSeriesDataSet
import torch.nn.functional as F


# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set its level to WARNING to hide INFO messages
loggers_to_silence = ["lightning.pytorch.utilities.rank_zero", "lightning.pytorch.accelerators.cuda"]
for logger_name in loggers_to_silence:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Supress warnings
warnings.filterwarnings(
    "ignore", 
    message=".*Attribute '.*' is an instance of `nn.Module`.*"
)
warnings.filterwarnings(
    "ignore", 
    message=".*Checkpoint directory .* exists and is not empty.*"
)
warnings.filterwarnings(
    "ignore", 
    message=".*but you set.*"
)


# Directories
DATA_DIR = config.DATA_DIR
MODELS_DIR = config.MODELS_DIR
METRICS_DIR = config.METRICS_DIR
FORECASTS_DIR = config.FORECASTS_DIR
# Parameters
ENCODER_LENGTH = config.ENCODER_LENGTH
HORIZON = config.HORIZON
GROUP_ID = config.GROUP_ID
TARGET = config.TARGET
QUANTILES = config.QUANTILES
MAIN_SEED = config.MAIN_SEED
ENSEMBLE_SEEDS = config.ALL_SEEDS
N_SAMPLES = config.N_SAMPLES
SAMPLES_PER_SEED = N_SAMPLES // len(ENSEMBLE_SEEDS)


def set_seed(seed: int):
    """
    Sets the seed for reproducibility across Python, NumPy, and PL.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)
    logger.info(f"--- Seed set to {seed} ---")


def tune_deepar(
    fold: str,
    train_data: TimeSeriesDataSet,
    val_data: TimeSeriesDataSet,
    n_trials: int = 40
) -> dict:
    """
    Performs hyperparameter tuning for DeepAR on a given fold.

    Args:
        fold (str): The fold being processed (e.g., 'fold1').
        train_data (TimeSeriesDataSet): The training TimeSeriesDataSet.
        val_data (TimeSeriesDataSet): The validation TimeSeriesDataSet.
        n_trials (int): The number of Optuna trials to run for tuning.
    
    Returns:
        dict: Dictionary with the best hyperparameters after tuning.
    """
    # --- 1. Set Seed, Define Paths and Check for Final Output ---
    set_seed(MAIN_SEED)

    model_dir = MODELS_DIR / "DeepAR" / fold
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = model_dir / "tensorboard_logs" / f"{fold}_tuning_deepar"

    best_params_path = model_dir / "best_hyperparameters.json"

    if best_params_path.exists():
        logger.info(f"--- Found existing best hyperparameters for {fold}. Loading... ---")
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        return best_params

    # Create Dataloader for tuning
    train_dataloader = train_data.to_dataloader(train=True, batch_size=128, num_workers=4)
    val_dataloader = val_data.to_dataloader(train=False, batch_size=128, num_workers=4)

    # --- 2. Define Optuna Objective Function ---
    def objective(trial: optuna.Trial) -> float:

        hidden_size = trial.suggest_categorical("hidden_size", [128, 256])
        rnn_layers = trial.suggest_int("rnn_layers", 3, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.30)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        grad_clip = trial.suggest_categorical("gradient_clip_val", [0.1, 0.5, 1.0])
        #alpha_reg = trial.suggest_float("alpha_reg", 5e-6, 1e-4, log=True)
        #gate_reg = trial.suggest_float("gate_reg", 3e-5, 2e-4, log=True)
        
        trial_logger = TensorBoardLogger(save_dir=str(tb_log_dir), name=f"deepar_{fold}", version=f"trial_{trial.number}")
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=5e-5, patience=10, verbose=False, mode="min")

        trainer = pl.Trainer(
            max_epochs = 100,
            accelerator = "gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            strategy="auto",
            deterministic="warn",
            gradient_clip_val = grad_clip,
            callbacks = [early_stop_callback],
            logger = trial_logger,
            enable_progress_bar = True,
            enable_model_summary = False
        )

        # --- 3. Instantiate the model with hyperparameters ---
        model = DeepAR.from_dataset(
            train_data,
            learning_rate=lr,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            dropout=dropout,
            loss=ZINBLoss(), #alpha_reg=alpha_reg)#, gate_reg=gate_reg)
            logging_metrics=nn.ModuleList([]) # DeepAR causes errors
        )

        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # --- 4. Assess CRPS for the trial ---
        logger.info("Assessing CRPS for the trial...")

        predictions = model.predict(
        val_dataloader,  
        mode="samples",
        n_samples=N_SAMPLES, # Number of samples for probabilistic forecast
        return_y=True
        )
    
        prediction_samples = predictions.output.cuda().contiguous()
        y_true, y_mask = predictions.y
        y_true = y_true.cuda()
        y_mask = y_mask.cuda()

        # Calculate CRPS
        crps = crps_from_samples(y_true, prediction_samples, y_mask)

        val_loss = early_stop_callback.best_score
        logger.info(f"Best validation loss for the trial is: {val_loss}")

        if val_loss is None or not np.isfinite(val_loss.item()):
            logger.warning(f"Optuna trial {trial.number} resulted in non-finite validation loss.")
            raise optuna.TrialPruned("Failed due to non-finite validation loss")

        # Controll the memory
        try:
            if 'model' in locals():
                model.cpu()
            if 'trainer' in locals():
                trainer.callbacks.clear()
                trainer.loggers.clear()
            del prediction_samples
            del trial_logger
            del early_stop_callback
            del trainer
            del model
            del y_true, y_mask
            del predictions
            #del p50_predictions, mean_predictions, abs_errors, avg_mean, avg_var, avg_std, p50_masked, zero_mask_p50, y_true_mean, y_true_var, y_true_std, y_true_masked, zero_mask_y

        except:
            pass   
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return crps
        
    # --- 4. Run Hyperparameter Tuning ---
    logger.info(f"--- Starting hyperparameter tuning for {fold} ({n_trials} trials) ---")
    storage_name = f"sqlite:///results/models/DeepAR/{fold}/deepar_tuning_{fold}.db"
    study_name = f"deepar_{fold}_tuning"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize")
    best_params = {}
    try:
        study.optimize(objective, n_trials=n_trials)
        # Print optuna table
        results_df = study.trials_dataframe()
        results_df = results_df.sort_values(by='value')
        logger.info("--- Optuna Tuning All Trials: ---")
        cols_to_log = [col for col in results_df.columns if 'params_' in col or col in ['number', 'value', 'state']]
        logger.info("\n" + results_df[cols_to_log].to_string())

        best_params = study.best_trial.params
        logger.info(f"--- Tuning complete. Best trial {study.best_trial.number} with value {study.best_value:.4f}. ---")
        logger.info(f"Best hyperparameters: {best_params}")
        
    except (optuna.exceptions.OptunaError, ValueError) as e:
         logger.error(f"Optuna optimization failed or found no successful trials: {e}.")
         logger.warning("Using default hyperparameters for final training.")
         # Define sensible defaults if tuning fails
         best_params = {
             "hidden_size": 128,
             "rnn_layers": 4,
             "dropout": 0.2,
             "lr": 1e-4
         }

    # --- 5. Save the best hyperparameters ---
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"--- Best hyperparameters saved to {best_params_path} ---")

    return best_params


def train_predict_deepar(
    fold: str,
    train_data: TimeSeriesDataSet,
    val_data: TimeSeriesDataSet,
    test_data: TimeSeriesDataSet,
    best_params: dict
    ):
    """
    Trains the final DeepAR model using a given set of hyperparameters, predicts validation parameters, and test samples.
    Iterates through 5 seeds.

    Args:
        fold (str): The fold being processed.
        train_data (TimeSeriesDataSet): Training dataset.
        val_data (TimeSeriesDataSet): Validation dataset.
        test_data (TimeSeriesDataSet): Test dataset.
        best_params (dict): Dictionary containing best hyperparameters after tuning.

    Returns:
        final_val_params (np.array), final_test_samples (np.array), val_index (np.recarray), test_y_true (np.array), test_weights (np.array), test_index (np.recarray)
    """
    # --- 1. Set aggregation variables ---
    ensemble_val_params = []
    ensemble_test_samples = []
    val_index = None
    test_y_true = None
    test_weights = None
    test_index = None
    
    # --- 2. Interate through all seeds ---
    for seed in ENSEMBLE_SEEDS:
        logger.info(f"--- Processing Seed {seed} ---")
        set_seed(seed)

        # --- 2.1 Define paths to save ---
        model_dir = MODELS_DIR / "DeepAR" / fold
        tb_log_dir = model_dir / "tensorboard_logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        best_model_path_ckpt = model_dir / f"best_model_{seed}.ckpt"

        # --- 2.2 Train the model ---
        if best_model_path_ckpt.exists():
            logger.info(f"--- Found existing trained DeepAR model for {fold} {seed} seed. Loading... ---")
            best_model = DeepAR.load_from_checkpoint(best_model_path_ckpt)
        else:
            # --- 2.2.1 Create dataloaders for training ---
            train_dataloader = train_data.to_dataloader(train=True, batch_size=128, num_workers=4)
            val_dataloader = val_data.to_dataloader(train=False, batch_size=128, num_workers=4)
            logger.info("--- Training final model with best hyperparameters. ---")
            final_logger = TensorBoardLogger(save_dir=tb_log_dir, name=f"deepar_{fold}_final_{seed}")
            checkpoint_callback = ModelCheckpoint(dirpath=model_dir, filename=f"best_model_{seed}", monitor="val_loss", mode="min", save_top_k=1)
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=15, verbose=True, mode="min")
    
            # --- 2.2.2 Conduct Training ---

            final_trainer = pl.Trainer(
                max_epochs = 150,
                accelerator = "gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                strategy="auto",
                deterministic="warn",
                gradient_clip_val = best_params["gradient_clip_val"],
                callbacks = [early_stop_callback, checkpoint_callback],
                logger = final_logger,
                enable_progress_bar = True,
                enable_model_summary = False
            )

            final_model = DeepAR.from_dataset(
                train_data,
                learning_rate = best_params["lr"],
                hidden_size = best_params["hidden_size"],
                rnn_layers=best_params["rnn_layers"],
                dropout = best_params["dropout"],
                loss = ZINBLoss(), #alpha_reg=best_params["alpha_reg"], gate_reg=best_params["gate_reg"])
                logging_metrics=nn.ModuleList([]) # DeepAR causes errors
            )

            try:
                final_trainer.fit(final_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)   
                
                best_model_path_str = checkpoint_callback.best_model_path
                if best_model_path_str:
                    logger.info(f"--- Best model checkpoint for {fold} {seed} saved in {best_model_path_str} ---")
                    best_model = DeepAR.load_from_checkpoint(Path(best_model_path_str))
                else:
                    logger.warning(f"--- Final training completed but no model checkpoint was saved for {fold} {seed} seed (check val_loss). ---")

            except Exception as e:
                logger.error(f"Final model training failed for fold {fold}: {e}", exc_info=True)
            
        # --- 2.3 Generate validation parameters ---
        # Set Model to Evaluation
        best_model.eval()

        # Create dataloader for val prediction (if model loaded)
        val_dataloader = val_data.to_dataloader(
            train=False, batch_size=128, num_workers=4
        )

        # Generate validation parameters
        logger.info("Generating validation parameters (mu, alpha, gate)...")
        val_raw = best_model.predict(
            val_dataloader,
            return_x=True,
            return_index=True, 
            mode="raw"
        )

        val_parameters_raw = val_raw.output.prediction.cpu()

        # Save val index
        if val_index is None:
            groups = np.array(val_raw.index)
            val_groups = groups[:, 1]
            val_time = val_raw.x['decoder_time_idx'].cpu().numpy()
            # Construct index
            N, H = val_time.shape
            groups_expanded = np.repeat(val_groups[:, None], H, axis=1)
            val_index_structured = np.core.records.fromarrays([groups_expanded.ravel(), val_time.ravel()], names='StockCode, time_idx')
            logger.debug(f'Index shape: {val_index_structured.shape}, 5 examples: {val_index_structured[:5]}')

            # Set values
            val_index = val_index_structured

        # Apply Activation on raw parameters
        val_mu = val_parameters_raw[..., 0]
        val_alpha = val_parameters_raw[..., 1]
        val_gate = val_parameters_raw[..., 2]
        
        # Activate (exponent seems to be already applied)
        val_mu = torch.clamp(val_mu, 1e-4, 1e8)
        # Alpha/Gate
        val_alpha = F.softplus(val_parameters_raw[..., 1]) + 1e-3
        val_alpha = torch.clamp(val_alpha, 1e-4, 40.0)
        val_gate = val_parameters_raw[..., 2].sigmoid() * 0.999 + 0.0005

        # Save the parameters
        ensemble_val_params.append(torch.stack([val_mu, val_alpha, val_gate], dim=-1))

        # --- 2.4 Generate prediction samples ---
        # Load dataloader
        predict_dataloader = test_data.to_dataloader(
            train=False, batch_size=128, num_workers=4
        )

        # Generate test samples
        logger.info("Generating probabilistic predictions (samples)...")
        predictions_raw = best_model.predict(
            predict_dataloader, 
            return_index=True,
            return_x=True, 
            mode="samples",
            n_samples=SAMPLES_PER_SEED, # Number of samples for probabilistic forecast
            return_y=True
        )

        # predictions_raw.output.prediction is normalized and has shape (n_time_series, horizon, 2)
        prediction_samples_tensor = predictions_raw.output.cpu()
        # Save y values, weights and index
        if test_y_true is None and test_weights is None and test_index is None:
            y_true_tensor, y_mask_tensor = predictions_raw.y
            y_true = y_true_tensor.cpu().numpy()
            y_mask = y_mask_tensor.cpu().numpy() 
            groups = np.array(predictions_raw.index)
            test_groups = groups[:, 1]
            test_time = predictions_raw.x['decoder_time_idx'].cpu().numpy()
            # Construct index
            N, H = test_time.shape
            groups_expanded = np.repeat(test_groups[:, None], H, axis=1)
            test_index_structured = np.core.records.fromarrays([groups_expanded.ravel(), test_time.ravel()], names='StockCode, time_idx')
            logger.debug(f'Index shape: {test_index_structured.shape}, 5 examples: {test_index_structured[:5]}')

            # Set values
            test_y_true = y_true
            test_weights = y_mask
            test_index = test_index_structured

        # Save samples
        ensemble_test_samples.append(prediction_samples_tensor)

        # --- 2.5 Clean up the run ---
        try:
            if 'best_model' in locals():
                best_model.cpu()
            if 'final_trainer' in locals():
                final_trainer.callbacks.clear()
                final_trainer.loggers.clear()
            del predictions_raw
            try:
                del final_logger
                del early_stop_callback
                del checkpoint_callback
                del final_trainer
                del final_model
            except:
                pass
            del best_model
            del y_true, y_mask
            del prediction_samples_tensor
            del val_dataloader, val_raw, val_parameters_raw, val_mu, val_alpha, val_gate
            del predict_dataloader 
            del groups, val_groups, val_time, groups_expanded, val_index_structured, test_groups, test_time, test_index_structured, y_true_tensor, y_mask_tensor
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # --- 3. Aggregate the results ---
    final_val_params = torch.stack(ensemble_val_params, dim=2).numpy()
    final_test_samples = torch.cat(ensemble_test_samples, dim=-1).numpy()

    return final_val_params, final_test_samples, val_index, test_y_true, test_weights, test_index


def save_deepar(
    fold: str,
    final_val_params: np.array,
    final_test_samples: np.array,
    val_index: np.recarray,
    test_y_true: np.array,
    test_weights: np.array,
    test_index: np.recarray
    ) -> Path:
    """
    Saves parameters and probabilistic predictions.

    Args:
        fold (str): The fold being processed.
        final_val_params (np.array): Validation parameters from 5 seeds with shape (N, T, 5, 3).
        final_test_samples (np.array): Test set samples with shape (N, T, 200).
        val_index (np.recarray): Validation set index.
        test_y_true (np.array): Test set true values.
        test_weights (np.array): Test set mask.
        test_index (np.recarray): Test set index.

    Returns:
        Path: Path to the saved .npz predictions file.
    """
    # --- 1. Define forecasts paths ---
    pred_dir = FORECASTS_DIR / "DeepAR"
    pred_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = pred_dir / f"test_predictions_deepar_{fold}.npz"

    if predictions_path.exists():
        logger.info(f"--- Predictions for {fold} already exist. Skipping. ---")
        return predictions_path

    # --- 2. Calculate RMSE, WAPE, and Debug Statistics ---
    logger.debug("Calculating RMSE...")
    # y_true_tensor = torch.tensor(test_y_true).cuda()
    # prediction_samples_tensor = torch.tensor(final_test_samples).cuda()
    # y_mask_tensor = torch.tensor(test_weights).cuda()
    p50_predictions = np.percentile(final_test_samples, 50, axis=2)
    mean_predictions = np.mean(final_test_samples, axis=2).round()
    rmse = np.sqrt(np.sum(((mean_predictions - test_y_true) ** 2) * test_weights) / (np.sum(test_weights) + 1e-8))
    logger.info(f"--- Test Set RMSE (p50 vs y_true): {rmse:.4f} ---")
    logger.debug("Calculating WAPE...")
    abs_errors = np.abs(p50_predictions - test_y_true) * test_weights
    wape = abs_errors.sum() / (np.abs(test_y_true * test_weights).sum() + 1e-8)
    logger.info(f"--- Test Set WAPE (p50 vs y_true): {wape:.4f} ---")
    # logger.info("Calculating CRPS...")
    # crps = crps_from_samples(y_true_tensor, prediction_samples_tensor, y_mask_tensor)
    # logger.info(f"--- Test Set CPRS: {crps:.4f} ---")

    # --- 3. Check Statistics ---
    logger.info("--- Verifying prediction scales (Debug Check) ---")
    # p50
    avg_mean = (mean_predictions * test_weights).sum() / (test_weights.sum() + 1e-8)
    avg_var = ((mean_predictions - avg_mean) ** 2 * test_weights).sum() / (test_weights.sum() + 1e-8)
    avg_std = np.sqrt(avg_var)
    p50_masked = p50_predictions * test_weights
    zero_mask_p50 = (p50_masked == 0) & (test_weights == 1)
    emp_zero_rate_p50 = zero_mask_p50.sum() / test_weights.sum()
    # y true
    y_true_mean = (test_y_true * test_weights).sum() / (test_weights.sum() + 1e-8)
    y_true_var = ((test_y_true - y_true_mean) ** 2 * test_weights).sum() / (test_weights.sum() + 1e-8)
    y_true_std = np.sqrt(y_true_var)
    y_true_masked = test_y_true * test_weights
    zero_mask_y = (y_true_masked == 0) & (test_weights == 1)
    emp_zero_rate_y = zero_mask_y.sum() / test_weights.sum()
    logger.info(f"    -> P50 Predictions: Mean={avg_mean:.4f}, Std={avg_std:.4f}, Empirical zero rate={emp_zero_rate_p50:.4f}")
    logger.info(f"    -> Y_True Values:   Mean={y_true_mean:.4f}, Std={y_true_std:.4f}, Empirical zero rate={emp_zero_rate_y:.4f}")

    # --- 4. Save Predictions in NPZ format ---
    logger.info(f"Saving predictions and parameters to {predictions_path}...")

    np.savez_compressed(
        predictions_path,
        val_params=final_val_params,
        val_index=val_index,
        predictions=final_test_samples,
        test_index=test_index
    )
    
    logger.info(f"--- Successfully saved predictions for {fold} ---")
    return predictions_path
    

def run_deepar_pipeline(
    fold: str,
    train_data: TimeSeriesDataSet,
    val_data: TimeSeriesDataSet,
    test_data: TimeSeriesDataSet,
    params_to_use: dict | None = None,
    n_trials: int = 40
):
    """
    Runs the complete tuning, training, and prediction pipeline for a single fold.

    Args:
        fold (str): The fold being processed.
        train_data (TimeSeriesDataSet): The training TimeSeriesDataSet.
        val_data (TimeSeriesDataSet): The validation TimeSeriesDataSet.
        test_data (TimeSeriesDataSet): The test TimeSeriesDataSet.
        params_to_use (dict | None, optional): 
            If provided, skip tuning and use these hyperparameters. 
            Defaults to None (run tuning).
        n_trials (int): The number of Optuna trials to run if tuning.
    
    Returns:
        dict: A dictionary containing paths to the generated artifacts and best hyperparameters.
    """
    
    # --- 1. Define Paths ---
    model_dir = MODELS_DIR / "DeepAR" / fold
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "DeepAR"
    metrics_path.mkdir(parents=True, exist_ok=True)
    training_metrics_path = metrics_path / f"deepar_costs_{fold}.json"
    tuning_metrics_path = metrics_path / f"deepar_tuning_{fold}.json"

    training_complete = False
    predictions_path = None
    
    try:
        # --- 2. Tuning (or skip) ---
        if params_to_use:
            logger.info(f"--- Skipping tuning. Using existing best hyperparameters ---")
            best_params = params_to_use
        else:
            # Configure tracker for tuning
            if not tuning_metrics_path.exists():
                tracker_tune = OfflineEmissionsTracker(country_iso_code="NLD", output_dir=str(METRICS_DIR), project_name=f"deepar_tune_{fold}", log_level='error')
                tracker_tune.start()
                start_time_tune = time.time()

            logger.info(f"--- Starting tuning for {fold} ---")
            best_params = tune_deepar(
                fold=fold,
                train_data=train_data,
                val_data=val_data,
                n_trials=n_trials
            )

            # Save tuning costs
            if not tuning_metrics_path.exists():
                end_time_tune = time.time()
                total_time_seconds_tune = end_time_tune - start_time_tune
                emissions_data_tune = tracker_tune.stop() 
                
                cost_metrics_tune = {
                    "fold": fold,
                    'phase': 'tuning',
                    "total_time_seconds": round(total_time_seconds_tune, 2),
                    "co2_emissions_kg": emissions_data_tune if emissions_data_tune is not None else 0
                }
                
                with open(tuning_metrics_path, 'w') as f:
                    json.dump(cost_metrics_tune, f, indent=4)
                
                logger.info(f"--- Saved cost metrics for {fold} to {tuning_metrics_path} ---")
                logger.info(f"    -> Total Time: {total_time_seconds_tune:.2f} seconds")
                if emissions_data_tune is not None:
                    logger.info(f"    -> CO2 Emissions: {emissions_data_tune:.6f} kg")
                else:
                    logger.warning(f"    -> CO2 Emissions: Could not be measured.")

        # --- 3. Final Training and Prediction (5 seeds) ---
        if best_params:
            # Configure tracker for trainiing
            if not training_metrics_path.exists():
                tracker_train = OfflineEmissionsTracker(country_iso_code="NLD", output_dir=str(METRICS_DIR), project_name=f"deepar_train_{fold}", log_level='error')
                tracker_train.start()
                start_time_train = time.time()

            final_val_params, final_test_samples, val_index, test_y_true, test_weights, test_index = train_predict_deepar(
                fold=fold,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                best_params=best_params
            )
            training_complete = True

            # Save training costs
            if not training_metrics_path.exists():
                end_time_train = time.time()
                total_time_seconds_train = end_time_train - start_time_train
                emissions_data_train = tracker_train.stop() 
                
                cost_metrics_train = {
                    "fold": fold,
                    'phase': 'training',
                    "total_time_seconds": round(total_time_seconds_train, 2),
                    "co2_emissions_kg": emissions_data_train if emissions_data_train is not None else 0
                }
                
                with open(training_metrics_path, 'w') as f:
                    json.dump(cost_metrics_train, f, indent=4)
                
                logger.info(f"--- Saved cost metrics for {fold} to {training_metrics_path} ---")
                logger.info(f"    -> Total Time: {total_time_seconds_train:.2f} seconds")
                if emissions_data_train is not None:
                    logger.info(f"    -> CO2 Emissions: {emissions_data_train:.6f} kg")
                else:
                    logger.warning(f"    -> CO2 Emissions: Could not be measured.")

        else:
            logger.error(f"--- Cannot proceed to training for {fold}, no hyperparameters found. ---")

        # --- 4. Save the results (+ Debug) ---
        if training_complete:
            predictions_path = save_deepar(
                fold=fold,
                final_val_params=final_val_params,
                final_test_samples=final_test_samples,
                val_index=val_index,
                test_y_true=test_y_true,
                test_weights=test_weights,
                test_index=test_index
            )
        else:
            logger.error(f"--- Cannot run prediction for {fold}, no model was trained. ---")

        return {
        "predictions_path": predictions_path,
        "hyperparameters": best_params
    }

    except Exception as e:
        logger.error(f"An error occurred during the overall process for fold {fold}: {e}", exc_info=True)
  

if __name__ == "__main__":
    pass    