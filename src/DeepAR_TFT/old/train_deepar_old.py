import config
from pathlib import Path
import json
import time
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import optuna
from codecarbon import OfflineEmissionsTracker
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger 

# GluonTS Imports
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.model.predictor import Predictor
from gluonts.evaluation import make_evaluation_predictions


# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set its level to WARNING to hide INFO messages
loggers_to_silence = ["lightning.pytorch.utilities.rank_zero", "lightning.pytorch.accelerators.cuda"]
for logger_name in loggers_to_silence:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


# --- Define global variables ---
# Columns
STATIC_CAT_COLS = config.DEEPAR_STATIC_CAT_COLS
STATIC_REAL_COLS = config.DEEPAR_STATIC_REAL_COLS
DYNAMIC_REAL_COLS = config.DEEPAR_DYNAMIC_REAL_COLS
GROUP_ID = config.DEEPAR_GROUP_ID
TARGET_COLUMN = config.DEEPAR_TARGET_COLUMN
# Directories
MODELS_DIR = config.MODELS_DIR
METRICS_DIR = config.METRICS_DIR
FORECASTS_DIR = config.FORECASTS_DIR
TENSORBOARD_DIR = MODELS_DIR / "DeepAR" / "tensorboard_logs"
# Parameters
ENCODER_LENGTH = config.ENCODER_LENGTH
HORIZON = config.HORIZON
FREQ = config.FREQ


def tune_deepar(
    train_data: ListDataset,
    val_data: ListDataset,
    cat_mappings: dict,
    fold: str,
    n_trials: int = 10,
):
    """
    Performs hyperparameter tuning and training for DeepAR using the GluonTS PyTorch backend.

    Args:
        train_data (ListDataset): ListDataset for the training data.
        val_data (ListDataset): ListDataset for the validation data.
        cat_mappings (dict): Factorized categories of categorical columns.
        fold (str): Curent fold to be processed.
        n_trials (int): Number of trials for tuning.

    Returns:
        best_params (dict): Dictionary containing best hyperparameters from tuning. 
    """

    # --- 1. Define Paths and Check for Final Output ---
    model_dir = MODELS_DIR / "DeepAR" / fold 
    model_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = model_dir / "best_hyperparameters.json"
    metrics_path = METRICS_DIR / "DeepAR"
    metrics_path.mkdir(parents=True, exist_ok=True) 

    if best_params_path.exists():
        logger.info(f"--- Found existing best hyperparameters for {fold}. Loading... ---")
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        return best_params

    try:
        # --- 2. Define Optuna Objective Function ---
        def objective(trial: optuna.Trial) -> float:
            try:
                num_layers = trial.suggest_int("num_layers", 2, 4)
                hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
                dropout = trial.suggest_float("dropout_rate", 0.2, 0.5)
                lr = trial.suggest_float("learning_rate", 8e-4, 3e-3, log=True)

                # Get cardinalities for static categgorical features
                cat_cardinalities = [len(cat_mappings[col]['categories']) for col in STATIC_CAT_COLS if col in cat_mappings]
                
                # Add tensorboard
                trial_log_dir = TENSORBOARD_DIR / f"tune_{fold}"
                tb_logger = TensorBoardLogger(
                    save_dir=str(trial_log_dir),
                    name=f"trial_{trial.number}",
                    version=0
                )

                # --- Use PyTorch Lightning Trainer arguments ---
                early_stop_callback = pl.callbacks.EarlyStopping(
                    monitor="val_loss", 
                    min_delta=1e-4, 
                    patience=10, 
                    verbose=False, 
                    mode="min"
                )
                
                trainer_kwargs = {
                    "max_epochs": 100,
                    "gradient_clip_val": 1.0,
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1 if torch.cuda.is_available() else "auto",
                    "enable_progress_bar": True,
                    "enable_model_summary": False,
                    "logger": tb_logger,
                    "callbacks": [early_stop_callback]
                }

                estimator = DeepAREstimator(
                    prediction_length=HORIZON,
                    context_length=ENCODER_LENGTH,
                    freq=FREQ,
                    distr_output=NegativeBinomialOutput(),
                    num_layers=num_layers,
                    hidden_size=hidden_size, 
                    dropout_rate=dropout,
                    lr = lr,
                    num_feat_static_cat=len(STATIC_CAT_COLS),
                    cardinality=cat_cardinalities if cat_cardinalities else None,
                    num_feat_static_real=len(STATIC_REAL_COLS),
                    num_feat_dynamic_real=len(DYNAMIC_REAL_COLS),
                    time_features=time_features_from_frequency_str(FREQ),
                    trainer_kwargs=trainer_kwargs 
                )

                logger.info(f"Starting Optuna trial {trial.number} training...")
                predictor = estimator.train(
                    training_data=train_data,
                    validation_data=val_data
                )
                logger.info(f"Optuna trial {trial.number} training finished.")
                
                # --- Get Validation Metric ---
                val_metric = early_stop_callback.best_score.item()
                
                if not np.isfinite(val_metric):
                        raise optuna.TrialPruned("Non-finite validation loss detected")
                
                logger.info(f"Trial {trial.number} val_loss: {val_metric:.4f}")
                return val_metric

            except optuna.TrialPruned as e:
                raise e
            except Exception as e:
                logger.error(f"Optuna trial {trial.number} failed: {e}", exc_info=True) 
                raise optuna.TrialPruned(f"Failed due to training error: {e}")

        # --- 5. Run Hyperparameter Tuning ---
        logger.info(f"--- Starting hyperparameter tuning for {fold} ({n_trials} trials) ---")
        study = optuna.create_study(direction="minimize")
        best_params = {}
        try:
            study.optimize(objective, n_trials=n_trials)
            # Print best params table
            results_df = study.trials_dataframe()
            results_df = results_df.sort_values(by='value')
            logger.info("--- Optuna Tuning All Trials: ---")
            cols_to_log = [col for col in results_df.columns if 'params_' in col or col in ['number', 'value', 'state']]
            logger.info("\n" + results_df[cols_to_log].to_string())

            best_params = study.best_trial.params
            logger.info(f"--- Tuning complete. Best trial {study.best_trial.number} with value {study.best_value:.4f}. ---")
            logger.info(f"Best hyperparameters: {best_params}")

        except (optuna.exceptions.OptunaError, ValueError) as e:
             logger.error(f"Optuna optimization failed or found no successful trials: {e}. Cannot proceed to final training.")
             best_params = None

        # --- 6. Save the best hyperparameters ---
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"--- Best hyperparameters saved to {best_params_path} ---")

        return best_params

    except Exception as e:
        logger.error(f"An error occurred during the DeepAR tuning process for {fold}: {e}", exc_info=True)


def train_deepar(
        train_data: ListDataset,
        val_data: ListDataset,
        cat_mappings: dict,
        best_params: dict,
        fold: str
):
    """
    Trains the final DeepAR model using a given set of hyperparameters.

    Args:
        train_data (ListDataset): Training dataset.
        val_data (ListDataset): Validation dataset.
        cat_mappings (dict): Factorized categories of categorical columns.
        best_params (dict): Dictionary with best hyperparameters.
        fold (str): The fold being processed.

    Returns:
        final_predictor (Object): Best model predictor.
    """
    
    # --- 1. Define predictor path ---
    model_path = MODELS_DIR / "DeepAR" / fold
    model_path.mkdir(parents=True, exist_ok=True)
    predictor_path = model_path / "predictor"

    if predictor_path.exists():
        logger.info("--- Training for the final model is already completed. Loading the predictor...")
        final_predictor = Predictor.deserialize(predictor_path)
        return final_predictor

    try:
        logger.info("--- Training final DeepAR model with best hyperparameters ---")
        # --- 2. Define cardinality for each categorical column ---
        cat_cardinalities = [len(cat_mappings[col]['categories']) for col in STATIC_CAT_COLS if col in cat_mappings]

        # Tensorboard logger
        final_log_dir = TENSORBOARD_DIR / f"final_{fold}"
        final_tb_logger = TensorBoardLogger(
            save_dir=str(final_log_dir), # Ensure save_dir is a string
            name="best_model",
            version=0
        )

        # --- 3. Define Trainer arguments ---
        final_trainer_kwargs = {
            "max_epochs": 150,
            "gradient_clip_val": 1.0,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1 if torch.cuda.is_available() else "auto",
            "enable_progress_bar": True,
            "enable_model_summary": True,
            "logger": final_tb_logger,
            "callbacks": [
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", 
                    min_delta=1e-4, 
                    patience=15, 
                    verbose=True, 
                    mode="min"
                )
            ]
        }

        # --- 4. Define the model and its parameters ---
        final_estimator = DeepAREstimator(
            prediction_length=HORIZON,
            context_length=ENCODER_LENGTH,
            freq=FREQ,
            distr_output=NegativeBinomialOutput(),
            num_layers=best_params["num_layers"],
            hidden_size=best_params["hidden_size"], 
            dropout_rate=best_params["dropout_rate"],
            lr=best_params["learning_rate"],
            num_feat_static_cat=len(STATIC_CAT_COLS),
            cardinality=cat_cardinalities if cat_cardinalities else None,
            num_feat_static_real=len(STATIC_REAL_COLS),
            num_feat_dynamic_real=len(DYNAMIC_REAL_COLS),
            time_features=time_features_from_frequency_str(FREQ),
            
            trainer_kwargs=final_trainer_kwargs
        )
        
        final_predictor = final_estimator.train(
            training_data=train_data,
            validation_data=val_data
        )
        
        predictor_path.mkdir(parents=True, exist_ok=True)
        final_predictor.serialize(predictor_path)
        logger.info(f"--- Saved final DeepAR predictor for {fold} to {predictor_path} ---")

        return final_predictor

    except Exception as e:
        logger.error(f"An error occurred during the DeepAR final training process for {fold}: {e}", exc_info=True)


def predict_deepar(
    fold: str,
    test_data: ListDataset,
    splits: dict,
    final_predictor):
    """
    Loads a trained GluonTS-Torch DeepAR model, predicts on the test set,
    and saves the results.

    Args:
        fold (str): The fold to evaluate.
        test_data (ListDataset): ListDataset for the gluonts preprocessed test data.
        splits (dict): The dictionary containing the original DataFrame splits.
        final_predictor: predictor object to evaluate on test set.
    
    Returns:
        predictions_path (Path): Path to the final predictions produced by the model.
    """
    logger.info(f"--- Starting GluonTS Prediction and MAE evaluation for {fold} ---")
    
    try:
        # --- 1. Define Paths ---
        forecasts_path = FORECASTS_DIR / "DeepAR"
        forecasts_path.mkdir(parents=True, exist_ok=True)
        predictions_path = forecasts_path / f"test_predictions_deepar_{fold}.npz"

        if predictions_path.exists():
            logger.info(f"--- Predictions for {fold} already exist. Skipping. ---")
            return predictions_path
            
        # --- 2. Generate Predictions ---
        logger.info("Generating predictions using make_evaluation_predictions...")
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,
            predictor=final_predictor,
            num_samples=200 
        )
        all_samples_list = []
        all_y_true_list = []
        
        # --- 3. Extract Samples, True Values and Index ---
        # Get predictions
        logger.info("Iterating over forecasts...")
        for forecast, ts in zip(forecast_it, ts_it):
            all_samples_list.append(forecast.samples)
            y_true_item = ts.iloc[-HORIZON:].values
            all_y_true_list.append(y_true_item)
        
        prediction_samples_transposed = np.stack(all_samples_list) # (num_series, 200, 52)
        prediction_samples = prediction_samples_transposed.transpose(0, 2, 1) # (num_series, 52, 200)
        
        # Get true values
        y_true = np.stack(all_y_true_list)

        logger.info(f"Generated prediction samples with shape: {prediction_samples.shape}")
        logger.info(f"Generated y_true with shape: {y_true.shape}")

        # Reconstruct index
        logger.info("Reconstructing index...")
        prediction_index_structured = None
        try:
            test_df = splits[fold]['test'].copy()
            test_df = test_df.sort_values(by=GROUP_ID)
            prediction_index_df = test_df[[GROUP_ID, 'Week']]
            prediction_index_df.columns = ['StockCode', 'Week']
            prediction_index_structured = prediction_index_df.to_records(index=False)
            logger.info(f"Created structured index with shape: {prediction_index_structured.shape}")
            
            expected_index_len = y_true.shape[0] * y_true.shape[1]
            if prediction_index_structured.shape[0] != expected_index_len:
                logger.warning(f"Shape mismatch! Index has {prediction_index_structured.shape[0]} rows, but y_true has {expected_index_len} values.")

        except Exception as e:
            logger.error(f"An error occurred during index reconstruction: {e}")

        # --- 4. Calculate MAE and Debug Statistics ---
        logger.info("Calculating MAE...")
        y_pred_p50 = np.percentile(prediction_samples, 50, axis=2)
        y_true_flat = y_true.reshape(-1)
        y_pred_p50_flat = y_pred_p50.reshape(-1)
        mae = np.mean(np.abs(y_true_flat - y_pred_p50_flat))
        logger.info(f"--- Test Set MAE (p50 vs y_true) for {fold}: {mae:.4f} ---")

        # Check for Normalization
        logger.info("--- Verifying prediction scales (Debug Check) ---")
        p50_mean = np.mean(y_pred_p50_flat)
        p50_std = np.std(y_pred_p50_flat)
        y_true_mean = np.mean(y_true_flat)
        y_true_std = np.std(y_true_flat)
        logger.info(f"    -> P50 Predictions: Mean={p50_mean:.4f}, Std={p50_std:.4f}")
        logger.info(f"    -> Y_True Values:   Mean={y_true_mean:.4f}, Std={y_true_std:.4f}")

        # --- 5. Save Predictions in NPZ format ---
        logger.info(f"Saving predictions to {predictions_path}...")

        np.savez_compressed(
            predictions_path,
            predictions=prediction_samples,
            index=prediction_index_structured
        )
        
        logger.info(f"--- Successfully saved predictions for {fold} ---")
        return predictions_path


    except Exception as e:
        logger.error(f"An error occurred during prediction/evaluation for fold {fold}: {e}", exc_info=True)
        return None


def run_deepar_pipeline(
    fold: str,
    train_data: ListDataset,
    val_data: ListDataset,
    test_data: ListDataset,
    splits: dict,
    cat_mappings: dict,
    params_to_use: dict | None = None,
    n_trials: int = 50
):
    """
    Runs the complete tuning, training, and prediction pipeline for a single fold.

    Args:
        fold (str): The fold being processed.
        train_data (ListDataset): ListDataset for the training data.
        val_data (ListDataset): ListDataset for the validation data.
        test_data (ListDataset): ListDataset for the test data.
        cat_mappings (dict): Dictionary containin factorized categories.
        params_to_use (dict | None, optional): 
            If provided, skip tuning and use these hyperparameters. 
            Defaults to None (run tuning).
        n_trials (int): The number of Optuna trials to run if tuning.
    
    Returns:
        dict: A dictionary containing paths to the generated artifacts.
    """
    # --- 1. Define Paths ---
    model_dir = MODELS_DIR / "DeepAR" / fold
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "DeepAR"
    metrics_path.mkdir(parents=True, exist_ok=True)
    cost_metrics_path = metrics_path / f"deepar_costs_{fold}.json"
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. Initialize Cost Tracking ---
    tracker = OfflineEmissionsTracker(country_iso_code="NLD", output_dir=metrics_path, project_name=f"deepar_{fold}", log_level='error')
    tracker.start()
    start_time = time.time()
    
    try:
        # --- 3. Conduct Hyperparameter Tuning
        if params_to_use:
            logger.info(f"--- Skipping tuning. Using parameters from {params_to_use} ---")
            best_params = params_to_use
        else:
            logger.info(f"--- Starting tuning for {fold} ---")
            best_params = tune_deepar(  
                train_data=train_data,
                val_data=val_data,
                cat_mappings=cat_mappings,
                fold=fold,
                n_trials=n_trials
            )

        # --- 4. Conduct Final Training ---
        if best_params:
            final_predictor = train_deepar(
                train_data=train_data,
                val_data=val_data,
                cat_mappings=cat_mappings,
                best_params=best_params,
                fold = fold
            )
        else:
            logger.error(f"--- Cannot proceed to training for {fold}, no hyperparameters found. ---")

        # --- 5. Make Final Prediction ---
        if final_predictor:
            predictions_path = predict_deepar(
                fold=fold,
                test_data=test_data,
                splits=splits,
                final_predictor=final_predictor
            )
        else:
            logger.error(f"--- Cannot run prediction for {fold}, no model was trained. ---")

    except Exception as e:
        logger.error(f"An error occurred during the overall process for fold {fold}: {e}", exc_info=True)
        

    finally:
        # --- 7. Stop Tracking and Save Costs ---
        end_time = time.time()
        total_time_seconds = end_time - start_time
        emissions_data = tracker.stop()

        cost_metrics = {
            "fold": fold,
            "total_time_seconds": round(total_time_seconds, 2),
            "co2_emissions_kg": emissions_data if emissions_data is not None else 0
        }
        
        with open(cost_metrics_path, 'w') as f:
            json.dump(cost_metrics, f, indent=4)
        
        logger.info(f"--- Saved cost metrics for GluonTS-Torch {fold} to {cost_metrics_path} ---")
        logger.info(f"    -> Total Time: {total_time_seconds:.2f} seconds")
        if emissions_data is not None:
             logger.info(f"    -> CO2 Emissions: {emissions_data:.6f} kg")
        else:
             logger.warning(f"    -> CO2 Emissions: Could not be measured.")

    return {
        "final_predictor": final_predictor,
        "cost_metrics_path": cost_metrics_path,
        "predictions_path": predictions_path,
        "best_params": best_params
    }