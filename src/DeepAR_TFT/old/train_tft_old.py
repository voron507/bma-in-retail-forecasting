import config
import json
import time
import logging
import numpy as np
import pandas as pd
import gc
from pathlib import Path
import optuna
from codecarbon import OfflineEmissionsTracker

# PyTorch and PyTorch Lightning for model training
import torch
import lightning.pytorch as pl 
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from DeepAR_TFT.zinb_loss import ZINBLoss, crps_from_quantiles, crps_from_samples, calculate_pointwise_log_likelihood
from pytorch_forecasting.metrics import NegativeBinomialDistributionLoss, QuantileLoss
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
import torch.nn.functional as F


# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set its level to WARNING to hide INFO messages
loggers_to_silence = ["lightning.pytorch.utilities.rank_zero", "lightning.pytorch.accelerators.cuda"]
for logger_name in loggers_to_silence:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


# --- Configuration ---
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
SAMPLES_PER_SEED = N_SAMPLES / len(ENSEMBLE_SEEDS)


def tune_tft(
    fold: str,
    train_data: TimeSeriesDataSet,
    val_data: TimeSeriesDataSet,
    n_trials: int = 40
) -> dict:
    """
    Performs hyperparameter tuning for TFT on a given fold.

    Args:
        fold (str): The fold being processed (e.g., 'fold1').
        train_data (TimeSeriesDataSet): The training TimeSeriesDataSet.
        val_data (TimeSeriesDataSet): The validation TimeSeriesDataSet.
        n_trials (int): The number of Optuna trials to run for tuning.
    
    Returns:
        dict: Dictionary with the best hyperparameters after tuning.
    """
    # --- 1. Define Paths and Check for Final Output ---
    model_dir = MODELS_DIR / "TFT" / fold
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = model_dir / "tensorboard_logs" / f"{fold}_tuning_tft"

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
        try:
            hidden_size = trial.suggest_categorical("hidden_size", [128, 160])
            dropout = trial.suggest_float("dropout", 0.15, 0.35)
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [32, 64])
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
            #gradient_clip = trial.suggest_categorical("gradient_clip", [0.5, 1.0])
            #alpha_reg = trial.suggest_float("alpha_reg", 1e-6, 1e-3, log=True)
            #gate_reg = trial.suggest_float("gate_reg", 9e-5, 1.2e-4, log=True)

            trial_logger = TensorBoardLogger(save_dir=str(tb_log_dir), name=f"tft_{fold}", version=f"trial_{trial.number}")
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")

            trainer = pl.Trainer(
                max_epochs = 100,
                accelerator = "auto",
                gradient_clip_val = 1.0,
                callbacks = [early_stop_callback],
                logger = trial_logger,
                enable_progress_bar = True,
                enable_model_summary = False
            )

            # --- 3. Instantiate the model with hyperparameters ---
            model = TemporalFusionTransformer.from_dataset(
                train_data,
                learning_rate=lr,
                hidden_size=hidden_size,
                attention_head_size=4,
                dropout=dropout,
                hidden_continuous_size=hidden_continuous_size,
                log_interval=10,
                reduce_on_plateau_patience=4,
                weight_decay = weight_decay,
                #loss = NegativeBinomialDistributionLoss(),
                # output_size=len(QUANTILES),
                # loss = QuantileLoss(quantiles=QUANTILES)
                loss=ZINBLoss() #alpha_reg=alpha_reg) #, gate_reg=gate_reg)
            )
            
            with torch.no_grad():
                model.output_layer.bias[0].fill_(3.0)
                model.output_layer.bias[2].fill_(-3.0)

            logger.info("--- Initialized Output Bias: Mean += 3.0, Gate = -3.0 ---")

            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            # --- 4. Assess CRPS for the trial ---
            logger.info("Assessing CRPS for the trial...")
            predictions_raw = model.predict(
                val_dataloader, 
                mode="raw",
                return_y=True
            )
            
            # predictions_raw.output.prediction has shape (n_time_series, horizon, 3)
            prediction = predictions_raw.output.prediction
            y_true_tensor, y_mask_tensor = predictions_raw.y
            y_true = y_true_tensor.cpu().numpy()
            y_mask = y_mask_tensor.cpu().numpy()

            # Sample from the distribution
            dist = model.loss
            if not isinstance(dist, ZINBLoss):
                raise ValueError("Model loss is not Zero Inflated Negative Binomial. Manual sampling logic needs adjustment.")
            
            distribution = model.loss.map_x_to_distribution(prediction)
            n_samples_to_draw = 200

            prediction_samples_tensor = distribution.sample((n_samples_to_draw,)).permute(1, 2, 0).contiguous()
            prediction_samples = prediction_samples_tensor.cpu().numpy()

            # Calculate CRPS
            crps = crps_from_samples(y_true_tensor, prediction_samples_tensor, y_mask_tensor)

            # Debug trials
            p50_predictions = np.percentile(prediction_samples, 50, axis=2)
            mean_predictions = np.mean(prediction_samples, axis=2).round()
            rmse = np.sqrt(np.sum(((mean_predictions - y_true) ** 2) * y_mask) / (np.sum(y_mask) + 1e-8))
            logger.info(f"--- Val Set RMSE (p50 vs y_true): {rmse:.4f} ---")
            abs_errors = np.abs(p50_predictions - y_true) * y_mask
            wape = abs_errors.sum() / (np.abs(y_true * y_mask).sum() + 1e-8)
            logger.info(f"--- Val Set WAPE (p50 vs y_true): {wape:.4f} ---")

            avg_mean = (mean_predictions * y_mask).sum() / (y_mask.sum() + 1e-8)
            avg_var = ((mean_predictions - avg_mean) ** 2 * y_mask).sum() / (y_mask.sum() + 1e-8)
            avg_std = np.sqrt(avg_var)
            p50_masked = p50_predictions * y_mask
            zero_mask_p50 = (p50_masked == 0) & (y_mask == 1)
            emp_zero_rate_p50 = zero_mask_p50.sum() / y_mask.sum()
            # y true
            y_true_mean = (y_true * y_mask).sum() / (y_mask.sum() + 1e-8)
            y_true_var = ((y_true - y_true_mean) ** 2 * y_mask).sum() / (y_mask.sum() + 1e-8)
            y_true_std = np.sqrt(y_true_var)
            y_true_masked = y_true * y_mask
            zero_mask_y = (y_true_masked == 0) & (y_mask == 1)
            emp_zero_rate_y = zero_mask_y.sum() / y_mask.sum()
            logger.info(f"    -> P50 Predictions: Mean={avg_mean:.4f}, Std={avg_std:.4f}, Empirical zero rate={emp_zero_rate_p50:.4f}")
            logger.info(f"    -> Y_True Values:   Mean={y_true_mean:.4f}, Std={y_true_std:.4f}, Empirical zero rate={emp_zero_rate_y:.4f}")

            val_loss = early_stop_callback.best_score
            logger.info(f"Best validation loss for the trial is: {val_loss}")

            if val_loss is None or not np.isfinite(val_loss.item()):
                logger.warning(f"Optuna trial {trial.number} resulted in non-finite validation loss.")
                raise optuna.TrialPruned("Failed due to non-finite validation loss")
            
            # Controll the memory
            try:
                del model
                del trainer
                del predictions_raw
                del y_true, y_mask
                del dist
                del prediction
                del distribution
                del prediction_samples_tensor
                del prediction_samples
                del p50_predictions, mean_predictions, rmse, abs_errors, wape, avg_mean, avg_var, avg_std, p50_masked, zero_mask_p50, emp_zero_rate_p50
                del y_true_mean, y_true_var, y_true_std, y_true_masked, zero_mask_y, emp_zero_rate_y
            except:
                pass   
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            return crps
            # return val_loss

        except optuna.TrialPruned as e:
            raise e
        except Exception as e:
             logger.error(f"Optuna trial {trial.number} failed during training: {e}")
             raise optuna.TrialPruned(f"Failed due to training error: {e}")

    # --- 5. Run Hyperparameter Tuning ---
    logger.info(f"--- Starting hyperparameter tuning for {fold} ({n_trials} trials) ---")
    study = optuna.create_study(direction="minimize")
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
        logger.info(f"--- Tuning complete. Best trial {study.best_trial.number} with CRPS value {study.best_value:.4f}. ---")
        logger.info(f"Best hyperparameters: {best_params}")
        
    except (optuna.exceptions.OptunaError, ValueError) as e:
         logger.error(f"Optuna optimization failed or found no successful trials: {e}.")
         logger.warning("Using default hyperparameters for final training.")
         # Define sensible defaults if tuning fails
         best_params = {
             "hidden_size": 128,
             "attention_head_size": 4,
             "dropout": 0.2,
             "lr": 1e-3,
             "hidden_continuous_size": 32
         }

    # --- 6. Save the best hyperparameters ---
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"--- Best hyperparameters saved to {best_params_path} ---")

    return best_params


def train_tft(
    fold: str,
    train_data: TimeSeriesDataSet,
    val_data: TimeSeriesDataSet,
    best_params: dict
):
    """
    Trains the final TFT model using a given set of hyperparameters.

    Args:
        fold (str): The fold being processed.
        train_data (TimeSeriesDataSet): Training dataset.
        val_data (TimeSeriesDataSet): Validation dataset.
        hyperparameters (dict): Dictionary containing best hyperparameters after tuning.

    Returns:
        Object, Path: Best model checkpoint loaded and path to it.
    """
    # --- 1. Define paths to save ---
    model_dir = MODELS_DIR / "TFT" / fold
    tb_log_dir = model_dir / "tensorboard_logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path_ckpt = model_dir / "best_model.ckpt"

    if best_model_path_ckpt.exists():
        logger.info(f"--- Found existing trained TFT model for {fold}. Loading... ---")
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path_ckpt)
        return best_model, best_model_path_ckpt

    # --- 2. Create dataloaders for final training ---
    train_dataloader = train_data.to_dataloader(train=True, batch_size=128, num_workers=4)
    val_dataloader = val_data.to_dataloader(train=False, batch_size=128, num_workers=4)

    logger.info("--- Training final model with best hyperparameters. ---")
    final_logger = TensorBoardLogger(save_dir=tb_log_dir, name=f"tft_{fold}_final")
    checkpoint_callback = ModelCheckpoint(dirpath=model_dir, filename="best_model", monitor="val_loss", mode="min", save_top_k=1)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=15, verbose=True, mode="min")

    # --- 3. Conduct Final Training ---
    final_trainer = pl.Trainer(
        max_epochs = 100,
        accelerator = "auto",
        gradient_clip_val = 1.0, #best_params["gradient_clip"],
        callbacks = [early_stop_callback, checkpoint_callback],
        logger = final_logger,
        enable_progress_bar = True
    )

    final_model = TemporalFusionTransformer.from_dataset(
        train_data,
        learning_rate = best_params["lr"],
        hidden_size = best_params["hidden_size"],
        attention_head_size = 4,
        dropout = best_params["dropout"],
        hidden_continuous_size = best_params["hidden_continuous_size"],
        log_interval=10,
        reduce_on_plateau_patience=4,
        weight_decay = best_params["weight_decay"],
        #loss = NegativeBinomialDistributionLoss(),
        # output_size=len(QUANTILES),
        # loss = QuantileLoss(quantiles=QUANTILES),
        loss = ZINBLoss() #alpha_reg=best_params["alpha_reg"]) #, gate_reg=best_params["gate_reg"])
    )

    with torch.no_grad():
        final_model.output_layer.bias[0].fill_(3.0)
        final_model.output_layer.bias[2].fill_(-3.0)

    logger.info("--- Initialized Output Bias: Mean += 3.0, Gate = -3.0 ---")


    try:
        final_trainer.fit(final_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        best_model_path_str = checkpoint_callback.best_model_path
        if best_model_path_str:
             logger.info(f"--- Best model checkpoint for {fold} saved in {best_model_path_str} ---")
             best_model = TemporalFusionTransformer.load_from_checkpoint(Path(best_model_path_str))
             return best_model, Path(best_model_path_str)
        else:
             logger.warning(f"--- Final training completed but no model checkpoint was saved for {fold} (check val_loss). ---")
             return None, None

    except Exception as e:
        logger.error(f"Final model training failed for fold {fold}: {e}", exc_info=True)
        return None, None
    

def predict_tft(
    fold: str,
    best_model: TemporalFusionTransformer,
    test_data: TimeSeriesDataSet,
    val_data: TimeSeriesDataSet
    ) -> Path | None:
    """
    Generates and saves probabilistic predictions for the test set.

    Args:
        fold (str): The fold being processed.
        best_model (Object): Trained model.
        test_data (TimeSeriesDataSet): Test TimeSeriesDataSet.
        val_data (TimeSeriesDataSet): Validation TimeSeriesDataSet.

    Returns:
        Path | None: Path to the saved .npz predictions file, or None if prediction failed.
    """
    logger.info(f"--- Starting prediction on test set for {fold} ---")
    
    # --- 1. Define forecasts paths ---
    pred_dir = FORECASTS_DIR / "TFT"
    pred_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = pred_dir / f"test_predictions_tft_{fold}.npz"

    if predictions_path.exists():
        logger.info(f"--- Predictions for {fold} already exist. Skipping. ---")
        return predictions_path

    try:
        # --- 2. Set Model to Evaluation ---
        best_model.eval()

        # --- 3. Create DataLoaders for Prediction ---
        val_dataloader = val_data.to_dataloader(
            train=False, batch_size=128, num_workers=4
        )

        predict_dataloader = test_data.to_dataloader(
            train=False, batch_size=128, num_workers=4
        )

        # # --- 4. Generate loss for validation (for BMA) ---
        # raw_val = best_model.predict(
        # val_dataloader,
        # mode="quantiles", 
        # return_x=True,
        # return_index=True)

        # val_preds = np.round(raw_val.output.cpu().numpy()).clip(min=0)
        # groups = np.array(raw_val.index)
        # val_groups = groups[:, 1]
        # val_time = raw_val.x['decoder_time_idx'].cpu().numpy()

        # --- 4. Generate Validation Parameters (for BMA) (mu, alpha, gate)
        logger.info("Generating validation parameters (mu, alpha, gate)...")
        val_raw = best_model.predict(
            val_dataloader,
            return_y=True,
            return_x=True,
            return_index=True, 
            mode="raw"
        )

        val_parameters_raw = val_raw.output.prediction.cpu()
        y_true_tensor, y_mask_tensor = val_raw.y
        y_true = y_true_tensor.cpu().numpy()
        y_mask = y_mask_tensor.cpu().numpy() 
        groups = np.array(val_raw.index)
        val_groups = groups[:, 1]
        val_time = val_raw.x['decoder_time_idx'].cpu().numpy()

        # Apply Activation on raw parameters
        val_mu = torch.exp(val_parameters_raw[..., 0])
        val_mu = torch.clamp(val_mu, 1e-4, 1e8)
        val_alpha = F.softplus(val_parameters_raw[..., 1]) + 1e-3
        val_alpha = torch.clamp(val_alpha, 1e-4, 40.0)
        val_gate = val_parameters_raw[..., 2].sigmoid() * 0.999 + 0.0005 

        val_parameters = np.stack([val_mu.numpy(), val_alpha.numpy(), val_gate.numpy()], axis=-1)

        # valid_log_lik = calculate_pointwise_log_likelihood(y_true, y_mask, val_parameters)

        # --- 5. Generate Probabilistic Predictions (mu, alpha, gate) ---
        logger.info("Generating probabilistic predictions (mu, alpha, gate)...")
        predictions_raw = best_model.predict(
            predict_dataloader, 
            return_index=True,
            return_x=True, 
            mode="raw",
            return_y=True
        )
        
        # predictions_raw.output.prediction is normalized and has shape (n_time_series, horizon, 3)
        prediction_normalized_tensor = predictions_raw.output.prediction
        prediction_normalized = prediction_normalized_tensor.cpu()
        y_true_tensor, y_mask_tensor = predictions_raw.y
        y_true = y_true_tensor.cpu().numpy()
        y_mask = y_mask_tensor.cpu().numpy() 
        groups = np.array(predictions_raw.index)
        test_groups = groups[:, 1]
        test_time = predictions_raw.x['decoder_time_idx'].cpu().numpy()

        # --- 6. Sample from the distribution ---
        dist = best_model.loss
        if not isinstance(dist, ZINBLoss):
            raise ValueError("Model loss is not Zero Inflated Negative Binomial. Manual sampling logic needs adjustment.")
        
        distribution = best_model.loss.map_x_to_distribution(prediction_normalized)
        n_samples_to_draw = 200

        prediction_samples_tensor = distribution.sample((n_samples_to_draw,)).permute(1, 2, 0).cuda()
        prediction_samples = prediction_samples_tensor.cpu().numpy()
        logger.info(f"Manually generated samples shape: {prediction_samples.shape}")

        # # --- 5. Generate Probabilistic Predictions ---
        # raw_predictions = best_model.predict(
        # predict_dataloader,
        # mode="quantiles", 
        # return_y=True,
        # return_x=True,
        # return_index=True)

        # test_preds_tensor = raw_predictions.output.cpu()
        # test_y_true_tensor = raw_predictions.y[0].cpu()
        # test_weight_tensor = raw_predictions.y[1].cpu()
        # predictions = np.round(test_preds_tensor.numpy()).clip(min=0)
        # y_true = test_y_true_tensor.numpy()
        # y_mask = test_weight_tensor.numpy()
        # groups = np.array(raw_predictions.index)
        # test_groups = groups[:, 1]
        # test_time = raw_predictions.x['decoder_time_idx'].cpu().numpy()

        # p50_predictions = predictions[:, :, 3]

        # # --- 6. Manually denormalize prediction samples ---
        # logger.info("De-normalizing prediction samples...")
        # target_normalizer = test_data.target_normalizer

        # # Since center=False, we only need scale
        # norm_params = target_normalizer.get_norm(prediction_index_df)
        # scales = norm_params[:, 1]
        # logger.info(scales.shape)
        # scales_broadcastable = scales[:, np.newaxis, np.newaxis]
        # prediction_samples_denormalized = prediction_samples_normalized * scales_broadcastable
        
        # logger.info(f"Denormalized prediction samples shape: {prediction_samples_denormalized.shape}") # (N, H, S)
        # logger.info(f"Denormalized y_true shape: {y_true.shape}")
       
        # --- 7. Calculate RMSE, WAPE, and Debug Statistics ---
        logger.info("Calculating RMSE...")
        p50_predictions = np.percentile(prediction_samples, 50, axis=2)
        mean_predictions = np.mean(prediction_samples, axis=2).round()
        rmse = np.sqrt(np.sum(((mean_predictions - y_true) ** 2) * y_mask) / (np.sum(y_mask) + 1e-8))
        logger.info(f"--- Test Set RMSE (p50 vs y_true): {rmse:.4f} ---")
        logger.info("Calculating WAPE...")
        abs_errors = np.abs(p50_predictions - y_true) * y_mask
        wape = abs_errors.sum() / (np.abs(y_true * y_mask).sum() + 1e-8)
        logger.info(f"--- Test Set WAPE (p50 vs y_true): {wape:.4f} ---")
        logger.info("Calculating CRPS...")
        crps = crps_from_samples(y_true_tensor, prediction_samples_tensor, y_mask_tensor)
        logger.info(f"--- Test Set CPRS: {crps:.4f} ---")
        # logger.info("Calculating CRPS (proxy)...")
        # test_loss = calculate_quantiles_crps(test_preds_tensor, test_y_true_tensor, test_weight_tensor, QUANTILES)
        # logger.info(f"--- Test Set Quantile Loss (CRPS Proxy): {test_loss:.4f} ---")

        # Check for Normalization
        logger.info("--- Verifying prediction scales (Debug Check) ---")
        # p50
        avg_mean = (mean_predictions * y_mask).sum() / (y_mask.sum() + 1e-8)
        avg_var = ((mean_predictions - avg_mean) ** 2 * y_mask).sum() / (y_mask.sum() + 1e-8)
        avg_std = np.sqrt(avg_var)
        p50_masked = p50_predictions * y_mask
        zero_mask_p50 = (p50_masked == 0) & (y_mask == 1)
        emp_zero_rate_p50 = zero_mask_p50.sum() / y_mask.sum()
        # y true
        y_true_mean = (y_true * y_mask).sum() / (y_mask.sum() + 1e-8)
        y_true_var = ((y_true - y_true_mean) ** 2 * y_mask).sum() / (y_mask.sum() + 1e-8)
        y_true_std = np.sqrt(y_true_var)
        y_true_masked = y_true * y_mask
        zero_mask_y = (y_true_masked == 0) & (y_mask == 1)
        emp_zero_rate_y = zero_mask_y.sum() / y_mask.sum()
        logger.info(f"    -> P50 Predictions: Mean={avg_mean:.4f}, Std={avg_std:.4f}, Empirical zero rate={emp_zero_rate_p50:.4f}")
        logger.info(f"    -> Y_True Values:   Mean={y_true_mean:.4f}, Std={y_true_std:.4f}, Empirical zero rate={emp_zero_rate_y:.4f}")

        # --- 8. Construct index ---
        # Val Index
        N, H = test_time.shape
        groups_expanded = np.repeat(val_groups[:, None], H, axis=1)
        val_index_structured = np.core.records.fromarrays([groups_expanded.ravel(), val_time.ravel()], names='StockCode, time_idx')
        logger.debug(f'Index shape: {val_index_structured.shape}, 5 examples: {val_index_structured[:5]}')

        # Test Index
        N, H = test_time.shape
        groups_expanded = np.repeat(test_groups[:, None], H, axis=1)
        test_index_structured = np.core.records.fromarrays([groups_expanded.ravel(), test_time.ravel()], names='StockCode, time_idx')
        logger.debug(f'Index shape: {test_index_structured.shape}, 5 examples: {test_index_structured[:5]}')

        # --- 9. Save Predictions in NPZ format ---
        logger.info(f"Saving predictions and parameters to {predictions_path}...")

        np.savez_compressed(
            predictions_path,
            # val_preds=val_preds,
            val_params=val_parameters,
            val_index=val_index_structured,
            # log_lik=valid_log_lik,
            predictions=prediction_samples,
            # test_preds=predictions,
            test_index=test_index_structured
        )
        
        logger.info(f"--- Successfully saved predictions for {fold} ---")
        return predictions_path

    except Exception as e:
        logger.error(f"Failed to generate predictions for {fold}: {e}", exc_info=True)
        return None
    

def run_tft_pipeline(
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
    model_dir = MODELS_DIR / "TFT" / fold
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / "TFT"
    metrics_path.mkdir(parents=True, exist_ok=True)
    cost_metrics_path = metrics_path / f"tft_costs_{fold}.json"

    # --- 2. Initialize Cost Tracking ---
    tracker = OfflineEmissionsTracker(country_iso_code="NLD", output_dir=METRICS_DIR, project_name=f"tft_{fold}", log_level='error')
    tracker.start()
    start_time = time.time()

    best_model = None
    predictions_path = None
    
    try:
        # --- 3. Tuning (or skip) ---
        if params_to_use:
            logger.info(f"--- Skipping tuning. Using existing best hyperparameters ---")
            best_params = params_to_use
        else:
            logger.info(f"--- Starting tuning for {fold} ---")
            best_params = tune_tft(
                fold=fold,
                train_data=train_data,
                val_data=val_data,
                n_trials=n_trials
            )

        # --- 4. Final Training ---
        if best_params:
            best_model, best_model_path = train_tft(
                fold=fold,
                train_data=train_data,
                val_data=val_data,
                best_params=best_params
            )
        else:
            logger.error(f"--- Cannot proceed to training for {fold}, no hyperparameters found. ---")

        # --- 5. Prediction ---
        if best_model:
            predictions_path = predict_tft(
                fold=fold,
                best_model=best_model,
                test_data=test_data,
                val_data=val_data
            )
        else:
            logger.error(f"--- Cannot run prediction for {fold}, no model was trained. ---")

        return {
        "best_model_path": best_model_path,
        "cost_metrics_path": cost_metrics_path,
        "predictions_path": predictions_path,
        "hyperparameters": best_params
        }

    except Exception as e:
        logger.error(f"An error occurred during the overall process for fold {fold}: {e}", exc_info=True)
    
    
    finally:
        # --- 6. Stop Tracking and Save Costs ---
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
        
        logger.info(f"--- Saved cost metrics for {fold} to {cost_metrics_path} ---")
        logger.info(f"    -> Total Time: {total_time_seconds:.2f} seconds")
        if emissions_data is not None:
             logger.info(f"    -> CO2 Emissions: {emissions_data:.6f} kg")
        else:
             logger.warning(f"    -> CO2 Emissions: Could not be measured.")
    
    