import pickle
import logging
import joblib
import numpy as np
import random
import gc
import os
from mxnet import nd
import mxnet as mx
from scipy.stats import nbinom
import config
import optuna
import pandas as pd
import json
import time
import warnings
from codecarbon import OfflineEmissionsTracker
from mxboard import SummaryWriter

# --- Import our modified model and trainer ---
from DeepTCN.MxnetModels.negbinom_models import TCN
from DeepTCN.MxnetModels.negbinom_trainer import nnTrainer
from DeepTCN.utils import rhoRisk, crps_from_samples_mx


# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings(
    "ignore", 
    message=".*should be a tuple.*"
)

# --- Define global variables ---
# Columns
TARGET_COL = config.TARGET
ID_COL = config.GROUP_ID
TIME_COL = config.TIME_COL
# Directories
FORECASTS_DIR = config.FORECASTS_DIR
MODELS_DIR = config.MODELS_DIR
METRICS_DIR = config.METRICS_DIR
# Parameters
ENCODER_LENGTH = config.ENCODER_LENGTH
HORIZON = config.HORIZON
MAIN_SEED = config.MAIN_SEED
ENSEMBLE_SEEDS = config.ALL_SEEDS
N_SAMPLES = config.N_SAMPLES
SAMPLES_PER_SEED = N_SAMPLES // len(ENSEMBLE_SEEDS)


def set_seed(seed: int):
    """
    Sets the seed for reproducibility across Python, NumPy, and MXNet.
    """
    # --- 1. Set base seed ---
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    # --- 2. Set Python hash seed ---
    os.environ['PYTHONHASHSEED'] = str(seed)

    # --- 3. MXNet / CuDNN Determinism ---
    os.environ['MXNET_ENFORCE_DETERMINISM'] = '1'
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    logger.info(f"--- Seed set to {seed} ---")


def load_data(pkl_file):
    """
    Loads the .pkl file.

    Args:
        pkl_file (Path): Pickle file path.
    """
    logger.debug(f"Loading data from {pkl_file}...")
    if not pkl_file.exists():
        logger.error(f"Data file not found: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    logger.debug("Data loaded.")

    # Extract arrays
    x_target = data['x_target'].astype(np.float32)
    y_target = data['y_target'].astype(np.float32)
    x_cat = data['x_cat'].astype(np.float32)
    x_real = data['x_real'].astype(np.float32)
    mask_future = data['mask_future'].astype(np.float32)
    index = data['index']

    logger.debug(f"  - x_target shape: {x_target.shape}")
    logger.debug(f"  - y_target shape: {y_target.shape}")
    logger.debug(f"  - x_cat shape: {x_cat.shape}")
    logger.debug(f"  - x_real shape: {x_real.shape}")
    logger.debug(f"  - index shape: {index.shape}")

    return (x_target, x_cat, x_real, y_target, mask_future, index)


def save_deeptcn(val_data_file, test_data_file, fold, params_list, samples_list):
    """
    Loads the best model and evaluates it on the test set, and saves.

    Args:
        test_data (Path): Test data path.
        val_data (Path): Validation data path.
        fold (str): Current fold to process.
        params_list: List of validation parameters from 5 seeds.
        samples_list: List of samples from 5 seeds.

    Returns:
        save_path_file (Path): Path to the saved predictions.
    """
    # Set context for calculating CRPS
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

    # --- 1. Process lists ---
    final_val_params = np.stack(params_list, axis=2)
    final_test_samples = np.concatenate(samples_list, axis=-1)

    # --- 2. Load test and validation data
    _, _, _, _, _, valIndex = load_data(val_data_file)
    _, _, _, testY, testMask, testIndex = load_data(test_data_file)

    # Get median and p90 for metrics
    mask_flat = testMask.reshape(-1).astype(bool)
    y_pred_p50_flat = np.percentile(final_test_samples, 50, axis=2).reshape(-1, 1)
    y_pred_mean_flat = np.mean(final_test_samples, axis=2).round().reshape(-1, 1)
    y_pred_p50_flat_masked = y_pred_p50_flat.reshape(-1)[mask_flat]
    y_pred_mean_flat_masked = y_pred_mean_flat.reshape(-1)[mask_flat]

    # --- 5. Calculate RMSE, WAPE, rho50 (Legacy), and rho90 (Legacy) and Debug Statistics ---
    y_true_flat = testY.reshape(-1, 1)
    y_true_flat_masked = y_true_flat.reshape(-1)[mask_flat]
    final_rmse = np.sqrt(np.mean(np.square(y_pred_mean_flat_masked - y_true_flat_masked)))
    abs_errors = np.abs(y_pred_p50_flat_masked - y_true_flat_masked)
    wape = abs_errors.sum() / np.maximum(y_true_flat_masked.sum(), 1e-8)
    # logger.info("Calculating CRPS on GPU...")
    # y_true_nd = nd.array(testY, ctx=ctx)
    # testMask_nd = nd.array(testMask, ctx=ctx)
    # pred_samples_nd = nd.array(final_test_samples, ctx=ctx)
    # crps = crps_from_samples_mx(y_true_nd, pred_samples_nd, testMask_nd, ctx)

    y_true_flat_rounded = np.round(y_true_flat_masked)
    final_rho50 = rhoRisk(y_pred_p50_flat_masked, y_true_flat_rounded, 0.5)
    y_pred_p90_flat = np.percentile(final_test_samples, 90, axis=2).reshape(-1, 1)
    y_pred_p90_flat_masked = y_pred_p90_flat.reshape(-1)[mask_flat]
    final_rho90 = rhoRisk(y_pred_p90_flat_masked, y_true_flat_rounded, 0.9)

    logger.info("--- Test Set Check Complete ---")
    logger.info(f"  Test RMSE: {final_rmse:.4f}")
    logger.info(f"  Test WAPE: {wape:.4f}")
    # logger.info(f"  Test CRPS: {crps:.4f}")
    logger.info(f"  Test Rho50:  {final_rho50:.4f}")
    logger.info(f"  Test Rho90:  {final_rho90:.4f}")

    logger.debug("--- Verifying prediction scales (Debug Check) ---")
    avg_mean = np.mean(y_pred_mean_flat_masked)
    avg_std = np.std(y_pred_mean_flat_masked)
    p50_0 = (y_pred_p50_flat_masked == 0).sum() / y_pred_p50_flat_masked.size
    y_true_mean = np.mean(y_true_flat_masked)
    y_true_std = np.std(y_true_flat_masked)
    y_true_0 = (y_true_flat_masked == 0).sum() / y_true_flat_masked.size
    logger.debug(f"    -> P50 Predictions: Mean={avg_mean:.4f}, Std={avg_std:.4f}, Empirical zero rate={p50_0:.4f}")
    logger.debug(f"    -> Y_True Values:   Mean={y_true_mean:.4f}, Std={y_true_std:.4f}, Empirical zero rate={y_true_0:.4f}")

    # --- 6. Save predictions ---
    save_path = FORECASTS_DIR / 'DeepTCN'
    save_path.mkdir(parents=True, exist_ok=True)
    save_path_file = save_path / f"test_predictions_deeptcn_{fold}.npz"
    np.savez_compressed(
        save_path_file,
        val_params=final_val_params,
        val_index=valIndex,
        predictions=final_test_samples,
        test_index=testIndex
    )
    logger.info(f"--- Predictions (samples) and index saved to {save_path_file} ---")

    return save_path_file


def train_predict_deeptcn(
    preprocessor_file,
    train_data_file,
    val_data_file,
    test_data_file,
    fold: str,
    params: dict,
    params_str: str,
    run_evaluation: bool = True,
    params_list = None,
    samples_list = None
    ):
    """
    Main callable function to run the DeepTCN training and inference.

    Args:
        preprocessor_file (Path): transformations applied to data.
        train_data_file (Path): Training data path.
        val_data_file (Path): Validation data path.
        test_data_file (Path): Test data path.
        fold (str): Current fold.
        params (dict): Parameter dictionary.
        params_str (str): A string to identify this run.
        run_evaluation (bool): Whether to run the prediction on test set or not.
        params_list (list): List of validation parameters to aggregate.
        samples_list (list): List of test samples to aggregate.
        
    Returns:
        crps (float): CRPS for the validation/test set.
        best_valid_loss (float): Best validation loss from training.
        final_rmse (float): Predicted RMSE for the test set.
        wape (float): Predicted WAPE for the test set.
        final_rho50 (float): Predicted Rho50 for the test set.
        best_model_path (Path): The file path to the best saved model.
        test_preds (Path): Path to the saved predictions.
    """
    logger.info(f"--- Starting DeepTCN Training for {fold} (Params: {params_str}) ---")
    # --- 1. Setup ---
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()
    logger.debug(f"Using context: {ctx}")
    
    # --- 2. Load Preprocessors and Data ---
    logger.debug(f"Loading preprocessors from {preprocessor_file}...")
    preprocessors = joblib.load(preprocessor_file)
    cat_features_info = preprocessors['cat_features_info']
    n_real_features = len(preprocessors['scalers']['covariates'].mean_)
    
    # --- Load all data ---
    trainX, trainCat, trainReal, trainY, trainMask, _ = load_data(train_data_file)
    valX, valCat, valReal, valY, valMask, _ = load_data(val_data_file)
    testX, testCat, testReal, _, _, _ = load_data(test_data_file)
    
    # --- 3. Build Model ---
    logger.debug("Initializing TCN model...")
    n_real_features = trainReal.shape[2] # (N, L, C) -> get C
    
    model = TCN(
        cat_features_info=cat_features_info,
        n_real_features=n_real_features,
        inputSize=ENCODER_LENGTH,
        outputSize=HORIZON,
        num_channels=params['num_channels'],
        dilations=params['dilations'], 
        embed_dim=params['embed_dim'],
        dropout=params['dropout']
    )
    
    # --- 4. Initialize Trainer ---
    trainer = nnTrainer(model, ctx, ctx)
    
    paramsDict = {
        'epochs': params['epochs'],
        'esEpochs': params['patience'],
        'evalCriteria': 'min',
        'batchSize': params['batch_size'],
        'learningRate': params['lr'],
        'alphaReg': params['alpha_reg'],
        # 'gateReg': params['gate_reg'],
        'weightDecay': params['weight_decay'],
        'clipGradient': params['clip_gradient'],
        'optimizer': 'adam',
        'initializer': mx.init.Xavier(magnitude=0.5)
    }
    
    # Configure tensorboard
    tb_log_dir = MODELS_DIR / "DeepTCN" / "tensorboard_logs" / fold / params_str
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = SummaryWriter(logdir=str(tb_log_dir), verbose=False)
    logger.debug(f"--- TensorBoard logs for this run will be saved in: {tb_log_dir} ---")

    logger.info("--- Starting Model Training ---")
    
    history, best_model_path = trainer.fit(
        mark=f"{fold}_{params_str}", # A unique name for this run
        trainX=trainX,
        trainCat=trainCat,
        trainReal=trainReal,
        trainY=trainY,
        trainMaskFuture=trainMask,
        testX=valX,
        testCat=valCat,
        testReal=valReal,
        testY=valY,
        testMaskFuture=valMask,
        paramsDict=paramsDict,
        fold=fold,
        tb_logger=tb_logger
    )
    
    logger.debug(f"--- Training complete. Best model saved to: {best_model_path} ---")

    # Get best loss
    best_valid_loss = min(history.get('valid_loss', [np.inf]))
    
    # --- 5. Generate validation parameters (For BMA) ---
    logger.info(f"--- Generating validation parameters (mu, alpha, gate) ---")
    # Predict
    mu, alpha, gate = trainer.predict(model, valX, valCat, valReal, batchSize=256)
    # Activate
    mu_stable = nd.clip(mu + 1e-3, 1e-4, 1e8)
    alpha_stable = nd.clip(alpha, 1e-4, 40.0)
    gate_stable = nd.clip(gate * 0.999 + 0.0005, 0.0005, 0.9995)
    # Move to numpy
    val_mu = mu_stable.asnumpy()
    val_alpha = alpha_stable.asnumpy()
    val_gate = gate_stable.asnumpy()
    
    # --- 6. Assess CRPS (For tuning) ---
    if not run_evaluation:
        # Define unbounded
        params_list, samples_list = -1, -1

        logger.info("Calculating CRPS...")
        y_true = valY
        
        # Generate samples
        n_samples_to_gen = N_SAMPLES
        n = 1.0 / (val_alpha + 1e-6)
        p = 1.0 / (1.0 + val_alpha * val_mu)
        nb_samples = nbinom.rvs(n, p, size=(*val_mu.shape[:-1], n_samples_to_gen))
        gate_broadcast = np.repeat(val_gate, n_samples_to_gen, axis=2)
        zero_mask = (np.random.rand(*gate_broadcast.shape) < gate_broadcast)
        pred_samples = np.where(zero_mask, 0.0, nb_samples).astype(np.float32)
        logger.info(f"Generated samples with shape: {pred_samples.shape}")

        # Calculate crps
        y_true_nd = nd.array(y_true, ctx=ctx)
        testMask_nd = nd.array(valMask, ctx=ctx)
        pred_samples_nd = nd.array(pred_samples, ctx=ctx)
        crps = crps_from_samples_mx(y_true_nd, pred_samples_nd, testMask_nd, ctx)

        logger.info(f"Best validation loss for the trial is: {best_valid_loss:.4f}")

        # Clean GPU RAM
        try:
            del model
            del trainer
            del history
            del tb_logger
            del paramsDict
            del mu, mu_stable, val_mu, alpha, alpha_stable, val_alpha, gate, gate_stable, val_gate, y_true
            del n, p, nb_samples, gate_broadcast, zero_mask
            del pred_samples
            del y_true_nd, testMask_nd, pred_samples_nd
        except:
            pass

        gc.collect()
        mx.gpu(0).empty_cache()

        return crps, params_list, samples_list

    
    if run_evaluation and not params_list is None and not samples_list is None:
        # Placeholder
        crps = -1

        # --- 7. Save validation parameters ---
        val_params = np.stack([val_mu, val_alpha, val_gate], axis=-1)
        params_list.append(val_params)
        # --- 8. Generate test samples ---
        logger.info("Generating predictions on test set...")
        mu, alpha, gate = trainer.predict(model, testX, testCat, testReal, batchSize=256)
        
        # Activate parameters and Move to numpy ---
        mu_stable = nd.clip(mu + 1e-3, 1e-4, 1e8)
        alpha_stable = nd.clip(alpha, 1e-4, 40.0)
        gate_stable = nd.clip(gate * 0.999 + 0.0005, 0.0005, 0.9995)

        pred_mu = mu_stable.asnumpy()
        pred_alpha = alpha_stable.asnumpy()
        pred_gate = gate_stable.asnumpy()
        
        # Generate samples from parameters
        n_samples_to_gen = SAMPLES_PER_SEED
        n = 1.0 / (pred_alpha + 1e-6)
        p = 1.0 / (1.0 + pred_alpha * pred_mu)

        logger.debug(f"Generating {n_samples_to_gen} samples from predicted ZINB distribution...")
        # Sample from NB
        nb_samples = nbinom.rvs(n, p, size=(*pred_mu.shape[:-1], n_samples_to_gen))
        # Broadcast gate over sample axis
        gate_broadcast = np.repeat(pred_gate, n_samples_to_gen, axis=2)
        zero_mask = (np.random.rand(*gate_broadcast.shape) < gate_broadcast)
        pred_samples = np.where(zero_mask, 0.0, nb_samples).astype(np.float32)
        
        # Save generated samples to the list
        samples_list.append(pred_samples)
        logger.info(f"Generated samples with shape: {pred_samples.shape}")

        # Clean GPU RAM
        try:
            del model
            del trainer
            del history
            del tb_logger
            del paramsDict
            del mu, mu_stable, val_mu, alpha, alpha_stable, val_alpha, gate, gate_stable, val_gate, val_params, pred_mu, pred_alpha, pred_gate
            del n, p, nb_samples, gate_broadcast, zero_mask
            del pred_samples
        except:
            pass

        gc.collect()
        mx.gpu(0).empty_cache()

        return crps, params_list, samples_list

    logger.error('Check run_evaluation, params_list, or samples_list')
    return -1, -1, -1

def _objective(trial: optuna.trial.Trial, fold: str, preprocessor_file, train_data_file, val_data_file, test_data_file):
    """
    Internal objective function for Optuna.

    Args:
        trial (Trial): Optuna trial configuration.
        fold (str): Current fold processed.
        preprocessor_file (Path): Path to the preprocessing dictionary file.
        train_data_file (Path): Path to the training data file.
        val_data_file (Path): Path to the validation data file.
        test_data_file (Path): Path to the test data file.

    Returns:
        best_valid_loss (float): Best validation loss from a training.
    """
    # --- 1. Define Hyperparameter search space ---
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        'dropout': trial.suggest_float('dropout', 0.05, 0.30),
        'embed_dim': trial.suggest_categorical('embed_dim', [20, 30]),
        'dilations': trial.suggest_categorical('dilations', [
            [1, 2, 4, 8, 16, 32],
            [1, 2, 4, 8, 16, 32, 64]
        ]),
        'num_channels': trial.suggest_categorical("num_channels", [32, 64, 128]),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        'alpha_reg': trial.suggest_float("alpha_reg", 1e-6, 1e-3, log=True),
        # 'gate_reg': trial.suggest_float("gate_reg", 7e-5, 1.8e-4, log=True),
        'clip_gradient': trial.suggest_categorical("clip_gradient", [5.0, 10.0]),
        'epochs': 50,
        'patience': 5
    }

    # Optuna can't log lists, so we create string representation
    dilations_str = 'd' + '_'.join(map(str, params['dilations']))
    params_str = (f"trial{trial.number}_lr{params['lr']:.0e}_bs{params['batch_size']}_drop{params['dropout']:.1f}_"
                  f"emb{params['embed_dim']}_ch{params['num_channels']}_wd{params['weight_decay']:.0e}_ar{params['alpha_reg']:.0e}_cg{params['clip_gradient']:.0e}_{dilations_str}") #_gr{params['gate_reg']:.0e}")

    # --- 2. Run the training function ---
    try:
        crps, _, _ = train_predict_deeptcn(
            preprocessor_file=preprocessor_file,
            train_data_file=train_data_file,
            val_data_file=val_data_file,
            test_data_file=test_data_file,
            fold=fold,
            params=params,
            params_str=params_str,
            run_evaluation=False
        )
    except Exception as e:
        logger.error(f"--- [Trial {trial.number}] FAILED with error: {e} ---")
        logger.error(f"Params: {params_str}", exc_info=True)
        raise optuna.exceptions.TrialPruned()

    return crps

def run_deeptcn_pipeline(
    preprocessor_file,
    train_data_file,
    val_data_file,
    test_data_file,
    fold: str = 'fold1', 
    n_trials: int = 50,
    best_params_path = None):
    """
    Main callable function to run Tuning, Training, and Prediction with DeepTCN.

    Args:
        preprocessor_file (Path): Path to the preprocessing dictionary file.
        train_data_file (Path): Path to the training data file.
        val_data_file (Path): Path to the validation data file.
        test_data_file (Path): Path to the test data file.
        fold (str): Current fold processed.
        n_trials (int): Number of trials to be conducted (Default 50).

    Returns:
        test_preds (Path): Path to the test data file.
        best_params_dict (dict): Dictionary containing best parameters.
        best_params_str (str): Short name of the best parameters.
    """
    
    # --- 1. Configure paths ---
    metrics_path = METRICS_DIR / "DeepTCN"
    metrics_path.mkdir(parents=True, exist_ok=True)
    training_metrics_path = metrics_path / f"deeptcn_costs_{fold}.json"
    tuning_metrics_path = metrics_path / f"deeptcn_tuning_{fold}.json"

    study_db_dir = MODELS_DIR / "DeepTCN" / fold
    study_db_dir.mkdir(parents=True, exist_ok=True)

    # If best params provided - no tuining
    if best_params_path is None:
        best_params_file = study_db_dir / "best_hyperparameters.json"
    else:
        best_params_file = best_params_path

    test_preds = None
    best_params_dict = None
    best_params_str = None

    try:
        # --- 2. Load Best Hyperparameters if they exist ---
        if best_params_file.exists():
            logger.info(f"--- Found existing best hyperparameters for {fold}. Loading from {best_params_file} ---")
            with open(best_params_file, "r") as f:
                loaded_data = json.load(f)
            best_params_dict = loaded_data["params"]
            best_params_str = loaded_data["params_str"]
            best_params_dict['epochs'] = 50
            best_params_dict['patience'] = 10
            logger.info(f"Loaded best parameters from trial {loaded_data.get('trial_number', 'unknown')}")
        else:
            # Configure Cost Tracker for Tuning
            if not tuning_metrics_path.exists():
                tracker_tune = OfflineEmissionsTracker(
                    country_iso_code="NLD",
                    output_dir=METRICS_DIR,
                    project_name=f"deeptcn_{fold}_tuning",
                    log_level='error' # Suppress info logs from codecarbon
                )
                tracker_tune.start()
                start_time_tune = time.time()
            
            # --- 3. Configute and run Optuna tuning ---
            logger.info(f"--- No existing best parameters found. Starting Optuna Hyperparameter Tuning for {n_trials} trials ---")
            set_seed(MAIN_SEED)
            storage_name = f"sqlite:///{study_db_dir.resolve()}/deeptcn_tuning_{fold}.db"

            study = optuna.create_study(
                study_name=f"deeptcn_{fold}_tuning",
                storage=storage_name,
                direction="minimize",
                load_if_exists=True
            )
        
            logger.info(f"--- Saving results to {storage_name} ---")
            study.optimize(lambda trial: _objective(trial, fold, preprocessor_file, train_data_file, val_data_file, test_data_file), n_trials=n_trials)

            # --- 4. Process and Save Tuning Results ---
            logger.info("--- Tuning Complete. Final Results: ---")
            results_df = study.trials_dataframe()
            results_df = results_df.sort_values(by='value')
            logger.info("\n" + results_df[['number', 'value', 'params_dropout', 'params_lr', 'params_batch_size', 'params_embed_dim', 'params_num_channels', 'params_weight_decay', 'params_alpha_reg', 'params_clip_gradient', 'params_dilations', 'state']].to_string()) # , 'params_gate_reg'

            best_trial = study.best_trial
            logger.info("\n--- Best Trial ---")
            logger.info(f"Trial Number: {best_trial.number}")
            logger.info(f"Validation Loss (NLL): {best_trial.value:.4f}")
            logger.info("Best Hyperparameters:")
            for key, value in best_trial.params.items():
                # Use json.dumps to handle logging the list
                logger.info(f"  - {key}: {json.dumps(value) if isinstance(value, list) else value}")
            
            # Construct param string
            dilations_str = "d" + "_".join(map(str, best_trial.params['dilations']))
            best_params_str = (f"trial{best_trial.number}_lr{best_trial.params['lr']:.0e}_bs{best_trial.params['batch_size']}_drop{best_trial.params['dropout']:.1f}_"
                            f"emb{best_trial.params['embed_dim']}_ch{best_trial.params['num_channels']}_wd{best_trial.params['weight_decay']:.0e}_ar{best_trial.params['alpha_reg']:.0e}_cg{best_trial.params['clip_gradient']:.0e}_{dilations_str}") #_gr{best_trial.params['gate_reg']:.0e}

            # Prepare dict to save (raw params from Optuna)
            params_to_save = {
                "params": best_trial.params,
                "params_str": best_params_str,
                "trial_number": best_trial.number,
                "best_value": best_trial.value
            }

            # Save to JSON
            with open(best_params_file, "w") as f:
                json.dump(params_to_save, f, indent=4)
            logger.info(f"--- Best parameters saved to {best_params_file} ---")

            # Prepare dict for training (if tuning done simulatneously)
            best_params_dict = {
                **best_trial.params,
                'epochs': 50, # Train for longer on the best model
                'patience': 10 # Use more patience
            }

            # Save Tuning Cost Metrics
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

                logger.info(f"--- Saved cost metrics for {fold} to {cost_metrics_tune} ---")
                logger.info(f"    -> Total Time: {total_time_seconds_tune:.2f} seconds")
                if emissions_data_tune is not None:
                    logger.info(f"    -> CO2 Emissions: {emissions_data_tune:.6f} kg")
                else:
                    logger.warning(f"    -> CO2 Emissions: Could not be measured.")
        
        # --- 5. Run final training, and inference on best params ---
        expected_pred_file = FORECASTS_DIR / 'DeepTCN' / f"test_predictions_deeptcn_{fold}.npz"
        
        if expected_pred_file.exists():
            logger.info(f"--- Found existing predictions at {expected_pred_file}. Skipping final training. ---")
            test_preds = expected_pred_file
        else:
            # Configure Cost Tracker for Training
            if not training_metrics_path.exists():
                tracker_train = OfflineEmissionsTracker(
                    country_iso_code="NLD", 
                    output_dir=str(METRICS_DIR), 
                    project_name=f"deeptcn_{fold}_training", 
                    log_level='error')
                tracker_train.start()
                start_time_train = time.time()

            logger.info("\n--- Running the best model to save final predictions (5 seeds) ---")
            params_list = []
            samples_list = []
            for seed in ENSEMBLE_SEEDS:
                set_seed(seed)
                logger.info(f"--- Processing Seed {seed} ---")
                _, params_list, samples_list = train_predict_deeptcn(
                    preprocessor_file=preprocessor_file,
                    train_data_file=train_data_file,
                    val_data_file=val_data_file,
                    test_data_file=test_data_file,
                    fold=fold,
                    params=best_params_dict,
                    params_str=f"BEST_{seed}_{best_params_str}",
                    run_evaluation=True,
                    params_list=params_list,
                    samples_list=samples_list
                )
            
            # --- 6. Save the results (+Debug) ---
            test_preds = save_deeptcn(
                val_data_file,
                test_data_file,
                fold,
                params_list,
                samples_list
            )

            logger.debug(f"--- Tuning, Training, and inference for fold '{fold}' complete. ---")
            logger.debug(f"Best predictions are saved in: {FORECASTS_DIR / f'test_predictions_deeptcn_{fold}.npz'}")

            # Save Training Cost Metrics
            if not training_metrics_path.exists():
                end_time_train = time.time()
                total_time_seconds_train = end_time_train - start_time_train
                emissions_train = tracker_train.stop()

                cost_metrics_train = {
                    "fold": fold,
                    'phase': 'training',
                    "total_time_seconds": round(total_time_seconds_train, 2),
                    "co2_emissions_kg": emissions_train if emissions_train is not None else 0
                }

                with open(training_metrics_path, 'w') as f:
                    json.dump(cost_metrics_train, f, indent=4)

                logger.info(f"--- Saved cost metrics for {fold} to {training_metrics_path} ---")
                logger.info(f"    -> Total Time: {total_time_seconds_train:.2f} seconds")
                if emissions_train is not None:
                    logger.info(f"    -> CO2 Emissions: {emissions_train:.6f} kg")
                else:
                    logger.warning(f"    -> CO2 Emissions: Could not be measured.")

    except Exception as e:
        logger.error(f"An error occurred during the overall process for fold {fold}: {e}", exc_info=True)


    return test_preds, best_params_file, best_params_str

if __name__ == "__main__":
    pass