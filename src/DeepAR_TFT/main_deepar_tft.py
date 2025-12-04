import config
import pandas as pd
import numpy as np
import logging
import time
import random
import torch
import sys
import lightning.pytorch as pl 


from DeepAR_TFT import preprocess_deepar_tft, train_deepar, train_tft


# --- Define global variables ---
MODELS_DIR=config.MODELS_DIR
MAIN_SEED = config.MAIN_SEED


# --- 1. Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE), # Log to a file
        logging.StreamHandler(sys.stdout)  # Log to the console
    ],
    force=True
)

logger = logging.getLogger(__name__)


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


def deepar_tft_pipeline():
    """
    Runs a full DeepAR and TFT pipeline.
    """
    # --- 1. Load preprocessed dataset for folds ---
    splits = preprocess_deepar_tft.load_splits()

    # --- 2. Run data preparation and full pipeline on each fold ---
    for fold in splits:

        # --- 2.1 Preprocess (DeepAR and TFT specific) data for the fold ---
        train_dt, val_dt, test_dt = preprocess_deepar_tft.preprocess_deepar_tft_fold(splits, fold)

        if fold == 'fold1':
            # --- 2.2 Full pipeline for DeepAR (fold 1) ---
            deepar_dict = train_deepar.run_deepar_pipeline(fold, train_dt, val_dt, test_dt, n_trials=40)

            # --- 2.3 Full pipeline for TFT (fold 1) ---
            tft_dict = train_tft.run_tft_pipeline(fold, train_dt, val_dt, test_dt, n_trials=40)
        
        else:
            # --- 2.4 Full pipeline for DeepAR (fold 2 + holdout) ---
            best_params = deepar_dict['hyperparameters']
            train_deepar.run_deepar_pipeline(fold, train_dt, val_dt, test_dt, best_params)

            # --- 2.5 Full pipeline for TFT (fold 2 + holdout)
            best_params = tft_dict['hyperparameters']
            train_tft.run_tft_pipeline(fold, train_dt, val_dt, test_dt, best_params)



if __name__ == "__main__":
    set_seed(MAIN_SEED)
    deepar_tft_pipeline()
    