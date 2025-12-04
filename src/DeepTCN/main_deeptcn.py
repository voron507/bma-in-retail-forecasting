import config
import pandas as pd
import numpy as np
import logging
import time
import sys
import random
import mxnet as mx
import os

from DeepTCN.preprocess_deeptcn import preprocess_deeptcn
from DeepTCN.negbinom_forecast import run_deeptcn_pipeline


# Define global variables
MODELS_DIR = config.MODELS_DIR
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


def main_tcn():
    """
    Main Pipeline for DeepTCN model.
    """
    # --- 1. Preprocess data for DeepTCN ---
    data_path = preprocess_deeptcn()

    # --- 2. Loop through the folds ---
    for fold in data_path:
        # Collect data for the fold
        tr_path = data_path[fold]['train_path']
        val_path = data_path[fold]['validation_path']
        test_path = data_path[fold]['test_path']
        preprocessors = data_path[fold]['preprocessors']
        
        # --- 3. Conduct Tuning, Training, and Prediction for the first fold ---
        if fold == 'fold1':
            _, best_params, _ = run_deeptcn_pipeline(preprocessors, tr_path, val_path, test_path, fold, 40)
        
        # --- 4. Conduct Training, and Prediction for the second fold and holdout ---
        else:
            run_deeptcn_pipeline(preprocessors, tr_path, val_path, test_path, fold, best_params_path = best_params)


if __name__ == "__main__":
    set_seed(MAIN_SEED)
    main_tcn()