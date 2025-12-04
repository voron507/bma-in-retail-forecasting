from pathlib import Path

# --- Project Root ---
# This finds the project's root directory automatically.
PROJECT_ROOT = Path(__file__).parent.parent

# --- Data Paths ---
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_FILE = PROJECT_ROOT / "project.log"
RAW_DATA_FILE = DATA_DIR / "raw" / "online_retail_II.csv"
METRICS_DIR = RESULTS_DIR / "metrics"
FORECASTS_DIR = RESULTS_DIR / "forecasts"
MODELS_DIR = RESULTS_DIR / "models"

# --- Model Parameters (Common) ---
MAIN_SEED = 42
ALL_SEEDS = [42, 45, 48, 51, 57]
HORIZON = 13
ENCODER_LENGTH = 26
N_SAMPLES = 200
BASELINE_LEVEL = 95
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
FREQ = "W"
WEIGHT = 'loss_weight'
IS_ACTIVE = 'is_active'
GROUP_ID = 'StockCode'
TARGET = 'TotalSales'
TIME_COL = 'time_idx'
STATIC_CAT_COLS = ['static_most_frequent_country']
DYNAMIC_KNOWN_CAT_COLS = ['month_of_year']
STATIC_REAL_COLS = ['avg_weekly_customers', 'std_weekly_customers']
DYNAMIC_KNOWN_REAL_COLS_DENORM = ['AvgPrice']
DYNAMIC_KNOWN_REAL_COLS_NORM = ['week_sin', 'week_cos', 'month_sin', 'month_cos']

# --- Model Parameters (TFT + DeepAR) ---
DYNAMIC_UNKNOWN_REAL_COLS = ['TotalSales']

if __name__ == "__main__":
    pass


