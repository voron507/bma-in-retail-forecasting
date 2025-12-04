import config
import logging
import pandas as pd
import numpy as np
import kagglehub
import shutil
from pathlib import Path

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define global variables ---
RAW_DATA_FILE = config.RAW_DATA_FILE
DATA_DIR = config.DATA_DIR

def load_data() -> pd.DataFrame:
    """
    Checks for the Online Retail II dataset, downloads it if not present, 
    and returns it as a pandas DataFrame

    Returns:
        pd.DataFrame: The loaded Online Retail II dataset.
    """

    # --- 1. Check if the dataset already exists ---
    if RAW_DATA_FILE.exists():
        logger.info(f"--- Found existing dataset at {RAW_DATA_FILE}. Loading from file. ---")
    else:
        logger.info("--- Dataset not found. Downloading from Kaggle... ---")
        
        # Ensure the target directory exists
        raw_data_dir = RAW_DATA_FILE.parent
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 2. Download the dataset to the kagglehub cache and get the path ---
        try:
            download_path_str = kagglehub.dataset_download("mashlyn/online-retail-ii-uci")
            source_csv = Path(download_path_str) / "online_retail_II.csv"
            
            # Copy the file from the cache to our local data directory
            shutil.copy(source_csv, RAW_DATA_FILE)
            logger.info(f"--- Dataset downloaded and saved to {RAW_DATA_FILE} ---")

        except Exception as e:
            logger.error(f"An error occurred during download or file operations: {e}")
            logger.error("Please ensure you are authenticated with Kaggle.")
            logger.error("See: https://www.kaggle.com/docs/api#authentication")
            return pd.DataFrame() # Return empty dataframe on failure

    # --- 3. Load and return the DataFrame ---
    try:
        # Specify encoding to handle potential special characters
        df = pd.read_csv(RAW_DATA_FILE, encoding='ISO-8859-1')
        logger.info("--- DataFrame loaded successfully. ---")
        return df
    except Exception as e:
        logger.error(f"Failed to read the CSV file: {e}")
        return pd.DataFrame()
    
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw retail dataset by renaming columns, dropping unneeded columns,
    changing data types, removing erroneous or irrelevant data, and handling missing Customer IDs.
    
    Parameters:
    df (pd.DataFrame): Raw retail dataset.
    
    Returns:
    df_clean (pd.DataFrame): Cleaned retail dataset.
    """
    # --- 1. Setup paths and check for existing cleaned dataset ---
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dataset_path = processed_dir / "cleaned_dataset.parquet"

    if cleaned_dataset_path.exists():
        logger.info(f"--- Found existing cleaned dataset. Loading from {cleaned_dataset_path} ---")
        return pd.read_parquet(cleaned_dataset_path)
    logger.info("--- Creating cleaned dataset ---")

    # --- 2. Rename columns, drop unneeded columns, change data types ---
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={'Customer ID': 'CustomerID'})
    df_clean = df_clean.drop(columns=['Description'])
    df_clean['Invoice'] = df_clean['Invoice'].astype('string')
    df_clean['StockCode'] = df_clean['StockCode'].astype('string')
    df_clean['CustomerID'] = df_clean['CustomerID'].astype('Int64').astype('string')
    df_clean['Country'] = df_clean['Country'].astype('string')
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

    # --- 3. Remove erroneous or irrelevant data ---
    # Number of rows before cleaning
    rows_before = df_clean.shape[0]
    # Remove cancelled orders
    df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
    # Remove impossible quantities and prices
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['Price'] > 0]
    # Remove duplicates
    df_clean.drop_duplicates(inplace=True)
    # Remove non-product entries (All non-product StockCodes are identified by letters only and analyzed in terms of price variation manually)
    df_clean = df_clean[~df_clean['StockCode'].isin(['POST', 'D', 'M', 'CRUK', 'S', 'BANK CHARGES', 'PADS', 'DOT', 'AMAZONFEE'])]
    # Number of rows after cleaning
    rows_after = df_clean.shape[0]
    logger.info(f'Number of rows before cleaning: {rows_before}')
    logger.info(f'Number of rows after cleaning: {rows_after}')

    # --- 4. Handle missing Customer IDs ---
    df_clean['CustomerID'] = df_clean['CustomerID'].fillna('Unknown')

    # --- 5. Save the cleaned dataset ---
    logger.info(f"\n--- Saving cleaned dataset to {cleaned_dataset_path} ---")
    df_clean.to_parquet(cleaned_dataset_path, index=False, engine='pyarrow')

    return df_clean

def create_interim_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Create an interim weekly aggregated dataset from the cleaned transactional data.

    Args:
    df_clean (pd.DataFrame): Cleaned transactional data with necessary columns.
    
    Returns:
    pd.DataFrame: Weekly aggregated dataset with all products and weeks.
    """
    # --- 1. Setup paths and check for existing interim dataset ---
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dataset_path = processed_dir / "interim_dataset.parquet"

    if interim_dataset_path.exists():
        logger.info(f"--- Found existing interim dataset. Loading from {interim_dataset_path} ---")
        return pd.read_parquet(interim_dataset_path)
    logger.info("--- Creating interim dataset ---")

    # --- 2. Transform Datetime to Week format (week starts on Monday) ---
    df_temp = df_clean.copy()
    df_temp['NormalizedDate'] = df_temp['InvoiceDate'].dt.normalize()
    df_temp['Week'] = df_temp['NormalizedDate'] - pd.to_timedelta(df_temp['NormalizedDate'].dt.weekday, unit='D')
    df_temp = df_temp.drop(columns=['InvoiceDate', 'NormalizedDate'])

    # --- 3. Create complete grid of all products and all weeks in the dataset ---
    all_products = df_temp['StockCode'].unique().tolist()
    all_weeks = pd.date_range(
        start=df_temp['Week'].min(),
        end=df_temp['Week'].max(),
        freq='W-MON'
    )
    complete_grid = pd.MultiIndex.from_product(
        [all_products, all_weeks],
        names=['StockCode', 'Week']
    ).to_frame(index=False)

    # --- 4. Aggregate sales data by StockCode and Week ---
    aggregated_sales = df_temp.groupby(['StockCode', 'Week']).agg(
        TotalSales=('Quantity', 'sum'),
        AvgPrice=('Price', 'mean'),
        n_customers_this_week=('CustomerID', 'nunique'),
        most_frequent_country=('Country', lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    ).reset_index()
    aggregated_sales.head()

    # --- 5. Merge complete grid with aggregated sales data ---
    df_weekly = pd.merge(
        complete_grid,
        aggregated_sales,
        on=['StockCode', 'Week'],
        how='left'
    )

    # --- 6. Handle missing values and data types ---
    df_weekly['TotalSales'] = df_weekly['TotalSales'].fillna(0)
    df_weekly['TotalSales'] = df_weekly['TotalSales'].astype(int)
    df_weekly['n_customers_this_week'] = df_weekly['n_customers_this_week'].fillna(0)
    df_weekly['n_customers_this_week'] = df_weekly['n_customers_this_week'].astype(int)
    df_weekly['AvgPrice'] = df_weekly.groupby('StockCode')['AvgPrice'].ffill()
    df_weekly['AvgPrice'] = df_weekly.groupby('StockCode')['AvgPrice'].bfill()
    df_weekly['AvgPrice'] = df_weekly['AvgPrice'].round(2)

    # --- 7. Identify active vs padded periods per product ---
    ## Active period is between the first and last observed sale
    logger.info("Identifying active vs padded weeks per StockCode...")
    sales_mask = df_weekly["TotalSales"] > 0
    first_sale_week = df_weekly.loc[sales_mask].groupby('StockCode')['Week'].min().rename('first_sale_week')
    
    # Merge back into df_weekly
    df_weekly = df_weekly.merge(first_sale_week, on='StockCode', how='left')

    # Create active mask
    df_weekly['is_active'] = df_weekly['Week'] >= df_weekly['first_sale_week']

    # Create padding mask
    df_weekly['is_padding'] = ~df_weekly['is_active']

    # Create numeric weight for losses (1.0 - active, 0.0 - padding)
    df_weekly['loss_weight'] = df_weekly['is_active'].astype('float32')

    # Drop cols
    df_weekly = df_weekly.drop(columns=['first_sale_week'])

    # --- 8. Order correctly ---
    logger.info("Sorting dataset by StockCode and Week...")
    df_weekly = df_weekly.sort_values(['StockCode', 'Week']).reset_index(drop=True)

    # --- 9. Save the final object ---
    logger.info(f"\n--- Saving segmented splits to {interim_dataset_path} ---")
    df_weekly.to_parquet(interim_dataset_path, index=False, engine='pyarrow')

    return df_weekly

def _find_first_price(df: pd.DataFrame) -> pd.Series:
    """
    Finds the first observed 'AvgPrice' for each product in the given dataframe.
    Support function for create_splits.

    Args:
        df (pd.DataFrame): DataFrame containing 'StockCode', 'Week', and 'AvgPrice' columns.

    Returns:
        pd.Series: A Series mapping 'StockCode' to its first observed 'AvgPrice'.
    """
    logger.info("Calculating first observed price for all products...")
    # --- 1. Find the index of the first week for each product ---
    observed_df = df[df['TotalSales'] > 0]
    first_week_indices = observed_df.loc[observed_df.groupby('StockCode')['Week'].idxmin()]
    # --- 2. Create the map: StockCode -> First AvgPrice ---
    first_price_map = first_week_indices.set_index('StockCode')['AvgPrice']
    logger.info("First price map created.")
    return first_price_map

def _impute_future_price(splits: dict, first_price_map: pd.Series):
    """
    Imputes 'AvgPrice' values in validation and test sets using.
    For Established products, uses the average price from the training set by product.
    For Cold-Start products, uses the first observed price from the entire dataset.
    Support function for create_splits.

    Args:
        splits (dict): The dictionary containing all pre-defined data splits.
        first_price_map (pd.Series): A Series mapping 'StockCode' to its first observed 'AvgPrice'.

    Returns:
        dict: The updated splits dictionary with imputed 'AvgPrice' in validation and test sets
    """
    logger.info("--- Applying 'future price' imputation to validation and test sets ---")

    # --- 1. Define which sets need imputation based on which training set ---
    folds_to_impute = {
        "fold1": {"train": splits['fold1']['train'], "impute_sets": [splits['fold1']['validation'], splits['fold1']['test']]},
        "fold2": {"train": splits['fold2']['train'], "impute_sets": [splits['fold2']['validation'], splits['fold2']['test']]},
        "hold_out": {"train": splits['hold_out']['train'], "impute_sets": [splits['hold_out']['validation'], splits['hold_out']['test']]}
    }

    # --- 2. Impute the data ---
    for fold_name, fold_data in folds_to_impute.items():
        train_df = fold_data["train"]
        active_train_df = train_df[train_df['TotalSales'] > 0] # Only consider weeks with sales
        logger.info(f"Calculating imputation values for {fold_name} based on its training set...")
        
        # --- 2.1 Find the average price for each product in the training set ---
        avg_price_map = active_train_df.groupby('StockCode')['AvgPrice'].mean()
        
        # --- 2.2 Apply imputation to both validation and test sets for this fold ---
        for target_df in fold_data["impute_sets"]:
            df_name = [k for k,v in locals().items() if v is target_df][0] # Get variable name for logging
            
            # --- 2.2.1 Impute Established products ---
            # Map average price. Cold-start products get NaN.
            target_df['AvgPrice_Imputed'] = target_df['StockCode'].map(avg_price_map)
            cold_start_products = target_df[target_df['AvgPrice_Imputed'].isna()]['StockCode'].unique()
            imputed_products = target_df[target_df['AvgPrice_Imputed'].notna()]['StockCode'].unique()
            logger.debug(f"Contains {len(cold_start_products)} cold-start products needing first price imputation.")
            logger.debug(f"Current average price for Established products: {target_df[target_df['StockCode'].isin(imputed_products)]['AvgPrice_Imputed'].mean():.2f}")
            
            # --- 2.2.2 Impute Cold-Start products ---
            # Map the pre-calculated first price to cold-start products
            cold_start_mask = target_df['AvgPrice_Imputed'].isna()
            target_df.loc[cold_start_mask, 'AvgPrice_Imputed'] = target_df.loc[cold_start_mask, 'StockCode'].map(first_price_map)
            logger.debug(f"Contains {len(target_df[target_df['AvgPrice_Imputed'].isna()]['StockCode'].unique())} NaNs after imputation.")
            logger.debug(f"Current average price for Established products: {target_df[target_df['StockCode'].isin(imputed_products)]['AvgPrice_Imputed'].mean():.2f}")
            logger.debug(f"Current average price for Cold-Start products: {target_df[target_df['StockCode'].isin(cold_start_products)]['AvgPrice_Imputed'].mean():.2f}")
            
            # --- 2.2.3 Overwrite the original 'AvgPrice' column with the imputed values ---
            target_df['AvgPrice'] = target_df['AvgPrice_Imputed'].astype(np.float32).round(2)
            target_df.drop(columns=['AvgPrice_Imputed'], inplace=True)
            logger.info(f"Finished imputing for {fold_name} - {df_name}.")

    return splits

def _add_static_dynamic_features(splits: dict) -> dict:
    """
    Adds static and dynamic features to each dataframe within the splits dictionary.
    Support function for create_splits.

    Args:
        splits (dict): The dictionary containing all pre-defined data splits.

    Returns:
        dict: The updated splits dictionary with added features.
    """
    logger.info("--- Adding static and dynamic features to all sets ---")

    for fold_name, fold_data in splits.items():
        logger.info(f"--- Performing feature engineering for {fold_name} ---")
        
        # --- 1. Create Static Features from the Training Set ---
        train_df = fold_data['train'].copy()
        active_train_df = train_df[train_df['TotalSales'] > 0]
        static_features = active_train_df.groupby('StockCode').agg(
            static_most_frequent_country=('most_frequent_country', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
            avg_weekly_customers=('n_customers_this_week', 'mean'),
            std_weekly_customers=('n_customers_this_week', 'std')
        ).reset_index()
        static_features['std_weekly_customers'] = static_features['std_weekly_customers'].fillna(0) # Handle NaN std

        # Define the minimum date for the fold
        min_date = fold_data['train']['Week'].min()
        
        # --- 2. Attach static and dynamic features to all sets in the fold ---
        for split_type, split_data in fold_data.items():
            # --- 2.1 Attach static features ---
            split_data = pd.merge(split_data, static_features, on='StockCode', how='left')
            split_data = split_data.drop(columns=['most_frequent_country', 'n_customers_this_week'])

            # --- 2.2 Handle missing static features for cold-start products
            split_data['avg_weekly_customers'] = split_data['avg_weekly_customers'].fillna(0)
            split_data['std_weekly_customers'] = split_data['std_weekly_customers'].fillna(0)
            split_data['static_most_frequent_country'] = split_data['static_most_frequent_country'].fillna("Unknown")

            # --- 2.3 Enforce Correct Data Types ---
            # Enforce string types for categoricals
            split_data['StockCode'] = split_data['StockCode'].astype(str)
            split_data['static_most_frequent_country'] = split_data['static_most_frequent_country'].astype(str)

            # Enforce float types for reals
            split_data['TotalSales'] = split_data['TotalSales'].astype(np.float32)
            split_data['AvgPrice'] = split_data['AvgPrice'].astype(np.float32)
            split_data['avg_weekly_customers'] = split_data['avg_weekly_customers'].astype(np.float32)
            split_data['std_weekly_customers'] = split_data['std_weekly_customers'].astype(np.float32)

            # --- 2.4 Create new dynamic time features ---
            # Track the time index as weeks
            split_data['time_idx'] = (split_data['Week'] - min_date).dt.days // 7
            # Add cyclical week of year features
            week = split_data['Week'].dt.isocalendar().week
            split_data['week_sin'] = np.sin(2 * np.pi * week / 52.0)
            split_data['week_cos'] = np.cos(2 * np.pi * week / 52.0)
            # Add cyclical month of year features
            month = split_data['Week'].dt.month
            split_data['month_of_year'] = month
            split_data['month_sin'] = np.sin(2 * np.pi * month / 12.0)
            split_data['month_cos'] = np.cos(2 * np.pi * month / 12.0)          
            split_data['year'] = split_data['Week'].dt.year.astype(str)

            # Ensure the updated split is stored back
            splits[fold_name][split_type] = split_data

    return splits

def _calculate_volatility_map(training_df):
    """
    Support function to calculate CoV and create a volatility map from a given training dataframe for create_splits.

    Args:
        training_df (pd.DataFrame): The training dataframe containing 'StockCode' and 'TotalSales' columns.

    Returns:
        pd.Series: A Series mapping 'StockCode' to its 'VolatilityLabel'.
    """
    # --- 1. Filter for products with sales before calculating stats ---
    products_with_sales = training_df[training_df['TotalSales'] > 0]

    # --- 2. Group by StockCode and calculate mean and standard deviation of weekly sales ---
    product_stats = products_with_sales.groupby('StockCode')['TotalSales'].agg(['mean', 'std']).reset_index()
    
    # Correcting possible NaN and 0 values causing troubles
    product_stats['std'] = product_stats['std'].fillna(0)
    product_stats['CoV'] = product_stats['std'] / (product_stats['mean'] + 1e-6) # Small epsilon to avoid division by zero

    # --- 3. Set fixed thresholds for CoV ---
        # Low Volatility: CoV < 0.5
        # Moderate Volatility: 0.5 <= CoV <= 1.0
        # High Volatility: CoV > 1.0
    bins = [-np.inf, 0.5, 1.0, np.inf]
    labels = ['Low Volatility', 'Moderate Volatility', 'High Volatility']
    
    # --- 4. Create the 'VolatilityLabel' column ---
    product_stats['VolatilityLabel'] = pd.cut(
        product_stats['CoV'],
        bins=bins,
        labels=labels
    )
    
    # Return a map (a pandas Series) for easy application
    return product_stats.set_index('StockCode')['VolatilityLabel']

def _add_segmentation(splits: dict) -> dict:
    """
    Adds 'ProductStatus' and 'VolatilityLabel' columns to each set within the splits dictionary.
    Support function for create_splits. 

    Args:
        splits (dict): The dictionary containing all pre-defined data splits.

    Returns:
        dict: The updated splits dictionary with fully segmented validation and test sets.
    """
    logger.info("--- Adding segmentation to all sets ---")

    # --- 1, Define fold 1 training ---
    reference_set = splits['fold1']['train'].copy()

    # --- 2. Calculate volatility map based on the training set ---
    volatility_map = _calculate_volatility_map(reference_set)
    logger.info(f"Volatility map created")
    
    # --- 3. Identify established products ---
    established_products = set(reference_set[reference_set['TotalSales'] > 0]['StockCode'].unique())
    logger.info(f"Found {len(established_products)} established products using Fold 1 training base.")

    for fold_name, fold_data in splits.items():
        # --- 4. Filter and Segment the sets ---
        for split_type, split_data in fold_data.items():
            # --- 4.1 Cold-Start Segmentation ---
            split_products = set(split_data['StockCode'].unique())
            cold_start_products = split_products - established_products
            split_data.loc[:, 'ProductStatus'] = split_data['StockCode'].apply(lambda x: 'Cold-Start' if x in cold_start_products else 'Established')
            logger.info(f"[{fold_name.upper()}] {split_type} segmented. Found {len(cold_start_products)} cold-start products.")

            # --- 4.2 Volatility Segmentation ---
            mapped_labels = split_data['StockCode'].map(volatility_map)
            if 'Unknown' not in mapped_labels.dtype.categories:
                mapped_labels = mapped_labels.cat.add_categories(['Unknown'])
            split_data.loc[:, 'VolatilityLabel'] = mapped_labels.fillna('Unknown')
            # Compute and output volatility distribution in the set
            active_products = split_data[split_data['TotalSales'] > 0][['StockCode', 'VolatilityLabel']].drop_duplicates()
            volatility_counts = active_products['VolatilityLabel'].value_counts()
            low = volatility_counts.get('Low Volatility', 0)
            mod = volatility_counts.get('Moderate Volatility', 0)
            high = volatility_counts.get('High Volatility', 0)
            unknown = volatility_counts.get('Unknown', 0)
            logger.info(f"    -> Volatility (Active Products): Low={low}, Moderate={mod}, High={high}, Unknown={unknown}")

            # --- 4.3 Save back the segmented set ---
            splits[fold_name][split_type] = split_data

    return splits


def create_splits(df_weekly: pd.DataFrame) -> dict:
    """
    Splits the weekly sales data into two cross-validation folds and a final hold-out set.
    Imputes future prices, adds static/dynamic features, and segments products.
    The strategy is as follows:
    - Hold-out: Train (21m), Test (3m)
    - Fold 1: Train (12m), Validation (3m), Test (3m)
    - Fold 2 (Expanding Window): Train (15m), Validation (3m), Test (3m)

    Args:
        df_weekly (pd.DataFrame): The preprocessed weekly data with a 'Week' column.

    Returns:
        dict: A dictionary containing all the dataframe splits.
    """
    # --- 1. Setup paths and check for existing segmented splits ---
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    segmented_splits_path = processed_dir / "segmented_splits.parquet"

    if segmented_splits_path.exists():
        logger.info(f"--- Found existing segmented splits. Loading from {segmented_splits_path} ---")
        df = pd.read_parquet(segmented_splits_path)
        # Ensure categorical types
        df['fold'] = df['fold'].astype('category')
        df['split_type'] = df['split_type'].astype('category')

        # --- 2. Recreate splits dictionary from parquet ---
        splits = {}
        for fold_name in df['fold'].cat.categories:
            splits[fold_name] = {}
            fold_df = df[df['fold'] == fold_name]
            for split_type in fold_df['split_type'].cat.categories:
                # Drop the helper columns before storing
                split_df = fold_df[fold_df['split_type'] == split_type].drop(columns=['fold', 'split_type']).copy()
                splits[fold_name][split_type] = split_df
        logger.info("--- Reconstructed splits dictionary from Parquet file. ---")
        return splits



    logger.info("--- Creating data splits ---")

    # --- 2. Define the main date boundaries based on the data's start date ---
    start_date = df_weekly['Week'].min()
    
    ## Define the end points for each period by adding months to the start date
    # Fold 1 boundaries (53w train, 13w val, 13w test)
    train_end_f1 = start_date + pd.DateOffset(weeks=53) # Due to the existing data, exactly one year of observations here is 53 weeks.
    val_end_f1 = start_date + pd.DateOffset(weeks=66) # 53 + 13
    test_end_f1 = start_date + pd.DateOffset(weeks=79) # 66 + 13
    
    # Fold 2 boundaries (66w train, 13w val, 13w test)
    train_end_f2 = start_date + pd.DateOffset(weeks=66)
    val_end_f2 = start_date + pd.DateOffset(weeks=79) # 66 + 13
    test_end_f2 = start_date + pd.DateOffset(weeks=92) # 79 + 13
    
    # Hold-out set boundary (106 weeks in total, so not exactly expanding nature to fit all data)
    hold_out_train_end = start_date + pd.DateOffset(weeks=80) # 93 - 13
    hold_out_val_end = start_date + pd.DateOffset(weeks=93) # 106 - 13
    
    # --- 3. Create the splits for Fold 1 ---
    train_fold1 = df_weekly[df_weekly['Week'] < train_end_f1].copy()
    val_fold1 = df_weekly[(df_weekly['Week'] >= train_end_f1) & (df_weekly['Week'] < val_end_f1)].copy()
    test_fold1 = df_weekly[(df_weekly['Week'] >= val_end_f1) & (df_weekly['Week'] < test_end_f1)].copy()
    
    # --- 4. Create the splits for Fold 2 ---
    train_fold2 = df_weekly[df_weekly['Week'] < train_end_f2].copy()
    val_fold2 = df_weekly[(df_weekly['Week'] >= train_end_f2) & (df_weekly['Week'] < val_end_f2)].copy()
    test_fold2 = df_weekly[(df_weekly['Week'] >= val_end_f2) & (df_weekly['Week'] < test_end_f2)].copy()
    
    # --- 5. Create the final hold-out set ---
    train_hold_out = df_weekly[df_weekly['Week'] < hold_out_train_end].copy()
    val_hold_out = df_weekly[(df_weekly['Week'] >= hold_out_train_end) & (df_weekly['Week'] < hold_out_val_end)].copy() # DL models require validation
    test_hold_out = df_weekly[df_weekly['Week'] >= hold_out_val_end].copy()

    # --- 6. Store splits in a dictionary for easy access ---
    splits = {
        "fold1": {"train": train_fold1, "validation": val_fold1, "test": test_fold1},
        "fold2": {"train": train_fold2, "validation": val_fold2, "test": test_fold2},
        "hold_out": {"train": train_hold_out, "validation": val_hold_out, "test": test_hold_out}
    }
    
    # --- 7. Impute future prices for validation and test sets ---
    first_price_map = _find_first_price(df_weekly)
    splits = _impute_future_price(splits, first_price_map)

    # --- 8. Add static and dynamic features to all sets ---
    splits = _add_static_dynamic_features(splits)

    # --- 9. Add segmentation to all sets ---
    splits = _add_segmentation(splits)

    # --- 10. Print a summary to verify the splits ---
    logger.info("--- Data Splitting Summary ---")
    for fold_name, fold_data in splits.items():
        logger.info(f"\n[{fold_name.upper()}]")
        for split_name, df_split in fold_data.items():
            start = df_split['Week'].min().strftime('%Y-%m-%d')
            end = df_split['Week'].max().strftime('%Y-%m-%d')
            price_mean = df_split['AvgPrice'].mean() if 'AvgPrice' in df_split.columns and not df_split['AvgPrice'].isnull().all() else 'N/A'
            volume_mean = df_split['TotalSales'].mean() if 'TotalSales' in df_split.columns and not df_split['TotalSales'].isnull().all() else 'N/A'
            price_median = df_split['AvgPrice'].median() if 'AvgPrice' in df_split.columns and not df_split['AvgPrice'].isnull().all() else 'N/A'
            price_mode = df_split['AvgPrice'].mode().iloc[0] if 'AvgPrice' in df_split.columns and not df_split['AvgPrice'].isnull().all() else 'N/A'
            logger.info(f"  - {split_name.capitalize():<12}: {start} to {end} ({len(df_split)} rows) - AvgPrice Mean: {price_mean if price_mean == 'N/A' else f'{price_mean:.2f}'}, TotalSales Mean: {volume_mean if volume_mean == 'N/A' else f'{volume_mean:.2f}'}")
            logger.debug(f"                     Median: {price_median if price_median == 'N/A' else f'{price_median:.2f}'} - Mode: {price_mode if price_mode == 'N/A' else f'{price_mode:.2f}'}")

    # --- 11. Save the final object ---
    logger.info(f"\n--- Saving segmented splits to {segmented_splits_path} ---")
    all_dfs = []
    for fold_name, fold_data in splits.items():
        for split_type, df_split in fold_data.items():
            df_copy = df_split.copy()
            df_copy['fold'] = fold_name
            df_copy['split_type'] = split_type
            all_dfs.append(df_copy)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df['fold'] = final_df['fold'].astype('category')
    final_df['split_type'] = final_df['split_type'].astype('category')

    final_df.to_parquet(segmented_splits_path, index=False, engine='pyarrow')

    return splits

def preprocess_data():
    # --- 1. Load Dataset ---
    df = load_data()
    # --- 2. Clean Dataset ---
    df_clean = clean_data(df)
    # --- 3. Transform Dataset ---
    df_weekly = create_interim_dataset(df_clean)
    # --- 4. Create Data Splits
    data_splits = create_splits(df_weekly)

    return data_splits

if __name__ == "__main__":
    pass



