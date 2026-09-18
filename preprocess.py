import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)


def impute_missing_values(df: pd.DataFrame, strategy: str = 'interpolate', group_col: Optional[str] = 'sat_id') -> pd.DataFrame:
    """
    Imputes missing values in the telemetry DataFrame.
    Designed for time-series data, it interpolates missing points within each satellite's trajectory.

    Args:
        df (pd.DataFrame): The input DataFrame containing orbital data.
        strategy (str): Imputation strategy. Options: 'interpolate', 'ffill', 'bfill'. Default is 'interpolate'.
        group_col (Optional[str]): Column to group by (typically 'sat_id') to prevent 
                                   cross-pollination of data between different satellites.

    Returns:
        pd.DataFrame: A new DataFrame with missing values imputed.
        
    Raises:
        ValueError: If an unsupported imputation strategy is provided.
    """
    logger.info(f"Imputing missing values using strategy: {strategy}")
    df_imputed = df.copy()
    
    if strategy not in ['interpolate', 'ffill', 'bfill']:
        raise ValueError(f"Unsupported imputation strategy: {strategy}")

    # Identify numeric columns for imputation
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

    if group_col and group_col in df_imputed.columns:
        logger.debug(f"Applying grouped imputation by {group_col}")
        if strategy == 'interpolate':
            df_imputed[numeric_cols] = df_imputed.groupby(group_col)[numeric_cols].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
        elif strategy == 'ffill':
            df_imputed[numeric_cols] = df_imputed.groupby(group_col)[numeric_cols].transform(lambda x: x.ffill().bfill())
        elif strategy == 'bfill':
            df_imputed[numeric_cols] = df_imputed.groupby(group_col)[numeric_cols].transform(lambda x: x.bfill().ffill())
    else:
        logger.warning(f"No grouping column '{group_col}' found. Imputing globally across all rows.")
        if strategy == 'interpolate':
            df_imputed[numeric_cols] = df_imputed[numeric_cols].interpolate(method='linear', limit_direction='both')
        elif strategy == 'ffill':
            df_imputed[numeric_cols] = df_imputed[numeric_cols].ffill().bfill()
        elif strategy == 'bfill':
            df_imputed[numeric_cols] = df_imputed[numeric_cols].bfill().ffill()

    # Drop any remaining NaNs (e.g., if a group was entirely NaN)
    missing_after = df_imputed[numeric_cols].isna().sum().sum()
    if missing_after > 0:
        logger.warning(f"Dropping rows due to {missing_after} un-imputable missing values.")
        df_imputed.dropna(subset=numeric_cols, inplace=True)
        
    return df_imputed


def normalize_features(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalizes specified numeric features using Z-score normalization (StandardScaler).

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_cols (List[str]): List of column names to normalize.
        scaler (Optional[StandardScaler]): An existing fitted scaler. If None, a new one is fitted.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: The normalized DataFrame and the fitted scaler object.
        
    Raises:
        KeyError: If any of the specified feature columns are missing from the DataFrame.
    """
    logger.info(f"Normalizing features: {feature_cols}")
    df_norm = df.copy()
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns for normalization: {missing_cols}")
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    if scaler is None:
        scaler = StandardScaler()
        df_norm[feature_cols] = scaler.fit_transform(df_norm[feature_cols])
        logger.info("Fitted a new StandardScaler.")
    else:
        df_norm[feature_cols] = scaler.transform(df_norm[feature_cols])
        logger.info("Applied existing StandardScaler.")
        
    return df_norm, scaler


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    window_size: int,
    horizon: int = 1,
    group_col: Optional[str] = 'sat_id'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates rolling-window sequences for time-series forecasting (e.g., LSTM input).
    Ensures that sequences do not cross boundaries between different satellites.

    Args:
        df (pd.DataFrame): The input DataFrame, chronologically sorted.
        feature_cols (List[str]): Columns to use as input features (X).
        target_cols (List[str]): Columns to predict (y).
        window_size (int): Number of historical time steps in each input sequence.
        horizon (int): Number of time steps ahead to predict. Defaults to 1.
        group_col (Optional[str]): Column indicating independent trajectories (e.g., satellite ID).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - X: 3D numpy array of shape (num_samples, window_size, num_features)
            - y: 2D numpy array of shape (num_samples, num_targets)
            
    Raises:
        ValueError: If window_size or horizon are invalid, or if there's insufficient data.
    """
    if window_size <= 0 or horizon <= 0:
        raise ValueError("window_size and horizon must be strictly positive integers.")

    logger.info(f"Creating sequences. Window size: {window_size}, Horizon: {horizon}")
    
    X_list, y_list = [], []
    
    # Process each satellite's trajectory independently to prevent data leakage
    groups = df.groupby(group_col) if group_col and group_col in df.columns else [(None, df)]
    
    for name, group in groups:
        # Ensure chronological order
        if 'epoch' in group.columns:
            group = group.sort_values(by='epoch')
            
        features = group[feature_cols].values
        targets = group[target_cols].values
        
        n_samples = len(features)
        
        if n_samples < window_size + horizon:
            logger.debug(f"Group {name} has insufficient data ({n_samples} rows) for window {window_size}.")
            continue
            
        # Create sliding windows
        for i in range(n_samples - window_size - horizon + 1):
            X_list.append(features[i : i + window_size])
            y_list.append(targets[i + window_size + horizon - 1])
            
    if not X_list:
        raise ValueError("No sequences could be created. Insufficient data length for the given window_size.")
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    logger.info(f"Sequence creation complete. Generated X shape: {X.shape}, y shape: {y.shape}")
    return X, y
