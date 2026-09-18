import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics (MAE and RMSE) for the forecasted coordinates.

    Args:
        y_true (np.ndarray): Ground truth coordinates (e.g., shape (N, 3)).
        y_pred (np.ndarray): Predicted coordinates from the model (e.g., shape (N, 3)).

    Returns:
        Dict[str, float]: Dictionary containing the 'MAE' and 'RMSE' scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    logger.info(f"Evaluation Metrics Computed - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return {"MAE": mae, "RMSE": rmse}

def detect_close_approaches(
    predicted_positions: Dict[str, np.ndarray], 
    threshold_km: float = 10.0
) -> List[Tuple[str, str, float]]:
    """
    Calculates closest approach Euclidean distances between satellites/debris to flag collision risks.
    Uses efficient vectorized pairwise distance calculations.

    Args:
        predicted_positions (Dict[str, np.ndarray]): A dictionary mapping satellite IDs to their 
                                                     predicted [x, y, z] coordinate arrays at a 
                                                     specific future timestep.
        threshold_km (float): The minimum safety distance in kilometers. Anything below this 
                              flags an alert. Default is 10.0 km.

    Returns:
        List[Tuple[str, str, float]]: A list of alerts containing (sat_id_1, sat_id_2, distance_km).
    """
    sat_ids = list(predicted_positions.keys())
    
    # Extract coordinates into an (N, 3) matrix
    try:
        coords = np.array(list(predicted_positions.values()), dtype=np.float64)
    except Exception as e:
        logger.error("Failed to parse predicted positions into coordinate matrix.")
        raise ValueError("predicted_positions must contain flat arrays of coordinates.") from e

    if len(sat_ids) < 2:
        logger.warning("Fewer than 2 objects provided; no collisions possible.")
        return []

    # Calculate pairwise Euclidean distances for all objects
    distances = pdist(coords, metric='euclidean')
    dist_matrix = squareform(distances)

    alerts = []
    num_objects = len(sat_ids)
    
    # Iterate through the upper triangle of the distance matrix to avoid duplicate pairs
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            dist = dist_matrix[i, j]
            if dist <= threshold_km:
                alerts.append((sat_ids[i], sat_ids[j], float(dist)))
                logger.warning(
                    f"COLLISION RISK ALERT: '{sat_ids[i]}' and '{sat_ids[j]}' "
                    f"projected to be {dist:.2f} km apart!"
                )

    if not alerts:
        logger.info(f"All clear. No objects projected within {threshold_km} km threshold.")
        
    return alerts
