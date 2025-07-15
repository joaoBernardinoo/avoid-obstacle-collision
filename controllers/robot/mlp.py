import joblib
from constants import MLP_PATH, LIDAR_MAX_RANGE
from sklearn.neural_network import MLPClassifier
import numpy as np


def load_mlp(MLP_PATH):
    """
    Loads a saved MLP model from disk.

    Parameters
    ----------
    MLP_PATH : str
        Path to the saved model.

    Returns
    -------
    mlp : MLPClassifier
        The loaded MLP model.
    """
    return joblib.load("mlp.pkl")


_mlp: MLPClassifier = load_mlp(MLP_PATH)


def MLP(arr):
    """
    Process a LiDAR array and use a trained MLP model to predict the index
    of the minimum absolute value. The LiDAR array is first processed to
    replace non-finite values with LIDAR_MAX_RANGE and then split in half.
    The first half is forced negative and the second half is forced positive.
    The absolute minimum value is then used to predict the index using the
    MLP model.
    
    Parameters:
    arr (list or array): LiDAR array to process.
    
    Returns:
    int: Index of the minimum absolute value.
    """
    lidar = np.array(arr, dtype=np.float64)
    
    lidar = np.where(np.isfinite(lidar), lidar, LIDAR_MAX_RANGE)
    
    lidar = np.nan_to_num(lidar, nan=LIDAR_MAX_RANGE, posinf=LIDAR_MAX_RANGE, neginf=LIDAR_MAX_RANGE)
    
    half = len(lidar) // 2
    lidar_processed = np.concatenate([
        -np.abs(lidar[:half]),  # First half: force negative
        np.abs(lidar[half:])     # Second half: force positive
    ])
    
    # Predict using the MLP model
    idx = _mlp.predict([lidar_processed])[0]
    abs_min = lidar_processed[idx]
    
    return abs_min