import joblib
from constants import MLP_PATH, LIDAR_MAX_RANGE
from sklearn.neural_network import MLPClassifier
import numpy as np


def load_mlp(MLP_PATH):
    return joblib.load("mlp.pkl")


_mlp: MLPClassifier = load_mlp(MLP_PATH)


def MLP(arr):
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