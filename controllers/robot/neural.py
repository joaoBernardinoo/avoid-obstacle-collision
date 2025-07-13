import joblib
from constants import MLP_PATH, LIDAR_MAX_RANGE
from sklearn.neural_network import MLPClassifier
import numpy as np

def load_mlp(MLP_PATH):
    return joblib.load("mlp.pkl")

_mlp: MLPClassifier = load_mlp(MLP_PATH)

def MLP(arr):
    lidar_np = np.array(arr)
    lidar_np = np.clip(lidar_np, 0, LIDAR_MAX_RANGE)
    return _mlp.predict([lidar_np])