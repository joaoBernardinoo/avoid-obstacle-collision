import joblib
from constants import MLP_PATH, LIDAR_MAX_RANGE
from sklearn.neural_network import MLPClassifier
import numpy as np


def load_mlp(MLP_PATH):
    return joblib.load("mlp.pkl")


_mlp: MLPClassifier = load_mlp(MLP_PATH)


def MLP(arr):
    lidar = np.array(arr, dtype=np.float32)
    lidar[np.isinf(lidar)] = LIDAR_MAX_RANGE
    idx = _mlp.predict([lidar])[0]
    return lidar[idx]
