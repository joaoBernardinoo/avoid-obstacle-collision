import os
import sys
from pathlib import Path
import numpy as np
import cv2
from typing import List

STEP_COUNT = 0
if sys.platform == 'linux':
    SAVE_PATH = Path('/home/dino/SSD/cnn_dataset')
else:
    SAVE_PATH = Path(os.path.dirname(__file__), 'cnn_dataset')

HDF5_SAVE_PATH = SAVE_PATH.parent / 'cnn_dataset.h5'

YELLOW = [0, 255, 255]

def get_limits(color_bgr: List[int]):
    c = np.array([[color_bgr]], dtype=np.uint8)
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]
    lower_limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    upper_limit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    return lower_limit, upper_limit

LOWER_LIMIT, UPPER_LIMIT = get_limits(color_bgr=YELLOW)

REPULSE = "cos"
VISION = True
DIST_NEAR = 0.6
MODE = "online"
ANGLE_FRONT = 0.0218
