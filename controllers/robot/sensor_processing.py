import numpy as np
import cv2
import math
from typing import List
from constants import LOWER_LIMIT, UPPER_LIMIT, DIST_NEAR, ANGLE_FRONT


def probTargetVisible(image) -> float:
    image = np.frombuffer(image, np.uint8).reshape((40, 200, 4))
    image = image.copy()
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvImage, LOWER_LIMIT, UPPER_LIMIT)
    mask = cv2.resize(mask, None, fx=3, fy=3)
    cv2.imshow("Webots Camera", mask)
    yellow_ratio = np.sum(mask > 0) / mask.size
    yellow_ratio = float(yellow_ratio)
    print(f"Proporção de pixels amarelos: {yellow_ratio:.2%}")
    prob = 1 - np.power(2, -25 * yellow_ratio)
    return prob


def getAngle(robot_node, target) -> float:
    obs_pos = target.getPosition()
    rob_pos = robot_node.getPosition()
    rob_rot = robot_node.getOrientation()
    rob_yaw = math.atan2(-rob_rot[1], rob_rot[0])
    dx = obs_pos[0] - rob_pos[0]
    dy = obs_pos[1] - rob_pos[1]
    angle = math.atan2(dy, dx) - rob_yaw
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return -angle


def GPS(robot_node, lidar_data, target):
    min_dist = min(lidar_data)
    min_index = lidar_data.index(min_dist)
    angle = getAngle(robot_node, target)
    if min_dist < DIST_NEAR and min_index < len(lidar_data) // 2:
        min_dist = -min_dist
    return min_dist, angle
