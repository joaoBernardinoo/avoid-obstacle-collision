import numpy as np
import h5py
import os
from typing import cast
from constants import SAVE_PATH, HDF5_SAVE_PATH


def collectDataHDF5(dist, angle, lidar_data, camera_data, cam_w, cam_h):
    image_np = np.frombuffer(camera_data, np.uint8).reshape((cam_h, cam_w, 4))
    lidar_np = np.array(lidar_data)

    with h5py.File(HDF5_SAVE_PATH, 'a') as hf:
        if 'camera_image' not in hf:
            hf.create_dataset('camera_image', data=[image_np],
                              compression="gzip", chunks=True,
                              maxshape=(None, cam_h, cam_w, 4))
            hf.create_dataset('lidar_data', data=[lidar_np],
                              compression="gzip", chunks=True,
                              maxshape=(None, len(lidar_np)))
            hf.create_dataset('dist', data=[dist],
                              compression="gzip", chunks=True,
                              maxshape=(None,))
            hf.create_dataset('angle', data=[angle],
                              compression="gzip", chunks=True,
                              maxshape=(None,))
        else:
            camera_dset = cast(h5py.Dataset, hf['camera_image'])
            camera_dset.resize((camera_dset.shape[0] + 1), axis=0)
            camera_dset[-1] = image_np

            lidar_dset = cast(h5py.Dataset, hf['lidar_data'])
            lidar_dset.resize((lidar_dset.shape[0] + 1), axis=0)
            lidar_dset[-1] = lidar_np

            dist_dset = cast(h5py.Dataset, hf['dist'])
            dist_dset.resize((dist_dset.shape[0] + 1), axis=0)
            dist_dset[-1] = dist

            angle_dset = cast(h5py.Dataset, hf['angle'])
            angle_dset.resize((angle_dset.shape[0] + 1), axis=0)
            angle_dset[-1] = angle

    return 0
