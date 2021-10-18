import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import os
import seaborn as sns
from DataHandler import DataHandler
from scipy import stats
from pyM2aia import M2aiaOnlineHelper

from frameworks.VoxelmorphTF import VoxelmorphTF as vxmtf


def calculate_tre(moving_landmarks: np.array,
                  moved_landmarks: np.array) -> np.array:
    differences = moved_landmarks - moving_landmarks
    distances_array = np.linalg.norm(differences, axis=1)
    return distances_array


def calculate_non_reg_tre(moving_landmarks: np.array,
                          fixed_landmarks: np.array) -> np.array:
    differences = fixed_landmarks - moving_landmarks
    distances_array = np.linalg.norm(differences, axis=1)
    return distances_array


dh = DataHandler(val_images=12)
dh.get_synthetic_data(
    fixed_path='/home/lschilling/datam2olie/synthetic/orig/t3/Synthetic_CT/',
    moving_path='/home/lschilling/datam2olie/synthetic/orig/t1/Synthetic_CT/',
    traverse_sub_dir=False)
moving_image_paths = dh.x_val
fixed_image_paths = dh.y_val
all_tre_lists = []
for (idx, _) in enumerate(moving_image_paths):
    moving_image = sitk.ReadImage(moving_image_paths[idx])
    fixed_image = sitk.ReadImage(fixed_image_paths[idx])
    moving_landmarks, fixed_landmarks = get_landmarks(moving_image_paths[idx],
                                                      indexing='zyx')
    moved_image, displacement, time = vxmtf.register_images(
        fixed_image,
        moving_image,
        weights_path=
        '/home/lschilling/PycharmProjects/image_registration_thesis/models/test_model_30_100_8batch.index'
    )
    moved_landmarks = vxmtf.get_moved_points(fixed_landmarks, displacement)
    tre = calculate_tre(moving_landmarks, moved_landmarks)
    tre_non_reg = calculate_non_reg_tre(moving_landmarks, fixed_landmarks)

    stats.describe(tre)
    stats.describe(tre_non_reg)

    moved_image = moved_image.squeeze(0)
    moved_image = moved_image.squeeze(3)
    moved_image = sitk.GetImageFromArray(moved_image)

    with M2aiaOnlineHelper('ShowContainer') as docker:
        docker.show([('moved_image_vxm', moved_image),
                     ('fixed_image', fixed_image),
                     ('moving_image', moving_image)])

print('tre')
