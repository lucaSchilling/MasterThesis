import os

import SimpleITK as sitk
import numpy as np
from DataHandler import DataHandler


def calculate_origin(image: sitk.Image,
                     new_size=None,
                     new_spacing=None) -> np.array:
    if new_size is not None and new_spacing is not None:
        return -new_size * new_spacing / 2.0
    else:
        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        return -size * spacing / 2.0


def calculate_shift(image: sitk.Image) -> np.array:
    center_mm = (np.array(image.GetSize()) * np.array(image.GetSpacing())) / 2
    shift_vector_mm = center_mm * (-1)
    return shift_vector_mm


def get_idx_resampled(image_native: sitk.Image, image_resampled: sitk.Image,
                      shift_vector_mm: np.array,
                      point: tuple) -> (tuple, tuple):
    point_native_mm = image_native.TransformContinuousIndexToPhysicalPoint(
        point)
    point_resampled_mm = point_native_mm + shift_vector_mm
    point_resampled_idx = image_resampled.TransformPhysicalPointToIndex(
        point_resampled_mm)
    return point_resampled_idx, point_resampled_mm


dh = DataHandler(val_images=12)
dh.get_synthetic_data(
    fixed_path='/home/lschilling/datam2olie/synthetic/orig/t3/Synthetic_CT/',
    moving_path='/home/lschilling/datam2olie/synthetic/orig/t1/Synthetic_CT/',
    traverse_sub_dir=False)
t1s_resampled = dh.x_val
t3s_resampled = dh.y_val
dh.get_synthetic_data(
    fixed_path='/home/lschilling/datam2olie/synthetic/native/t3/Synthetic_CT/',
    moving_path='/home/lschilling/datam2olie/synthetic/native/t1/Synthetic_CT/',
    traverse_sub_dir=False)
t1s_native = dh.x_val
t3s_native = dh.y_val

output_dir = '/home/lschilling/datam2olie/synthetic/orig/CT_points_t1_t3/'
vector_fields_dir = '/home/lschilling/datam2olie/synthetic/native/CT_vector_fields/'

for (image_idx, _) in enumerate(t3s_resampled):
    t1_native = sitk.ReadImage(t1s_native[image_idx])
    t3_native = sitk.ReadImage(t3s_native[image_idx])
    t1_resampled = sitk.ReadImage(t1s_resampled[image_idx])
    t3_resampled = sitk.ReadImage(t3s_resampled[image_idx])
    size_resampled = t1_resampled.GetSize()
    model_name = os.path.basename(t3s_native[image_idx]).replace(
        '_atn_3.nrrd', '')
    vector_field_path = f'{vector_fields_dir}{model_name}_vec_frame1_to_frame2.txt'

    shift_vector_t1_mm = calculate_shift(t1_native)
    shift_vector_t3_mm = calculate_shift(t3_native)

    vector_field = np.genfromtxt(vector_field_path,
                                 usecols=(2, 3, 4, 6, 7, 8),
                                 names='1X, 1Y, 1Z, 2X, 2Y, 2Z',
                                 dtype=None,
                                 skip_header=2)

    points_t1 = [
        (float(vector_field[idx]['1X']), float(vector_field[idx]['1Y']),
         float(vector_field[idx]['1Z']))
        for (idx, _) in enumerate(vector_field)
    ]
    points_t3 = [
        (float(vector_field[idx]['2X']), float(vector_field[idx]['2Y']),
         float(vector_field[idx]['2Z']))
        for (idx, _) in enumerate(vector_field)
    ]

    points_t1_resampled_idx = []
    points_t3_resampled_idx = []
    points_t1_resampled_mm = []
    points_t3_resampled_mm = []

    for (point_idx, _) in enumerate(points_t1):
        point_t1_resampled_idx, point_t1_resampled_mm = get_idx_resampled(
            t1_native, t1_resampled, shift_vector_t1_mm, points_t1[point_idx])
        point_t3_resampled_idx, point_t3_resampled_mm = get_idx_resampled(
            t3_native, t3_resampled, shift_vector_t3_mm, points_t3[point_idx])

        if all(((y >= t1 >= 0) and (y >= t3 >= 0)) for t1, t3, y in zip(
                point_t1_resampled_idx, point_t3_resampled_idx, size_resampled)
               ) and t1_resampled.GetPixel(point_t1_resampled_idx) != 0.0:
            points_t1_resampled_idx.append(point_t1_resampled_idx)
            points_t3_resampled_idx.append(point_t3_resampled_idx)
            points_t1_resampled_mm.append(point_t1_resampled_mm)

    np.savez_compressed(f'{output_dir}{model_name}_idx',
                        t1=np.array(points_t1_resampled_idx),
                        t3=np.array(points_t3_resampled_idx))
    np.savez_compressed(f'{output_dir}{model_name}_mm',
                        t1=np.array(points_t1_resampled_mm),
                        t3=np.array(points_t3_resampled_mm))
