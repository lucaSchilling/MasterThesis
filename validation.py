import json
import os

from DataHandler import DataHandler
from frameworks.Airlab import Airlab
from frameworks.SimpleElastix import SimpleElastix
import SimpleITK as sitk
import numpy as np
from scipy import stats


def get_landmarks(fixed_image_path: str,
                  indexing: str = 'zyx') -> (np.array, np.array):
    model_name = os.path.basename(fixed_image_path).replace('_atn_3.nrrd', '')
    loaded_points = np.load(
        f'/home/lschilling/datam2olie/synthetic/orig/CT_points_t1_t3/{model_name}_idx.npz'
    )
    moving_landmarks = loaded_points['t1']
    fixed_landmarks = loaded_points['t3']
    if indexing == 'zyx':
        # swap columns because numpy and vxm use zyx indexing and the data uses xyz indexing
        moving_landmarks[:, [0, 2]] = moving_landmarks[:, [2, 0]]
        fixed_landmarks[:, [0, 2]] = fixed_landmarks[:, [2, 0]]
    else:
        assert indexing == 'xyz', f'indexing can only be xyz or zyx. Got: {indexing}'
    return moving_landmarks.astype(np.float64), fixed_landmarks.astype(
        np.float64)


def get_tre(moving_landmarks: np.array, moved_landmarks: np.array) -> np.array:
    differences = moved_landmarks - moving_landmarks
    distances_array = np.linalg.norm(differences, axis=1)
    return distances_array


def get_tre_non_reg(moving_landmarks: np.array,
                    fixed_landmarks: np.array) -> np.array:
    differences = fixed_landmarks - moving_landmarks
    distances_array = np.linalg.norm(differences, axis=1)
    return distances_array


def get_mse(moved_image: sitk.Image, fixed_image: sitk.Image) -> float:
    moved_image_np = sitk.GetArrayFromImage(moved_image).astype(np.float64)
    fixed_image_np = sitk.GetArrayFromImage(fixed_image).astype(np.float64)
    difference = np.subtract(moved_image_np, fixed_image_np)
    squared_difference = np.square(difference)
    mse = squared_difference.mean()
    return mse


def get_jacobian_np(displacement: sitk.Image) -> np.array:
    jacobian_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jacobian_np = sitk.GetArrayFromImage(jacobian_filter.Execute(displacement))
    return jacobian_np


def get_moved_points(points: np.array, displacement: sitk.Image) -> np.array:
    displacement_copy = displacement.__copy__()
    displacement_transform = sitk.DisplacementFieldTransform(displacement_copy)
    moved_points = [
        displacement_transform.TransformPoint(point) for point in points
    ]
    return moved_points


framework_name = 'vxmtf'
dataset = 'synthetic'
model_path = '/home/lschilling/PycharmProjects/image_registration_thesis/models/test_model_30_100_8batch'
model_name = os.path.basename(model_path)
if framework_name == 'vxmtf':
    from frameworks.VoxelmorphTF import VoxelmorphTF
    from frameworks.VoxelmorphTorch import VoxelmorphTorch
else:
    from frameworks.VoxelmorphTorch import VoxelmorphTorch
    from frameworks.VoxelmorphTF import VoxelmorphTF

frameworks = {
    'airlab': Airlab(),
    'simpleelastix': SimpleElastix(),
    'vxmtf': VoxelmorphTF(),
    'vxmth': VoxelmorphTorch()
}
framework = frameworks[framework_name]

dh = DataHandler(val_images=12)
dh.get_synthetic_data(
    fixed_path='/home/lschilling/datam2olie/synthetic/orig/t3/Synthetic_CT/',
    moving_path='/home/lschilling/datam2olie/synthetic/orig/t1/Synthetic_CT/',
    traverse_sub_dir=False)
moving_image_paths = dh.x_val
fixed_image_paths = dh.y_val
if dataset == 'synthetic':
    tre_list = []
    tre_non_reg_list = []
time_list = []
mse_list = []
jacobian_reflections_list = []
for (idx, _) in enumerate(moving_image_paths):
    moving_image = sitk.ReadImage(moving_image_paths[idx])
    fixed_image = sitk.ReadImage(fixed_image_paths[idx])
    moved_image, displacement, time = framework.register_images(
        fixed_image, moving_image, weights_path=model_path)
    jacobian = get_jacobian_np(displacement)
    jacobian_reflections_list.append(len(jacobian[jacobian < 0]))
    time_list.append(time)
    mse_list.append(get_mse(moving_image, fixed_image))
    if dataset == 'synthetic':
        moving_landmarks, fixed_landmarks = get_landmarks(
            fixed_image_paths[idx], indexing='zyx')
        moving_landmarks_xyz, fixed_landmarks_xyz = get_landmarks(
            fixed_image_paths[idx], indexing='xyz')
        moved_landmarks = framework.get_moved_points(fixed_landmarks,
                                                     displacement)
        moved_landmarks_sitk = get_moved_points(fixed_landmarks_xyz,
                                                displacement)
        tre_test = get_tre(moving_landmarks_xyz, moved_landmarks_sitk)
        print(stats.describe(tre_test))
        tre_non_reg_test = get_tre_non_reg(moving_landmarks_xyz,
                                           fixed_landmarks_xyz)
        print(stats.describe(tre_non_reg_test))

        tre = get_tre(moving_landmarks, moved_landmarks)
        tre_non_reg = get_tre_non_reg(moving_landmarks, fixed_landmarks)
        tre_list.append(tre.mean())
        tre_non_reg_list.append(tre_non_reg.mean())
        print(stats.describe(tre))
        print(stats.describe(tre_non_reg))
        pass

output_path = os.path.join(
    '/home/lschilling/PycharmProjects/MasterThesis/models', dataset,
    (model_name if model_name else framework_name))
if not os.path.exists(output_path):
    os.mkdir(output_path)

results_dict = {
    'time_list': time_list,
    'mse_list': mse_list,
    'jacobian_reflections_list': jacobian_reflections_list,
    'time_mean': np.array(time_list).mean(),
    'mse_mean': np.array(mse_list).mean(dtype=float),
    'jacobian_reflections_mean': np.array(jacobian_reflections_list).mean()
}

if dataset == 'synthetic':
    results_dict['tre_list'] = tre_list
    results_dict['tre_non_reg_list'] = tre_non_reg_list
    results_dict['tre_mean'] = np.array(tre_list).mean()
    results_dict['tre_non_reg_mean'] = np.array(tre_non_reg_list).mean()

json.dump(results_dict,
          open(os.path.join(output_path, 'results.json'), 'w'),
          indent=4)
