from DataHandler import DataHandler
import SimpleITK as sitk
import numpy as np
from pyM2aia import M2aiaOnlineHelper
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pathlib


def resample_image(image, size, spacing, origin, interpolator=2) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size.tolist())
    resampler.SetOutputSpacing(spacing.tolist())
    resampler.SetOutputOrigin(origin.tolist())
    image_np = sitk.GetArrayFromImage(image)
    resampler.SetDefaultPixelValue(int(image_np.min()))
    resampler.SetInterpolator(interpolator)

    image_resampled = resampler.Execute(image)
    normalized_image = normalize_image(image_resampled)
    return normalized_image


def normalize_image(image: sitk.Image) -> sitk.Image:
    image_np = sitk.GetArrayFromImage(image)
    min_value = image_np.min()
    max_value = image_np.max()
    image_np = (image_np - min_value) / (max_value - min_value)
    image_result = sitk.GetImageFromArray(image_np)
    image_result.SetSpacing(image.GetSpacing())
    image_result.SetOrigin(image.GetOrigin())
    return image_result


def resample_image_0_0_0_centered(image_path: str, size: np.array,
                                  spacing: np.array) -> sitk.Image:
    image = sitk.ReadImage(image_path)
    new_origin = calculate_origin(image)
    image.SetOrigin(new_origin.tolist())
    new_origin = calculate_origin(image, size, spacing)
    image_resampled = resample_image(image, size, spacing, new_origin)
    return image_resampled


def calculate_origin(image: sitk.Image, size=None, spacing=None) -> np.array:
    if size is None or spacing is None:
        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        return -size * spacing / 2.0
    else:
        return -size * spacing / 2.0


def get_all_new_origins(image_array: np.array) -> list:
    origin_list = []
    for image in image_array:
        image_sitk = sitk.ReadImage(image)
        new_origin = calculate_origin(image_sitk)
        origin_list.append(new_origin)
    return origin_list


def get_physical_sizes(image_array: np.array) -> list:
    list_of_dicts = []
    for image in image_array:
        image_sitk = sitk.ReadImage(image)
        size = image_sitk.GetSize()
        spacing = image_sitk.GetSpacing()
        physical_volume = np.dot(size, spacing)
        physical_size = (size[0] * spacing[0], size[1] * spacing[1],
                         size[2] * spacing[2])
        sizes_dict = {
            'physical_volume': physical_volume,
            'physical_size': physical_size,
            'spacing': spacing,
            'path': image
        }
        list_of_dicts.append(sizes_dict)
    return list_of_dicts


def get_all_sizes_spacings(image_array: np.array) -> (list, list):
    sizes = []
    spacings = []
    for image in image_array:
        image_sitk = sitk.ReadImage(image)
        size = image_sitk.GetSize()
        spacing = image_sitk.GetSpacing()
        sizes.append(size)
        spacings.append(spacing)
    return sizes, spacings


def plot_3d_list(point_list: list) -> None:
    sns.set(style="darkgrid")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [point_tuple[0] for point_tuple in point_list]
    y = [point_tuple[1] for point_tuple in point_list]
    z = [point_tuple[2] for point_tuple in point_list]

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.scatter(x, y, z)

    plt.show()


dh = DataHandler(val_images=0)
dh.get_synthetic_data(
    fixed_path='/home/lschilling/datam2olie/synthetic/native/t3/',
    moving_path='/home/lschilling/datam2olie/synthetic/native/t1/',
    traverse_sub_dir=True)
all_image_paths = np.concatenate((dh.x_train, dh.y_train), axis=0)
origin_list = get_all_new_origins(all_image_paths)
plot_3d_list(origin_list)
all_sizes, all_spacings = get_all_sizes_spacings(all_image_paths)
print(stats.describe(all_sizes))
print(stats.describe(all_spacings))
plot_3d_list(all_sizes)
plot_3d_list(all_spacings)

physical_sizes = get_physical_sizes(all_image_paths)
x_sizes = [sizes_dict['physical_size'][0] for sizes_dict in physical_sizes]
x_size_percentile = np.percentile(x_sizes, 75)
y_sizes = [sizes_dict['physical_size'][1] for sizes_dict in physical_sizes]
y_size_percentile = np.percentile(y_sizes, 75)
z_sizes = [sizes_dict['physical_size'][2] for sizes_dict in physical_sizes]
z_size_percentile = np.percentile(z_sizes, 75)

x_spacings = [sizes_dict['spacing'][0] for sizes_dict in physical_sizes]
x_spacing_percentile = np.percentile(x_spacings, 75)
y_spacings = [sizes_dict['spacing'][1] for sizes_dict in physical_sizes]
y_spacing_percentile = np.percentile(y_spacings, 75)
z_spacings = [sizes_dict['spacing'][2] for sizes_dict in physical_sizes]
z_spacing_percentile = np.percentile(z_spacings, 75)

# This would be the closest isotropic spacing with a size that can be downsampled
#new_size, new_spacing = np.array((256, 256, 105)), np.array((2, 2, 2))
new_size, new_spacing = np.array((256, 256, 128)), np.array((1.8, 1.8, 1.8))

for path in dh.x_train:
    image_resampled = resample_image_0_0_0_centered(path, new_size,
                                                    new_spacing)
    path = pathlib.Path(path)
    parts = list(path.parts)
    index = path.parts.index('native')
    parts[index] = 'orig'
    new_path = pathlib.Path(*parts)
    if not new_path.parent.exists():
        new_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image_resampled, str(new_path))
