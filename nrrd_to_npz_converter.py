import SimpleITK as sitk
import os
import re
import numpy as np

from DataHandler import DataHandler

output_dir = '/home/lschilling/datam2olie/synthetic/orig/npz'

dh = DataHandler(val_images=0)
dh.get_synthetic_data(
    fixed_path='/home/lschilling/datam2olie/synthetic/orig/t3/Synthetic_CT/',
    moving_path='/home/lschilling/datam2olie/synthetic/orig/t1/Synthetic_CT/',
    traverse_sub_dir=False)
all_image_paths = np.concatenate((dh.x_train, dh.y_train), axis=0)
for image_path in all_image_paths:
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    new_filename = re.sub('\.nrrd$', '', os.path.basename(image_path))
    np.savez_compressed(os.path.join(output_dir, new_filename), vol=image)
