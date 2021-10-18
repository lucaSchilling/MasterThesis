import SimpleITK as sitk
from pyM2aia import M2aiaOnlineHelper

from DataHandler import DataHandler

dh = DataHandler(batch_size=1, val_images=10, shuffle=False)
dh.get_synthetic_data(
    fixed_path='/home/lschilling/datam2olie/synthetic/orig/t3/Synthetic_CT/',
    moving_path='/home/lschilling/datam2olie/synthetic/orig/t1/Synthetic_CT/')

data_gen = dh.data_gen_voxelmorph(dh.x_train,
                                  dh.y_train,
                                  random_resampling=True)


def mywrite(dict):
    # temporary save images with given name
    for (k, v) in dict:
        print('test')


in_sample, out_sample = next(data_gen)
moving_image = in_sample[0]
fixed_image = in_sample[1]
fixed_image = sitk.GetImageFromArray(fixed_image.squeeze(0))
moving_image = sitk.GetImageFromArray(moving_image.squeeze(0))

with M2aiaOnlineHelper('ShowContainer') as docker:
    docker.write = mywrite
    docker.show(([('fixed_image', fixed_image)]))

print('test')
