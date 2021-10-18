import numpy as np

from frameworks.VoxelmorphTorch import VoxelmorphTorch
from frameworks.SimpleElastix import SimpleElastix
from frameworks.VoxelmorphTF import VoxelmorphTF
import SimpleITK as sitk
import neurite as ne
import os
from DataHandler import DataHandler
from frameworks.Airlab import Airlab


def get_jacobian_np(displacement: sitk.Image) -> np.array:
    jacobian_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jacobian_np = sitk.GetArrayFromImage(jacobian_filter.Execute(displacement))
    return jacobian_np


dh = DataHandler()
dh.get_mnist_data(select_number=5)
val_generator = dh.data_gen_val(data_x=dh.x_val, data_y=dh.y_val)
moving_image, fixed_image = next(val_generator)

while True:
    moving_image_np = sitk.GetArrayFromImage(moving_image)
    fixed_image_np = sitk.GetArrayFromImage(fixed_image)

    # # SimpleElastix
    moved_image_elastix, displacement_elastix, time_elastix = SimpleElastix.register_images(
        moving_image=moving_image, fixed_image=fixed_image, loss='MSE')
    jacobian_elastix = get_jacobian_np(displacement_elastix)

    # # Airlab
    moved_image_airlab, displacement_airlab, time_airlab, loss_airlab = Airlab.register_images(
        moving_image=moving_image, fixed_image=fixed_image, loss='MSE')
    jacobian_airlab = get_jacobian_np(displacement_airlab)

    # Voxelmorph Tensorflow
    # moved_image_vxmtf, displacement_vxmtf, time_vxmtf, loss_vxmtf = VoxelmorphTF.register_images(
    #     moving_image=moving_image,
    #     fixed_image=fixed_image,
    #     weights_path=
    #     '/home/lschilling/PycharmProjects/MasterThesis/models/mnist/vxmtf_ep300_st100_lr0.0001_bat8_final_loss0_0085.h5',
    #     loss='MSE',
    # )
    # jacobian_vxmtf = get_jacobian_np(displacement_vxmtf)

    # # Voxelmorph PyTorch
    moved_image_vxmth, displacement_vxmth, time_vxmth = VoxelmorphTorch.register_images(
        moving_image=moving_image,
        fixed_image=fixed_image,
        weights_path=
        '/home/lschilling/PycharmProjects/MasterThesis/models/mnist/ep300_st100_lr0.0001_bat80300.pt',
        loss='MSE',
    )
    jacobian_vxmth = get_jacobian_np(displacement_vxmth)

    images = [
        moving_image_np,
        fixed_image_np,
    #sitk.GetArrayFromImage(moved_image_vxmtf), jacobian_vxmtf,
        sitk.GetArrayFromImage(moved_image_vxmth),
        jacobian_vxmth,
        sitk.GetArrayFromImage(moved_image_airlab),
        jacobian_airlab,
        sitk.GetArrayFromImage(moved_image_elastix),
        jacobian_elastix
    ]

    titles = [
        'moving',
        'fixed',
    #f'voxelmorph: {round(time_vxmtf, 2)}sec mse loss: {round(loss_vxmtf, 3)}',
    #f'jacobian_vxm < 0: {len(jacobian_vxmtf[jacobian_vxmtf < 0])}',
        f'voxelmorph: {round(time_vxmth, 2)}sec',
        f'jacobian_vxm < 0: {len(jacobian_vxmth[jacobian_vxmth < 0])}',
        f'airlab: {round(time_airlab, 2)}sec mse loss: {round(loss_airlab, 3)}',
        f'jacobian_airlab < 0: {len(jacobian_airlab[jacobian_airlab < 0])}',
        f'simpleElastix: {round(time_elastix, 2)}sec',
        f'jacobian_elastix < 0: {len(jacobian_elastix[jacobian_elastix < 0])}'
    ]
    ne.plot.slices(images,
                   titles=titles,
                   cmaps=['gray'],
                   do_colorbars=True,
                   grid=True)

    titles = [
    #'displacement_vxmtf',
        'displacement_vxmth',
        'displacement_airlab',
        'displacement_elastix'
    ]
    ne.plot.flow(
        [
    #sitk.GetArrayFromImage(displacement_vxmtf).squeeze(),
            sitk.GetArrayFromImage(displacement_vxmth).squeeze(),
            sitk.GetArrayFromImage(displacement_airlab).squeeze(),
            sitk.GetArrayFromImage(displacement_elastix).squeeze()
        ],
        titles=titles,
        width=5)
    moving_image, fixed_image = next(val_generator)
