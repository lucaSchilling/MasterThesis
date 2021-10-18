import os
import sys
import time

import SimpleITK as sitk
import airlab as al
import torch as th

from frameworks.ImageRegistrationInterface import ImageRegistrationInterface

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Airlab(ImageRegistrationInterface):
    @staticmethod
    def register_images(moving_image: sitk.Image,
                        fixed_image: sitk.Image,
                        weights_path: str = None,
                        loss='MSE',
                        gpu='1') -> {sitk.Image, sitk.Image, int}:

        # set the used data type
        dtype = th.float64
        device = th.device(f"cuda:{gpu}")

        fixed_image = al.Image(sitk.Cast(fixed_image, sitk.sitkFloat32), dtype,
                               device)
        moving_image = al.Image(sitk.Cast(moving_image, sitk.sitkFloat32),
                                dtype, device)

        # create pairwise registration object
        registration = al.PairwiseRegistration()

        # choose the affine transformation model
        transformation = al.transformation.pairwise.AffineTransformation(
            moving_image, opt_cm=True)
        # transformation = al.transformation.pairwise.RigidTransformation(moving_image, opt_cm=False)
        transformation.init_translation(fixed_image)
        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        if loss == 'MSE':
            image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)
        elif loss == 'MI':
            image_loss = al.loss.pairwise.MI(fixed_image,
                                             moving_image,
                                             sigma=0.01,
                                             spatial_samples=0.5,
                                             background=None)
        else:
            raise NotImplementedError(
                f'{loss} is not implemented yet please select one of the following losses [MSE, MI]'
            )
        registration.set_image_loss([image_loss])

        # choose the Adam optimizer to minimize the objective
        optimizer = th.optim.Adam(transformation.parameters(), lr=0.01)

        registration.set_optimizer(optimizer)
        # registration.set_number_of_iterations(500)
        registration.set_number_of_iterations(50)

        # start the registration
        start_time = time.time()
        registration.start(StopPatience=10)
        end_time = time.time()
        # warp the moving image with the final transformation result
        unit_displacement = transformation.get_displacement()

        # interpolating the displacement_field could be an alternative implementation but needs further research
        # orig_size = orig_moving_image.size
        # orig_displacement = nnf.interpolate(displacement_swapped, size=orig_size, mode='trilinear')
        # orig_displacement = orig_displacement.squeeze(0)
        # orig_displacement = orig_displacement.permute(1, 2, 3, 0)
        # warped_orig_image = al.transformation.utils.warp_image(orig_moving_image, orig_displacement)
        # warped_orig_image.origin = warped_orig_image.origin + cm_displacement

        moved_image = al.transformation.utils.warp_image(
            moving_image, unit_displacement)

        displacement = al.transformation.utils.unit_displacement_to_displacement(
            unit_displacement)
        displacement = sitk.GetImageFromArray(displacement.cpu(),
                                              isVector=True)
        return moved_image.itk(), displacement, end_time - start_time
