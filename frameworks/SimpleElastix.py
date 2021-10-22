import time

import SimpleITK as sitk
import numpy as np

from frameworks.ImageRegistrationInterface import ImageRegistrationInterface


class SimpleElastix(ImageRegistrationInterface):
    @staticmethod
    def register_images(moving_image: sitk.Image,
                        fixed_image: sitk.Image,
                        weights_path: str = None,
                        loss='MSE') -> {sitk.Image, np.ndarray, int}:
        moving_image_np = sitk.GetArrayFromImage(moving_image)
        default_value = np.min(moving_image_np)
        elastix_image_filter = sitk.ElastixImageFilter()
        elastix_image_filter.SetFixedImage(fixed_image)
        elastix_image_filter.SetMovingImage(moving_image)
        parameter_map_vector = sitk.VectorOfParameterMap()
        affine_parameter = sitk.GetDefaultParameterMap("affine")
        affine_parameter['ImageSampler'] = ['Random']
        affine_parameter['Transform'] = ['EulerTransform']
        if loss == 'MSE':
            affine_parameter['Metric'] = ['AdvancedMeanSquares']
        elif loss == 'MI':
            affine_parameter['Metric'] = ['AdvancedMattesMutualInformation']
        affine_parameter['Optimizer'] = ['AdaptiveStochasticGradientDescent']
        affine_parameter['AutomaticTransformInitialization'] = ['true']
        affine_parameter['AutomaticTransformInitializationMethod'] = [
            'GeometricalCenter'
        ]

        affine_parameter['NumberOfSpatialSamples'] = ['15000']

        # Optional but will reduce time by 75% without worsen the result
        affine_parameter['DefaultPixelValue'] = [f'{default_value}']
        affine_parameter['Interpolator'] = ['LinearInterpolator']
        affine_parameter['ResampleInterpolator'] = ['FinalLinearInterpolator']
        affine_parameter['ResultImagePixelType'] = ['float']

        parameter_map_vector.append(affine_parameter)
        elastix_image_filter.SetParameterMap(parameter_map_vector)

        # The actual registration process
        start_time = time.time()
        moved_image = elastix_image_filter.Execute()
        end_time = time.time()

        transformix_image_filter = sitk.TransformixImageFilter()
        transformix_image_filter.SetTransformParameterMap(
            elastix_image_filter.GetTransformParameterMap())
        transformix_image_filter.ComputeDeformationFieldOn()
        transformix_image_filter.SetMovingImage(moving_image)
        transformix_image_filter.Execute()
        displacement = transformix_image_filter.GetDeformationField()
        return moved_image, displacement, end_time - start_time
