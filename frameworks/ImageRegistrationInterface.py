from abc import ABC, abstractmethod

import SimpleITK as sitk
import numpy as np


class ImageRegistrationInterface(ABC):
    @staticmethod
    @abstractmethod
    def register_images(
            moving_image: sitk.Image,
            fixed_image: sitk.Image) -> {sitk.Image, sitk.Image, int}:
        pass
