import torch
import math
import numbers
import numpy as np
import pdb

from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms.functional as TF
import random


class RandomRotate(object):
    """
    Flip the image and mask from top to bottom or from left to right
    Args:
        sample: include image and label
    Returns:
        image and label (dict)
    """

    def __init__(self, angels):
        self.angels = angels

    def __call__(self, image):

        angle = random.choice(self.angels)
        image = TF.rotate(image, angle)
        return image