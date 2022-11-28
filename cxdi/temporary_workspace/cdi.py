import numpy as np
from scipy import ndimage

from .utils import utils2d


def find_support(img,gaussian_sigma=5,fraction=0.8):
    fft = utils2d.fft2(img,ret="abs")
    # gaussian filter helps make region contigous
    fft = ndimage.gaussian_filter(fft,gaussian_sigma)
    t = np.percentile(fft,fraction*100)
    return fft>t

