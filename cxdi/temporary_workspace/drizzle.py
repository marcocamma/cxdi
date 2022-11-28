""" Functions to upsample the low resolution data collected by
    translating the detector.
    It is VERY easy to get the sign of the translation wrong, so collect
    simple data (direct beam for example) and/or try with repetitions=0
"""
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from .utils.utils2d import shift_image, shift_images


def shift_images_diagonal(images):
    shifts = []
    nimages = len(images)
    nhalf = int((nimages - 1) / 2)
    for i in range(1, nimages + 1):
        shifts.append((-nhalf + i - 1, -nhalf + i - 1))
    return shift_images(images, shifts)


def upsample(images, factor, same_integrated_intensity=True):
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[np.newaxis, :]
    if isinstance(factor, int):
        factor = factor, factor
    nx, ny = factor
    images = images.repeat(ny, axis=1).repeat(nx, axis=2)
    if images.shape[0] == 1:
        images = images[0]
    if same_integrated_intensity:
        try:
            images /= nx * ny
        except TypeError:  # can be problem for integers
            images = images / (nx * ny)
    return images
    # ndimage.zoom(image,factor,order=order)


def downsample(images, factor, same_integrated_intensity=True):
    """ down sample the image(s) by a certain factor
    Parameters
    ----------
    a : array_like
        Input array; can be 2D or 3D; if 3D first index is for different images
    shift : int or tuple of ints
        if int the same factor is used for each axis
    """
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[np.newaxis, :]
    if isinstance(factor, int):
        factor = factor, factor
    nx, ny = factor
    n1 = images[0].shape[0] // ny
    n2 = images[0].shape[1] // nx
    images = images[:, : n1 * ny, : n2 * nx]
    images = images.reshape((-1, n1, ny, n2, nx))
    images = images.sum(axis=(2, 4))
    if images.shape[0] == 1:
        images = images[0]
    if not same_integrated_intensity:
        images *= ny * nx
    return images


def rebin(images, factor, same_integrated_intensity=True):
    images = np.asarray(images)
    d = downsample(images, factor, same_integrated_intensity=same_integrated_intensity)
    return upsample(d, factor, same_integrated_intensity=same_integrated_intensity)


def drizzle(images, shifts, repetitions=50, auto_stop=True):
    """ shifts is a list of list ( (x1,y1), (x2,y2), ... ) 
        shifts are in fractional pixels of the real detector image
        auto_stop True means it might stop before the requested
        number of repetitions if corrections are getting bigger
    """
    N = len(images)
    images = np.asarray(images)

    # find upsampling factors for x and y based on shifts
    shiftsx = [s[0] for s in shifts]
    shiftsy = [s[1] for s in shifts]
    factorx = round(1 / np.max(np.diff(np.unique(shiftsx))))
    factory = round(1 / np.max(np.diff(np.unique(shiftsy))))
    factor = (factorx, factory)
    # upsample images
    nx = factorx * images[0].shape[1]
    ny = factory * images[0].shape[0]
    images_upsampled = np.empty((N, ny, nx))
    for i, img in enumerate(images):
        images_upsampled[i] = upsample(img, factor)

    images_upsampled_shifted = np.empty_like(images_upsampled)
    # find shifts in units of the upsampled images
    shifts_indeces_upsampled = [
        (int(s[0] * factorx), int(s[1] * factory)) for s in shifts
    ]
    # shift images
    for i in range(len(images)):
        img = images_upsampled[i]
        ix, iy = shifts_indeces_upsampled[i]
        images_upsampled_shifted[i] = shift_image(img, ix, iy)

    # first guess
    I = images_upsampled_shifted.mean(0)

    fac = (factorx, factory)
    di_rep_last = np.inf
    for _ in range(repetitions):
        di_rep = 0
        for i in range(N):
            ix, iy = shifts_indeces_upsampled[i]
            I_to_compare = shift_image(I, -ix, -iy)
            I_to_compare = rebin(I_to_compare, fac)
            di = (I_to_compare - images_upsampled[i]) / N
            di_rep += di
        if auto_stop:
            di_rep = np.abs(di_rep).sum()
            if di_rep > di_rep_last:
                # print("auto_stop at",_)
                break
            else:
                di_rep_last = di_rep
        I = I - di

    return I


def diagonal_drizzle(images, repetitions=5, direction="+", auto_stop=True):
    """ Simple drizzle function. It assumes that the data have beem measured simmetrically with scans like:
    d2scan(ydet,-0.025,0.025,zccd,-0.025,0.025,4,100,save_images=True)
    or
    d2scan(ydet,-0.03,0.03,zccd,-0.03,0.03,4,100,save_images=True)
    """
    N = len(images)
    nmiddle = int((N - 1) / 2)
    shifts = []
    for i in range(N):
        dx = (-nmiddle + i) / N
        if direction == "+":
            shifts.append((-dx, -dx))
        else:
            shifts.append((dx, dx))
    return drizzle(images, shifts, repetitions=repetitions, auto_stop=auto_stop)
