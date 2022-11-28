import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from datastorage import DataStorage as ds
import fabio

from . import utils1d

def _read(fname):
    with fabio.open(fname) as f:
        data = f.data
        header = f.header
    return ds(data=data,header=header)

def define_xy(img, scale=1, mesh=False):
    ny, nx = img.shape
    x, y = np.arange(nx) * scale, np.arange(ny) * scale
    if mesh:
        x, y = np.meshgrid(x, y)
    return x, y

def _shift_image_integers(img, dx, dy):
    img = np.roll(img, (dy, dx), axis=(0, 1))
    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0
    return img

def _shift_image_floats(img, dx, dy,**kw):
    """ not inversion of dx,dy for ndimage is along axis """
    # fastest for 0th order but still slower than
    # _shift_image_integers
    return ndimage.shift(img,(dy,dx),**kw)

def shift_image(img,dx,dy,**kw):
    """ 
    shift an image by dx(axis=1) and dy(axis=0)
    It automatically understand if a fast rolling can be used
    else an interpolation is performed using scipy.ndimage.shift
    **kw is sent to scipy.ndimage.shift
    """
    if isinstance(dx,int) and isinstance(dy,int):
        ok_int = True
    elif np.isclose(dx,round(dx)) and np.isclose(dy,round(dy)):
        ok_int = True
    else:
        ok_int = False
    if ok_int:
        return _shift_image_integers(img,dx,dy)
    else:
        return _shift_image_floats(img,dx,dy,**kw)

def shift_images(images,shifts,**kw):
    """
    shifts images[i] by shifts[i], **kw used if not integer shifts.
    See shift_image doc
    """
    for i in range(len(images)):
        images[i] = shift_image(images[i], *shifts[i],**kw)
    return images


def _pad(img, output_shape, offsets):
    """
    no check on inputs, internal function to be called by pad
    """
    # Create an array of zeros with the reference shape
    ishape = img.shape
    oshape = output_shape
    out = np.zeros(oshape)
    # Create a list of slices from offset to offset + shape in each dimension
    idx = [slice(o, o + s) for (o, s) in zip(offsets, ishape)]
    # make future numpy happy
    idx = tuple(idx)
    # Insert the array in the result at the specified offsets
    out[idx] = img
    return out


def pad(img, output_shape, offsets="auto"):
    """
    array: Array to be padded
    reference: Reference array with the desired shape,
        every dimension that contains a string can be "x2" meaning twice as big
        along that axis
    offsets: list of offsets (auto will be autodetermined to have the original image centered)
    """
    # Create an array of zeros with the reference shape
    ishape = img.shape

    # if single string, make list of string x2→ (x2,x2)
    if isinstance(output_shape, str):
        output_shape = [output_shape for _ in range(img.ndim)]

    # needed to modify it
    output_shape = list(output_shape)

    for i in range(img.ndim):
        if isinstance(output_shape[i], str):
            factor = float(output_shape[i][1:])  # assumses x3.4
            output_shape[i] = round(ishape[i] * factor)

    if isinstance(offsets, str) and offsets == "auto":
        offsets = [round((o - i) / 2) for (i, o) in zip(ishape, output_shape)]
    return _pad(img, output_shape, offsets)


def fft2(img, shift=True, norm=False,resize=None, ret="complex"):
    """
    Parameters
    ----------
    resize : None or shape
        if not None, the image is resize (using pad function)
        before doing fft
    norm : bool
        if True, divide by 1/sqrt(shape[0]*shape[1]) 
        note that ifft2 is designed to work with norm=False
    ret : {"complex","phase","angle","mag","abs","abs2"}
        what to return (abs2 = abs**2)
    """
    if resize is not None:
        img = pad(img, resize)
    fft = np.fft.fft2(img)
    if norm: fft /= np.sqrt((img.shape[0]*img.shape[1]))
    if shift:
        fft = np.fft.fftshift(fft)
    if ret in ("mag", "abs"):
        fft = np.abs(fft)
    elif ret in ("mag2", "abs2"):
        fft = np.abs(fft)**2
    elif ret in ("phase", "angle"):
        fft = np.angle(fft)
    else:
        pass
    return fft

def ifft2(img, shift=False, ret="complex"):
    """
    Parameters
    ----------
    img : ndarray resulting from fft(img,shift=False)
    ret : {"complex","phase","angle","mag","abs","abs2"}
        what to return (abs2 = abs**2)
    """
    fft = np.fft.ifft2(img)
    if ret in ("mag", "abs"):
        fft = np.abs(fft)
    elif ret in ("mag2", "abs2"):
        fft = np.abs(fft)**2
    elif ret in ("phase", "angle"):
        fft = np.angle(fft)
    else:
        pass
    return fft




def _autocorrelate(image):
    ny, nx = image.shape
    hy, hx = round(ny / 2), round(nx / 2)
    cx = np.empty(nx)
    cy = np.empty(ny)

    for i in range(nx):
        i2 = shift_image(image, i, 0)
        cx[i] = (image * i2).sum()
    for i in range(ny):
        i2 = shift_image(image, 0, i)
        cy[i] = (image * i2).sum()
    # use fftshift to move max to center
    cx = np.fft.fftshift(cx)
    cy = np.fft.fftshift(cy)
    return cx, cy


def _speckle_size(image):
    """ internal function, it works on one image, no check on inputs """
    ix, iy = _autocorrelate(image)
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])

    sx = utils1d.find_fwhm(x, ix)
    sy = utils1d.find_fwhm(y, iy)
    return sx, sy


def speckle_size(images, region=(slice(200), slice(200))):
    """
    Parameters
    region : None, "auto", list of slices or list of integers
        if None : use full image
        if auto : take 1/4,1/4 size
        if list of integers (n1,n2) it is converted to (slice(n1),slice(n2))
    """
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[np.newaxis, :, :]
    nimages = len(images)
    speckle_sizes = np.zeros((nimages, 2))

    if isinstance(region, str) and region == "auto":
        region = slice(round(images[0].shape[0] / 4), round(images[0].shape[1] / 4))
    if region is not None:
        # convert int to slice(int) if needed
        region = [slice(n) if isinstance(n, int) else n for n in region]

        # make future numpy happy
        region = tuple(region)

    for i, img in enumerate(images):
        if region is not None:
            sx, sy = _speckle_size(img[region])
        else:
            sx, sy = _speckle_size(img)
        speckle_sizes[i][0] = sx
        speckle_sizes[i][1] = sy
    if nimages == 1:
        speckle_sizes = speckle_sizes[0]
    return speckle_sizes

def rotate_by_pi(img,center):
    import cv2
    center = tuple(center)
    rot_mat = cv2.getRotationMatrix2D(center, 180, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
    newimage = np.empty_like(img)
    cy,cx = center
    # number of pixels on the right of the center
    ry,rx = img.shape[0]-cy,img.shape[1]
    #newimage[cy:,cx:] = img[cy:2*cy
    #img = shift_image(img,-center[0],-center[1])
    img = img[::-1,::-1]
    sx = -2*center[0]
    sy = -2*center[1]
    img = shift_image(img,sx,sy)
    return img

