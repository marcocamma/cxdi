import numpy as np
from scipy.interpolate import UnivariateSpline

from matplotlib import pyplot as plt


def find_fwhm(x, y, bkg="auto"):
    # create a spline of x and y-y.max()/2
    if isinstance(bkg, str) and bkg == "auto":
        bkg = (y[0] + y[-1]) / 2

    try:
        y = y - bkg
    except:
        pass
    # plt.plot(x,y)
    spline = UnivariateSpline(x, y - y.max() / 2, s=0)
    r1, r2 = spline.roots()  # find the roots
    return np.abs(r2 - r1)
