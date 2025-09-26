import os
import sys
import json
import numpy as np
import rvt.vis
import rvt.default
from osgeo import gdal
from osgeo import osr
from scipy.ndimage import gaussian_filter

from pathlib import Path

# for colored text on console
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

def logging(msg, tag=None):
    if tag == bcolors.HEADER:
        print(f'{bcolors.HEADER}{msg}{bcolors.ENDC}')
    elif tag == bcolors.OKBLUE:
        print(f'{bcolors.OKBLUE}{msg}{bcolors.ENDC}')
    elif tag == bcolors.OKCYAN:
        print(f'{bcolors.OKCYAN}{msg}{bcolors.ENDC}')
    elif tag == bcolors.OKGREEN:
        print(f'{bcolors.OKGREEN}{msg}{bcolors.ENDC}')
    elif tag == bcolors.WARNING:
        print(f'{bcolors.WARNING}[W] {msg}{bcolors.ENDC}')
    elif tag == bcolors.FAIL:
        print(f'{bcolors.FAIL}[F] {msg}{bcolors.ENDC}')
    elif tag == bcolors.BOLD:
        print(f'{bcolors.BOLD}{msg}{bcolors.ENDC}')
    elif tag == bcolors.UNDERLINE:
        print(f'{bcolors.UNDERLINE}{msg}{bcolors.ENDC}')
    else:
        print(msg)
def lowPass(z, l0):
    # create a low-pass filter that smooths topography using a Gaussian kernel
    lY, lX = np.shape(z)
    x, y = np.arange(-lX/2, lX/2), np.arange(-lY/2, lY/2)
    X, Y = np.meshgrid(x, y)
    filt = 1/(2*np.pi*l0**2)*np.exp(-(X**2 + Y**2)/(2*l0**2))
    ftFilt = np.fft.fft2(filt)
    ftZ = np.fft.fft2(z)
    ftZNew = ftZ*ftFilt
    zNew = np.fft.ifft2(ftZNew).real
    zNew = np.fft.fftshift(zNew)
    return zNew

def read_dem(im_path):
    ds = gdal.Open(im_path)
    dem_arr = ds.GetRasterBand(1).ReadAsArray()
    geotransform = ds.GetGeoTransform()
    dem_res_x = geotransform[1]
    dem_res_y = -geotransform[5]
    dem_no_data = ds.GetRasterBand(1).GetNoDataValue()
    return dem_arr, dem_res_x, dem_res_y, dem_no_data

#def get_slope(dem_arr, dem_res_x, dem_res_y, dem_no_data):
#    x, y = np.gradient(dem_arr)
#    slope = np.arctan(np.sqrt(x*x + y*y))
#    slope = np.degrees(slope)
#    limit = 50
#    slope_arr = np.clip(slope, 0, limit)
#    return slope_arr

def get_slope(dem_arr, dem_res_x, dem_res_y, dem_no_data):
    dict_slope_aspect = rvt.vis.slope_aspect(
        dem=dem_arr,
        resolution_x=dem_res_x,
        resolution_y=dem_res_y,
        output_units='degree',
        ve_factor=1,
        no_data=dem_no_data
    )
    limit = 50
    slope_arr = dict_slope_aspect['slope']
    slope_arr = np.clip(slope_arr, 0, limit)                                    # enhances contrast
    return slope_arr


def integral_image(dem, data_type=np.float64):
    dem = dem.astype(data_type)
    return dem.cumsum(axis=0).cumsum(axis=1)

def mean_filter(dem, kernel_radius):
    """Applies mean filter (low pass filter) on DEM. Kernel radius is in pixels. Kernel size is 2 * kernel_radius + 1.
    It uses matrix shifting (roll) instead of convolutional approach (works faster).
    It returns mean filtered dem as numpy.ndarray (2D numpy array)."""
    radius_cell = int(kernel_radius)

    if kernel_radius == 0:
        return dem

    # store nans
    idx_nan_dem = np.isnan(dem)

    # mean filter
    dem_pad = np.pad(dem, (radius_cell + 1, radius_cell), mode="edge")
    # store nans
    idx_nan_dem_pad = np.isnan(dem_pad)
    # change nan to 0
    dem_pad[idx_nan_dem_pad] = 0

    # kernel nr pixel integral image
    dem_i_nr_pixels = np.ones(dem_pad.shape)
    dem_i_nr_pixels[idx_nan_dem_pad] = 0
    dem_i_nr_pixels = integral_image(dem_i_nr_pixels, np.int64)

    dem_i1 = integral_image(dem_pad)

    kernel_nr_pix_arr = (np.roll(dem_i_nr_pixels, (radius_cell, radius_cell), axis=(0, 1)) +
                         np.roll(dem_i_nr_pixels, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
                         np.roll(dem_i_nr_pixels, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
                         np.roll(dem_i_nr_pixels, (radius_cell, -radius_cell - 1), axis=(0, 1)))
    mean_out = (np.roll(dem_i1, (radius_cell, radius_cell), axis=(0, 1)) +
                np.roll(dem_i1, (-radius_cell - 1, -radius_cell - 1), axis=(0, 1)) -
                np.roll(dem_i1, (-radius_cell - 1, radius_cell), axis=(0, 1)) -
                np.roll(dem_i1, (radius_cell, -radius_cell - 1), axis=(0, 1)))
    mean_out = mean_out / kernel_nr_pix_arr
    mean_out = mean_out.astype(np.float32)
    mean_out = mean_out[radius_cell:-(radius_cell + 1), radius_cell:-(radius_cell + 1)]  # remove padding
    # nan back to nan
    mean_out[idx_nan_dem] = np.nan

    return mean_out

def msrm(dem,
         resolution,
         feature_min,
         feature_max,
         scaling_factor,
         ve_factor=1,
         no_data=None
         ):
    """
    Taken directly from RVTPY
    Compute Multi-scale relief model (MSRM).

    Parameters
    ----------
    dem : numpy.ndarray
        Input digital elevation model as 2D numpy array.
    resolution : float
        DEM pixel size.
    feature_min: float
        Minimum size of the feature you want to detect in meters.
    feature_max: float
        Maximum size of the feature you want to detect in meters.
    scaling_factor: int
        Scaling factor, if larger than 1 it provides larger range of MSRM values (increase contrast and visibility),
        but could result in a loss of sensitivity for intermediate sized features.
    ve_factor : int or float
        Vertical exaggeration factor.
    no_data : int or float
        Value that represents no_data, all pixels with this value are changed to np.nan .

    Returns
    -------
    msrm_out : numpy.ndarray
        2D numpy result array of Multi-scale relief model.
    """
    if not (10000 >= ve_factor >= -10000):
        raise Exception("rvt.visualization.msrm: ve_factor must be between -10000 and 10000!")
    if resolution < 0:
        raise Exception("rvt.visualization.msrm: resolution must be a positive number!")

    # change no_data to np.nan
    if no_data is not None:
        dem[dem == no_data] = np.nan

    dem = dem.astype(np.float32)
    dem = dem * ve_factor

    if feature_min < resolution:  # feature_min can't be smaller than resolution
        feature_min = resolution

    scaling_factor = int(scaling_factor)  # has to be integer

    # calculation of i and n (from article)
    i = int(np.floor(((feature_min - resolution) / (2 * resolution)) ** (1 / scaling_factor)))
    n = int(np.ceil(((feature_max - resolution) / (2 * resolution)) ** (1 / scaling_factor)))

    # lpf = low pass filter
    relief_models_sum = np.zeros(dem.shape)  # sum of all substitution of 2 consecutive
    nr_relief_models = 0  # number of additions (substitutions of 2 consecutive surfaces)
    last_lpf_surface = 0

    # generation of filtered surfaces (lpf_surface)
    for ndx in range(i, n + 1, 1):
        kernel_radius = ndx ** scaling_factor
        # calculate mean filtered surface
        lpf_surface = mean_filter(dem=dem, kernel_radius=kernel_radius)
        if not ndx == i:  # if not first surface
            relief_models_sum += (last_lpf_surface - lpf_surface)  # substitution of 2 consecutive lpf_surface
            nr_relief_models += 1
        last_lpf_surface = lpf_surface

    msrm_out = relief_models_sum / nr_relief_models

    return msrm_out

def get_msrm(dem_arr, dem_res_x, dem_res_y, dem_no_data):
    feature_min = 1                                                             # minimum size of the feature you want to detect in meters
    feature_max = 10                                                            # maximum size of the feature you want to detect in meters
    scaling_factor = 1                                                          # scaling factor
    msrm_arr = rvt.vis.msrm(
        dem=dem_arr,
        resolution=dem_res_x,
        feature_min=feature_min,
        feature_max=feature_max,
        scaling_factor=scaling_factor,
        ve_factor=1,
        no_data=dem_no_data
    )
    
    return msrm_arr


#def get_msrm(dem_arr, dem_res_x, dem_res_y, dem_no_data):
#    feature_min = 1
#    feature_max = 10
#    scaling_factor = 1
#    relief_arr = np.zeros_like(dem_arr)

#    for rad in range(feature_min, feature_max + 1):
#        kernel_radius = rad * scaling_factor
#        low_pass_filtered = gaussian_filter(dem_arr, sigma=kernel_radius)
#        relief = np.abs(dem_arr - low_pass_filtered)
#        relief_arr += relief

#    relief_arr /= (feature_max - feature_min + 1)

#    limit = 0.5
#    msrm_arr = np.clip(relief_arr, -limit, limit)
#    #msrm_arr*=0.0
#    return msrm_arr

def get_msrmZero(dem_arr, dem_res_x, dem_res_y, dem_no_data):
    msrm = np.zeros_like(dem_arr)
    return msrm



def get_hpass(dem_arr, dem_res_x=None, dem_res_y=None, dem_no_data=None):
    limit = 0.5
    pad = 15
    # pad images to remove edge artifacts during high pass filtering
    z = np.pad(dem_arr, pad, mode='edge')
    #z = z * 0.3048
    lPass = lowPass(z, 3)
    hPass = (z - lPass)
    hPass = np.clip(hPass[pad:-pad, pad:-pad], -limit, limit)
    return hPass

def resolve_path(file):
    FILE = Path(file).resolve()
    ROOT = FILE.parents[1]                                                      # root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))                                              # add ROOT to PATH
    return Path(os.path.relpath(ROOT, Path.cwd()))                              # relative

def update_config(ROOT, args):
    # add config file arguments to args
    with open(ROOT / args.config) as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args
