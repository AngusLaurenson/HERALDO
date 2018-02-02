# # # HERALDO core library # # #

'''
It also adds batch processing and increased robustness of the fitting

This library contains a set of basic, lower level functions for use in a jupyter (ipython) notebook for the purposes of analysing x-ray holographic diffraction patterns that have magnetic contrast due to XMCD.

USE:
import heraldo as he

raw_data = he.load_nxs_data(dir, fname, group, dset)

WISH LIST OF FEATURES:
 - hot pixel detection
 - move all power into real part
 - fast manual correction
'''

# import dependancies
import h5py
import scipy as sp
import re
from scipy.ndimage.filters import convolve, gaussian_filter, gaussian_gradient_magnitude
from skimage import io, feature, color, measure, draw, img_as_float, filters, exposure
import scipy.fftpack as fftpack
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

class callibration(object):
    """docstring for callibration.
    This object records the callibration for a run of heraldo experiments"""
    def __init__(self):
        super(callibration, self).__init__()
        self.origin = [0.0,0.0]
        self.angle = 0.0

    def manual_origin(self, x, y):
        self.origin = [x, y]

    def auto_origin(self, data, sigma):
        try:
            self.origin     = get_offset_hough(data, sigma)
        except:
            print('auto callibration failed :( \n)')
            return 0
    def auto_angle(self, data, sigma = 10):
        self.angle          = get_angle(data, sigma)

# normalise an array
def norm(a):
    return (a-sp.amin(a)) / sp.amax(a)

# returns 2D image data recorded by CCD
# nxs files are essentially HDF files
def load_nxs_data(fdir,fname,group='let',dset='data_03'):
    # if no group name is given it just guesses
    if (group == 'let'):
        group = fname.split('_')[0]+'x_'+fname.split('_')[1].split('.')[0];

    # open the nxs file and return the array
    with h5py.File(fdir + fname,'r') as f:
        return sp.squeeze(f[group]['scan_data'][dset])

# returns an estimated centre of the diffraction pattern
# uses RANSAC, outlier discarding fitting, of a circle to
# the diffraction rings from the beam stop (autocorrelation)
# Problem child
def get_offset_ransac(data, sigma=2):

    # normalise the Image
    image = data/sp.amax(data)

    # canny edge detection to return boolean array
    edges = feature.canny(image, sigma)
    coords = sp.column_stack(sp.nonzero(edges))

    # fit circle to canny edges
    model, inliers = measure.ransac(coords, measure.CircleModel,
                                    min_samples=500, residual_threshold=1,
                                    max_trials=1000)
    # return the origin of the array
    return (model.params[1],model.params[0])

def get_offset_hough(data, sigma=0, circles=20):

    # Load picture and detect edges
    image = data/sp.amax(data)
    edges = canny(image, sigma)

    # Detect two radii
    hough_radii = np.arange(20, 35, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent N circles
    N = 20
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=N)
    # Calculate the average consensus of the fitted circles
    x=0;y=0;
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        x = x + center_x; y = y + center_y

    return (x,y)

# returns an estimated angle of the diffraction line from the reference slit
def get_angle(data, sigma=10):
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(
    sp.arange(data.shape[0]), sp.argmax(gaussian_filter(data,sigma), axis=1))
    return sp.arctan(slope)

# returns a 2D array of a constant gradient
# zero centred on the slit diffraction line
# when applied in the diffraction space, effectively
# performs real space gradient operation along the
# axis of the reference slit
def differential_filter(data,theta,origin):
    x,y = sp.meshgrid(sp.arange(data.shape[0]),
                  sp.arange(data.shape[1]))
    return sp.pi * 2 * (+sp.cos(theta)*(x-origin[0])-sp.sin(theta)*(y-origin[1]))

# create a plane wave array to correct the phase shift introduced
# by the fourier transform as the centre of the diffraction pattern
# is displaced from the origin of the array (at [0,0]).
# See the Fourier shift theorem.
def phase_correction(data, origin):
    x,y = sp.meshgrid(sp.arange(data.shape[0]),
                     sp.arange(data.shape[1]))
    return data * sp.exp(1j*2*sp.pi*(origin[0] * x / data.shape[0] + origin[1] * y / data.shape[1]))

# function to do it all in a oner...
def reconstruct(fdir, fname1,fname2, calibrator):

    # words = re.split(('_|\.'), fname1); dname1 = words[0] + words[2] + '_' + words[1]
    # words = re.split(('_|\.'), fname2); dname2 = words[0] + words[2] + '_' + words[1]
    group1 = fname1.split('_')[0]+'x_'+fname1.split('_')[1].split('.')[0];
    group2 = fname2.split('_')[0]+'x_'+fname2.split('_')[1].split('.')[0];

    raw_1 = load_nxs_data(fdir,fname1,group1,'data_03')
    raw_2 = load_nxs_data(fdir,fname2,group2,'data_03')

    # median filter to remove hot pixels
    # raw_1 = filters.median(raw_1)
    # raw_2 = filters.median(raw_2)

    sum_data = norm(raw_1)+norm(raw_2); dif_data = norm(raw_1)-norm(raw_2)

    try:
        origin = calibrator.origin
    except:
        print('failed to make fit of the centre')
        return 0
    try:
        angle = calibrator.angle
    except:
        print('failed to find angle')
    filtered_data = dif_data*differential_filter(dif_data, angle, origin)
    filtered_data = fftpack.fftshift(fftpack.fft2(filtered_data))
    final_data = phase_correction(filtered_data, origin)
    return final_data
