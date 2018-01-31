# # # HERALDO core library # # #

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
from skimage import io, feature, color, measure, draw, img_as_float, filters
import scipy.fftpack as fftpack
from PIL import Image

# normalise an array
def norm(a):
    return (a-sp.amin(a)) / sp.amax(a)

# returns 2D image data recorded by CCD
# nxs files are essentially HDF files
def load_nxs_data(fdir,fname,group,dset):
    with h5py.File(fdir + fname,'r') as f:
        return sp.squeeze(f[group]['scan_data'][dset])

# returns an estimated centre of the diffraction pattern
# uses RANSAC, outlier discarding fitting, of a circle to
# the diffraction rings from the beam stop (autocorrelation)
# Problem child
def get_offset(data, sigma=5):
    image = gaussian_filter(data,sigma)
    # image = data;

    # normalise the Image
    image = image/sp.amax(image)

    edges = feature.canny(image)
    coords = sp.column_stack(sp.nonzero(edges))

    model, inliers = measure.ransac(coords, measure.CircleModel,
                                    min_samples=500, residual_threshold=1,
                                    max_trials=1000)

    origin = (model.params[1],model.params[0])
    rr, cc = draw.circle_perimeter(int(model.params[0]),
                                   int(model.params[1]),
                                   int(model.params[2]),
                                   shape=image.shape)

    image[rr, cc] = 10

    return (model.params[1],model.params[0])

# returns an estimated angle of the diffraction line from the reference slit
def get_angle(data):
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(
    sp.arange(data.shape[0]), sp.argmax(gaussian_filter(data,10), axis=1))
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
def reconstruct(fdir, fname1,fname2, sigma):

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

    # sum_data = raw_1 + raw_2; dif_data = raw_1 - raw_2;
    try:
        origin = get_offset(sum_data, sigma)
    except:
        print('failed to make fit of the centre')
        return 0
    try:
        angle = get_angle(sum_data)
    except:
        print('failed to find angle')
    filtered_data = dif_data*differential_filter(dif_data, angle, origin)
    filtered_data = fftpack.fftshift(fftpack.fft2(filtered_data))
    final_data = phase_correction(filtered_data, origin)
    return final_data
