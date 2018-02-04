import h5py as hd
import scipy as sp
from skimage import feature, transform, color, exposure
from scipy import ndimage, fftpack, optimize
import matplotlib.pyplot as plt
from matplotlib import colors
import colorcet

class reconstructor():
    """docstring for reconstructor."""
    def __init__(self, fnum1, fnum2):
        super(reconstructor, self).__init__()
        self.raw_1 = self.load_nxs_data(fnum1)
        self.raw_2 = self.load_nxs_data(fnum2)
        self.sum_dif()

    def load_nxs_data(self,fname):
        # Accepts either file number or full name:
        # if input is number, make file fname
        if fname.isdigit() == True:
            fnum = fname
            fname = 'scan_' + str(fname) + '.nxs'
        # else assume the input is file name and make number
        else:
            fnum = ''.join([a for a in fname if a.isdigit()])
        # open the nxs file and roots around for the data array
        with hd.File(fname,'r') as f:
            # explore the contents and scan group
            group = [a for a in list(f.keys()) if a.find(str(fnum))][0]
            # explore the contents of the scan data and find nonzero array
            dset = [a for a in list(f[group]['scan_data'].keys()) if sp.squeeze(f[group]['scan_data'][a].size >= 10)][0]
            # make name for variable
            return sp.squeeze(f[group]['scan_data'][dset])

    def sum_dif(self):
        # optimising equalisation of input images to maximise signal to noise ratio, remove non magnetic signals
        def sum_abs_dif_data(ratio):
            return sp.sum(sp.absolute(self.raw_1 - ratio * self.raw_2))
        ratio = optimize.minimize(sum_abs_dif_data, 1).x[0]
        self.dif_data = self.raw_1 - ratio * self.raw_2
        self.sum_data = self.raw_1 + self.raw_2

    def auto_centre(self, data, sigma=0, count=10):
        # hough transform to fit circles and vote
        # Load picture and detect edges
        image = data / sp.amax(data)
        edges = feature.canny(image, sigma)

        # Detect two radii
        hough_radii = sp.arange(20, 35, 2)
        hough_res = transform.hough_circle(edges, hough_radii)

        accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii,total_num_peaks=count)

        self.centre = [sp.mean(cx),sp.mean(cy)]
        self.centre_radius = sp.mean(radii)

    def auto_angle(self, data, sigma=5):
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(
        sp.arange(data.shape[0]), sp.argmax(ndimage.gaussian_filter(data,sigma), axis=1))
        self.angle = sp.arctan(slope)

    def manual_angle(self, angle):
        print('\nwhy are you using this function? \n')
        self.angle = angle

    def display_callibration(self):
        # calculate the fit line from callibration parameters
        x = sp.arange(self.sum_data.shape[0])
        c = self.centre[0] - self.angle*self.centre[1]
        y = sp.sin(self.angle)*x + c

        plt.figure(figsize = (10,10));
        plt.imshow(self.sum_data, cmap = 'plasma')
        plt.scatter(*self.centre,marker='+',color='red')
        plt.plot(y,x,'r--')
        plt.show()

    # def beam_stop_blur(self, data):
    #     cx, cy = *self.centre
    #     x = sp.arange(data.shape[0]) - cx;
    #     y = sp.arange(data.shape[1]) - cy;
    #
    #     sp.ones(shape=)

    # perform filtering, inverse fourier transform and phase phase_correction
    # requires prior callibration
    def reconstruct(self):
        # differential_filter
        x,y = sp.meshgrid(sp.arange(self.sum_data.shape[0]),
                      sp.arange(self.sum_data.shape[1]))
        diff_filter = sp.pi * 2 * (+sp.cos(self.angle)*(x-self.centre[0]) \
        -sp.sin(self.angle)*(y-self.centre[1]))

        # apply differential_filter
        filtered_data = self.dif_data * diff_filter
        # fourier transform
        filtered_data = fftpack.fftshift(fftpack.fft2(filtered_data))
        # phase correct for offset fourier transform
        filtered_data = filtered_data * sp.exp(1j*2*sp.pi*(self.centre[0] * x / self.sum_data.shape[0] + self.centre[1] * y / self.sum_data.shape[1]))

        self.magnetic = filtered_data

    def display(self, data, cmap='coolwarm', linthresh=500):
        plt.figure()
        plt.imshow(data,
            norm=colors.SymLogNorm(linthresh=linthresh),
            origin='lower',
            cmap = cmap,
            interpolation='none'
        )
        plt.show()

    def phase_amp_data(self, data):
        # generate phase color array
        norm = plt.Normalize()
        phase_colors = colorcet.cm['cyclic_mygbm_30_95_c78_s25'](norm(sp.angle(data)))
        # generate amplitude greyscale array
        norm = plt.Normalize()
        amplitude_colors = colorcet.cm['linear_grey_10_95_c0'](norm(sp.absolute(data)**(0.2)))
        # combine amplitude and phase
        phase_amplitude = phase_colors*amplitude_colors
        # return
        return phase_amplitude

    def max_contrast(self):
    # rotate the reconstructed magnetic contrast
    # in the complex plane by a constant phase
    # such that the power in the real domain is maximised
        def sum_abs_imag(angle):
            temp = sp.exp(1j*angle) * self.magnetic
            return sp.sum(sp.absolute(temp))
        angle = optimize.minimize(sum_abs_imag, 0.1).x[0]
        return self.magnetic * sp.exp(1j*angle)

    def full_reconstruct(self):
        # check for subtracted 'dif_data', else generate
        # self.sum_dif() and check if callibration exists, if non try to auto detection
        if hasattr(self, 'sum_data'):
            pass
        else:
            self.sum_dif()
        if hasattr(self, 'angle'):
            pass
        else:
            self.auto_angle(self.sum_data)
        if hasattr(self, 'centre'):
            pass
        else:
            try:
                self.auto_centre(self.sum_data);
            except:
                print('failed to fit centre automatically\n set manually\n')
        # apply the reconstruction
        self.reconstruct()

    def save_image(self, name, data):
        plt.ioff
        plt.figure()
        plt.imshow(data,
            norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03),
            origin='lower',
            cmap = 'plasma',
            interpolation='none'
            )
        plt.savefig(name+'.png')
