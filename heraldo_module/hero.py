import h5py as hd
import scipy as sp
from skimage import feature, transform, color, exposure
from scipy import ndimage, fftpack
import matplotlib.pyplot as plt
from matplotlib import colors
import colorcet

class reconstructor():
    """docstring for reconstructor."""
    def __init__(self, fnum1, fnum2):
        super(reconstructor, self).__init__()
        self.raw_1 = self.load_nxs_data(fnum1)
        self.raw_2 = self.load_nxs_data(fnum2)

    def load_nxs_data(self,fnum):
        # convert file number into a file name
        fname = 'scan_' + str(fnum) + '.nxs'
        # open the nxs file
        with hd.File(fname,'r') as f:
            # explore the contents and scan group
            group = [a for a in list(f.keys()) if a.find(str(fnum))][0]
            # explore the contents of the scan data and find nonzero array
            dset = [a for a in list(f[group]['scan_data'].keys()) if sp.squeeze(f[group]['scan_data'][a].size >= 10)][0]
            # make name for variable
            return sp.squeeze(f[group]['scan_data'][dset])

    def sum_dif(self):
        # function left here in order to add in noise filtering
        self.dif_data = self.raw_1 - self.raw_2;
        self.sum_data = self.raw_1 + self.raw_2;

    def auto_centre(self, data, sigma=0, count=10):
        # hough transform to fit circles and vote
        # Load picture and detect edges
        image = data / sp.amax(data)
        edges = feature.canny(image, sigma)

        # Detect two radii
        hough_radii = sp.arange(20, 35, 2)
        hough_res = transform.hough_circle(edges, hough_radii)

        accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii,total_num_peaks=count)
        # Calculate the average consensus of the fitted circles
        x=0;y=0;
        image = color.gray2rgb(image)
        for center_y, center_x, radius in zip(cy, cx, radii):
            x = x + center_x; y = y + center_y
        x = x/count; y = y/count
        self.centre = [x,y]

    def manual_centre(self, x, y):
        self.centre = [x,y]

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

    def display(self, data):
        plt.figure()
        plt.imshow(data,
            norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03),
            origin='lower',
            cmap = 'plasma',
            interpolation='none'
        )
        plt.show()

    def phase_amp_data(self,data):
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

    def full_reconstruct(self):
        # check for subtracted 'dif_data', else generate
        # self.sum_dif() and check if callibration exists, if non try to auto detection
        if (self.angle == 0.0): self.auto_angle();
        if (self.centre == [0.0, 0.0]): self.auto_centre();

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
