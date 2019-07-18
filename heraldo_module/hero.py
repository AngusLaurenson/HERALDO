import h5py as hd
import scipy as sp
import fabio
from skimage import feature, transform, color, exposure
from scipy import ndimage, fftpack, optimize
import matplotlib.pyplot as plt
from matplotlib import colors
import colorcet

class reconstructor():
    """docstring for reconstructor."""
    def __init__(self, fname1, fname2):
        super(reconstructor, self).__init__()
        self.raw_1 = self.load_data(fname1)
        self.raw_2 = self.load_data(fname2)

    # logic to open any data File
    def load_data(self, fname):
        try:
            return fabio.open(fname).data
        except:
            pass
        try:
            return self.load_nxs_data(fname)
        except:
            print('failure to read data')
            pass

    def load_nxs_data(self,fname):
        # Accepts either file number or full name:
        # if input is number, make file fname
        if fname.isdigit() == True:
            fnum = fname
            fname = 'scanx_' + str(fname) + '.nxs'
        # else assume the input is file name and make number
        else:
            fnum = ''.join([a for a in fname.split('/')[-1] if a.isdigit()])
        # open the nxs file and roots around for the data array
        with hd.File(fname,'r') as f:
            # explore the contents and scan group
            group = [a for a in list(f.keys()) if a.find(str(fnum))][0]
            # explore the contents of the scan data and find nonzero array
            dset = [a for a in list(f[group]['scan_data'].keys()) if sp.squeeze(f[group]['scan_data'][a].size >= 200)][0]
            # make name for variable
            return sp.squeeze(f[group]['scan_data'][dset])

    def sum_dif(self, equalisation=True):
        # optimising equalisation of input images to maximise signal to noise ratio, remove non magnetic signals

        # this section does not cancel out the aperture in the magnetic images
        # it is likely that normalising over a subset of the image would be better
        if equalisation is True and hasattr(self, 'ratio') is False:
            def sum_abs_dif_data(ratio):
                return sp.sum(sp.absolute(self.raw_1[:500,1500:] - ratio * self.raw_2[:500,1500:]),axis=(0,1))

            ratio = optimize.minimize(sum_abs_dif_data, 1).x[0]
            self.ratio = ratio

        elif equalisation is False:
            self.ratio = 1

        self.dif_data = self.raw_1 - self.ratio * self.raw_2
        self.sum_data = self.raw_1 + self.ratio * self.raw_2
        self.div_data = self.raw_1/self.raw_2

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

    def display_callibration(self):
        # calculate the fit line from callibration parameters
        x = sp.arange(self.sum_data.shape[0])
        c = self.centre[0] - self.angle*self.centre[1]
        y = sp.sin(self.angle)*x + c

        plt.figure();
        plt.imshow(self.sum_data, cmap = 'plasma')
        plt.scatter(*self.centre,marker='x',color='orange')
        plt.plot(y,x,'g--')
        plt.show()

    def beam_stop_blur(self, data, sigma=3):
        index = sp.indices(self.sum_data.shape)
        index[0,:,:] = index[0,:,:] - self.centre[1]
        index[1,:,:] = index[1,:,:] - self.centre[0]

        mask = 1-ndimage.gaussian_filter(sp.where(sp.sum(index**2, axis=0)>(2*self.centre_radius)**2+40, 0.0, 1.0), sigma)
        return data*mask

    def beam_stop_stopper(self, data, sigma=5, edge=60):
        '''Uses circular smoothed mask to cutout beamstop at centre'''
        radius = self.centre_radius + edge
        # setup the metric space and empty mask
        position = sp.array(self.centre)
        indices = sp.indices(data.shape)
        mask = sp.zeros(data.shape)
        # filled circle of ones at the centre
        mask[sp.where(sp.sum((indices-position[:,None,None])**2,axis=0) < radius**2)] = 1
        # smooth the circle of ones out
        mask = sp.ndimage.gaussian_filter(mask,sigma)
        # apply inverse mask to the data and return
        return data * (1 - mask/sp.amax(mask))

    # perform filtering, inverse fourier transform and phase phase_correction
    # requires prior callibration
    def differential_filtering(self, data):
        # differential_filter
        x,y = sp.meshgrid(sp.arange(data.shape[1]),
                      sp.arange(data.shape[0]))
        self.diff_filter = sp.pi * 2 * (+sp.cos(self.angle)*(x-self.centre[0]) \
        -sp.sin(self.angle)*(y-self.centre[1]))

        # apply differential_filter
        filtered_data = data * self.diff_filter
        return  filtered_data

    def fourier_2D(self, data):
        # fourier transform
        return fftpack.fftshift(fftpack.fft2(data),axes=(0,1))

    def offset_correction(self, data):
        # phase correct for offset fourier transform
        x,y = sp.meshgrid(sp.arange(data.shape[1]),
                      sp.arange(data.shape[0]))

        data = data * sp.exp(1j*2*sp.pi*(self.centre[0] * x / self.sum_data.shape[0] + self.centre[1] * y / self.sum_data.shape[1]))
        return data

    def display(self, data, cmap='coolwarm', linthresh=10):
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

    def max_real(self, data):
        angles = sp.ravel(sp.angle(data))
        amplitudes = sp.ravel(sp.absolute(data))
        zipped = list(zip(angles,amplitudes))
        res = sorted(zipped, key = lambda x : x[0])
        angles, amplitudes = list(zip(*res))
        filtered = sp.ndimage.gaussian_filter1d(amplitudes, 100, mode='wrap')
    #     print(angles[sp.where(filtered == sp.amax(filtered))[0][0]])
        return angles[sp.where(filtered == sp.amax(filtered))[0][0]], angles, amplitudes
        
        
        return angle, angle2

    def reconstruct(self, blur = True, equalisation=True):
        # check for subtracted 'dif_data', else generate
        # self.sum_dif() and check if callibration exists,
        # if non try to auto detection
        if hasattr(self, 'sum_data'):
            pass
        else:
            # create sum and difference images with automatic equalisation
            self.sum_dif(equalisation=equalisation)
        if hasattr(self, 'angle'):
            pass
        else:
            # calcualte automatically the angle of diffraction line
            self.auto_angle(self.sum_data)
        if hasattr(self, 'centre'):
            pass
        else:
            try:
                # find centre of diffraction rings
                self.auto_centre(self.sum_data);
            except:
                print('failed to fit centre automatically\n set manually\n')

        # define intermediate data arrays
        dif = self.dif_data.copy()
        sum = self.sum_data.copy()
        div = self.div_data.copy()

        # beam stop blurr application to the images
        dif = self.beam_stop_stopper(dif, 10, 50)
        sum = self.beam_stop_stopper(sum, 10, 50)
        div = self.beam_stop_stopper(div, 10, 50)

        dif = self.differential_filtering(dif)
        sum = self.differential_filtering(sum)
        div = self.differential_filtering(div)

        dif = self.fourier_2D(dif)
        sum = self.fourier_2D(sum)
        div = self.fourier_2D(div)

        dif = self.offset_correction(dif)
        sum = self.offset_correction(sum)
        div = self.offset_correction(div)


        # assign attributes
        self.magnetic = dif
        self.charge = sum
        self.magnetic_div = div

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
