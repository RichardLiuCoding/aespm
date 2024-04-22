import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
from igor2 import binarywave as bw


def load_ibw(file, ss=False):
    '''
    Load the ibw file as an IBWData object.

    Input:
        file     - String: path to the ibw file
        ss         - Boolean: if True then the ibw file will be treated as domain switching
                 spectroscopy file.
    Output:
        IBWData object:
        self.z          - Numpy array: 2D numpy array containing topography channel. 
                         Default is the "Height" channel.
        self.size       - float: Map size in the unit of meter
        self.mode        - String: Imaging mode. Currently support "AC Mode", "Contact Mode", "PFM Mode"
                         "SS Mode" and "DART Mode".
        self.header     - Dict: All the setup information.
        self.channels   - list: List of channel names
        self.data       - Numpy array: An array of all the saved image data in this ibw file in
                         the same order as self.channels
    Examples:
        
    '''
    
    # Return an IBWData object
    return IBWData(file)

class IBWData(object):
    '''
    Data structure for AR IBW maps.

    Attributes:
        self.z          - Numpy array: 2D numpy array containing topography channel. 
                         Default is the "Height" channel.
        self.size       - float: Map size in the unit of meter
        self.mode        - String: Imaging mode. Currently support "AC Mode", "Contact Mode", "PFM Mode"
                         "Spec" and "DART Mode".
        self.header     - Dict: All the setup information.
        self.channels   - list: List of channel names
        self.data       - Numpy array: An array of all the saved image data in this ibw file in
                         the same order as self.channels
        
    Methods:
        None
    '''
    def __init__(self, path):
        super(IBWData, self).__init__()

        self._load_ibw(path)

        # Spectroscopy files:
        if "ARDoIVCurve" in self.header:
            self.mode = "Spec"
            try:
                self._load_ss()
            except IndexError:
                pass
        # Image files:
        else:
            try:
                self.size = self.header['ScanSize']
                self.mode = self.header['ImagingMode']
                z_index = self.channels.index('Height')
                self.z = self.data[z_index]
                
                try:
                    # Separate DART mode from general PFM mode
                    if self.mode == "PFM Mode":
                        if len(self.channels) > 4:
                            self.mode = "DART Mode"
                            self.channels = ['Height', 'Amplitude1', 'Amplitude2', 'Phase1', 'Phase2', 'Frequency']
                        else:
                            self.channels = ['Height', 'Amplitude', 'Deflection', 'Phase']
                except IndexError:
                    pass
            except KeyError:
                pass

    def _load_ibw(self, path):

        t = bw.load(path)
        wave = t.get('wave')

        # Decode the notes section to parse the header
        if isinstance(wave['note'], bytes):
            try:
                parsed_string = wave['note'].decode('utf-8').split('\r')
            except:
                parsed_string = wave['note'].decode('ISO-8859-1').split('\r')

        # Load the header
        self.header = {}

        for item in parsed_string:
            try:
                key, value = item.split(':', 1)
                value = value.strip()  # Remove leading/trailing whitespace
            except ValueError:
                continue  # For items that do not split correctly

            # Determine the data type of the value and convert
            if '.' in value or 'e' in value:  # Floating point check
                try:
                    self.header[key] = float(value)
                except ValueError:
                    self.header[key] = value
            elif value.lstrip('-').isdigit():  # Integer check
                self.header[key] = int(value)
            else:
                self.header[key] = value

        # Load the data
        # Transpose the data matrix
        self.data = wave['wData'].T
        # data = wave['wData']

        # Load the channel names
        self.channels = [self.header.get(f'Channel{i+1}DataType', 'Unknown') for i in range(np.shape(self.data)[0])]

    def _load_ss(self, nan=True):

        if nan is True:
            bias_raw= self.data[-1]
            index_not_nan = np.where(~np.isnan(bias_raw))

            bias = bias_raw[index_not_nan]
            amp, phase1, phase2 = self.data[2][index_not_nan], self.data[3][index_not_nan], self.data[4][index_not_nan]
        else:
            bias = self.data[-1]
            amp = self.data[2]
            phase1 = self.data[3]
            phase2 = self.data[4]

        phase1 = self._correct_phase_wrapping(phase1)
        phase2 = self._correct_phase_wrapping(phase2)

        # Let's count how many times the bias has changed (on->off, off->on)
        index_bp = np.where(np.diff(bias) != 0)[0] + 1
        index_bp = np.concatenate([[0], index_bp])

        # Output array length
        length = len(index_bp) // 2

        bias_on = np.zeros(length)
        bias_off = np.zeros(length)

        phase1_on = np.zeros(length)
        phase1_off = np.zeros(length)

        phase2_on = np.zeros(length)
        phase2_off = np.zeros(length)

        amp_on = np.zeros(length)
        amp_off = np.zeros(length)

        for i in range(length * 2-1):
            if i % 2 == 0: # bias off
                phase1_off[i//2] = np.mean(phase1[index_bp[i]:index_bp[i+1]])
                phase2_off[i//2] = np.mean(phase2[index_bp[i]:index_bp[i+1]])
                amp_off[i//2] = np.mean(amp[index_bp[i]:index_bp[i+1]])
                bias_off[i//2] = np.mean(bias[index_bp[i]:index_bp[i+1]])
            else:
                bias_on[i//2] = np.mean(bias[index_bp[i]:index_bp[i+1]])
                phase1_on[i//2] = np.mean(phase1[index_bp[i]:index_bp[i+1]])
                phase2_on[i//2] = np.mean(phase2[index_bp[i]:index_bp[i+1]])
                amp_on[i//2] = np.mean(amp[index_bp[i]:index_bp[i+1]])
        self.bias = bias_on[1:]
        self.phase1_on = phase1_on[1:]
        self.phase1_off = phase1_off[1:]
        self.phase2_on = phase2_on[1:]
        self.phase2_off = phase2_off[1:]
        self.amp_on = amp_on[1:] * np.cos(phase2_off[1:]/180*np.pi)
        self.amp_off = amp_off[1:] * np.cos(phase1_off[1:]/180*np.pi)

        # return bias[1:], amp_off[1:], phase1_off[1:], phase2_off[1:]

    def _correct_phase_wrapping(self, ph, lower=-90, upper=270):
        '''
        Correct the phase wrapping in Jupiter.
        
        Input:
            Ph     - Array: array of phase values
            lower - float: lower bound of phase limit in your instrument
            upper - float: upper bound of phase limit in your instrument
        Output:
            ph_shift - Array: phase with wrapping corrected
        '''
        # Use the phase value measured at last pixel as the offset in the lock-in
        ph_shift = ph - ph[-1]

        index_upper = np.where(ph_shift > upper)
        index_lower = np.where(ph_shift < lower)
        ph_shift[index_upper] -= 360
        ph_shift[index_lower] += 360

        return ph_shift

# Function to read and plot all the ibw files

def display_ibw_folder(folder, mode=None, key=['Height'], cmaps=None, paras=None, save=None, **kwargs):
    '''
    Display all the ibw files with specified modes in a given folder.

    Input:
        folder  - Required: path to the folder to be explored
        mode    - Optional: if not given, all available modes will be displayed
                            can be 'AC Mode', 'PFM Mode', 'Contact Mode', 'Spec' and 'DART Mode'
                            To be added: 'SKPM Mode'
        key     - Optional: list of channels to be displayed
        cmaps     - Optional: list of color maps that will be used for each channels in key
        paras   - Optional: to be added for visualizing imaging parameters like surface voltage
        save     - Optional: if None, no image will be saved. If not None, each image will be 
                            saved as fileName + save
        **kwarg - Optional: Additional keyword arguments are sent to imshow().

    Output:
        ibw_files   -list: ibw file names in the same order as they are displayed
        data        -list: SciFiReader object of each ibw file displayed

    Example use:
        ibw_files, data = display_ibw(folder, mode='AC Mode', key=['Height', 'ZSensor'])
    '''
    file_names = os.listdir(folder)
    file_names = sorted(file_names)

    file_displayed = []
    out = []

    if key is not None:
        if not isinstance(key, list):
            key = list(key)

    display_index = 0
    for index, file in enumerate(file_names):
        if file.endswith('.ibw'):
            fname = os.path.join(folder, file)

            try:
                t = load_ibw(fname)

                if save is not None:
                    save = save + ' ' + file.split('.')[0]

                if t.mode != 'Spec': # skip the spectrum ibw files
                    if mode == None:
                        display_ibw(t, key=key, display_index=display_index, cmaps=cmaps, save=save, **kwargs)
                        file_displayed.append(fname)
                        out.append(t)
                        display_index += 1
                    elif mode == t.mode:
                        display_ibw(t, key=key, display_index=display_index, cmaps=cmaps, save=save, **kwargs)
                        file_displayed.append(fname)
                        out.append(t)
                        display_index += 1
                else:
                    pass
            except TypeError:
                pass

    return file_displayed, out

def display_ibw(file, key=None, titles=None, display_index=None, cmaps=None, save=None, **kwargs):
    '''
    Display a single ibw with specified by the file path.

    Input:
        file    - Required: path to the file to be displayed or the loaded ibw object returned by load_ibw()
        key     - Optional: list of channels to be displayed
        titles     - Optional: list of titles corresponding to the key
        display_index - Optional: index provided by display_ibw_folder() function
        cmaps     - Optional: list of color maps that will be used for each channels in key
        save     - Optional: if None, no image will be saved. If not None, each image will be 
                            saved as fileName + save
        **kwarg - Optional: Additional keyword arguments are sent to imshow().

    Output:
        ibw_files   -list: ibw file names in the same order as they are displayed
        data        -list: SciFiReader object of each ibw file displayed

    Example use:
        ibw_files, data = display_ibw(folder, mode='AC Mode', key=['Height', 'ZSensor'])
    '''

    try:
        if type(file) is str:
            t = load_ibw(file)
        else:
            t = file

        if key is not None:
            if not isinstance(key, list):
                key = list(key)
        else:
            key = t.channels

        if titles is not None:
            if not isinstance(titles, list):
                titles = list(titles)

        if cmaps is not None:
            if not isinstance(cmaps, list):
                cmaps = list(cmaps)

        if t.mode != 'Spec': # skip the spectrum ibw files
            indices = find_channel(obj=t, key=key)
            if len(indices) == 1: # Only one channel will be displayed
                plt.figure(figsize=[4,4])
                to_plot = t.data[indices[0]]
                if cmaps is None:
                    im = plt.imshow(to_plot, extent=[0, t.size*1e6, 0, t.size*1e6], **kwargs)
                else:
                    im = plt.imshow(to_plot, extent=[0, t.size*1e6, 0, t.size*1e6], cmap=cmaps[0], **kwargs)
                if display_index is None:
                    title = "{}: {}".format(t.mode, key[0]) if not titles else titles
                else:
                    title = "{}: {}-{}".format(display_index, t.mode, key[0]) if not titles else titles

                plt.title(title)
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.tight_layout()
                if save is not None:
                    plt.savefig('{}.png'.format(save), dpi=400, bbox_inches='tight', pad_inches=0.1)
            else:
                n_cols = len(indices)
                fig,ax=plt.subplots(1, n_cols, figsize=[n_cols*3+1, 3])
                for i in range(len(indices)):
                    to_plot = t.data[indices[i]]
                    if cmaps is None:
                        im = ax[i].imshow(to_plot, extent=[0, t.size*1e6, 0, t.size*1e6], **kwargs)
                    else:
                        im = ax[i].imshow(to_plot, extent=[0, t.size*1e6, 0, t.size*1e6], cmap=cmaps[i], **kwargs)
                    divider = make_axes_locatable(ax[i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)
                    if titles is None:
                        if not i:
                            ax[i].set_title("{}: {}-{}".format(display_index, t.mode, t.channels[indices[i]]))
                        else:
                            ax[i].set_title("{}".format(t.channels[indices[i]]))
                    else:
                        ax[i].set_title(titles[i])
                    plt.tight_layout()
                    if save is not None:
                        plt.savefig('{}.png'.format(save), dpi=400, bbox_inches='tight', pad_inches=0.1)
        else:
            pass
    except TypeError:
        pass

def find_channel(obj, key):
    if key is None:
        return np.arange(len(obj.channels))
    else:
        index = []
        channels = obj.channels

        for item in key:
            if item in channels:
                index.append(channels.index(item))
        return index

def plane_subtract(A, pts=None):
    '''
    Remove a plane fitting from a map. If pts are given, the plane will be
    fitted based on pts. Otherwise, the three corner points will used to fit 
    for the plane.

    Inputs:
        A        - Required : A 2D numpy array.
        pts     - Optional : coordinates in pixels for three points: [pt1, pt2, pt3]
    Returns:
        A_out   - Map after a plane fit is removed
    '''

    H, W = np.shape(A)
    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)
    
    if pts == None:
        p0 = [W-1,0]
        p1 = [0,H-1]
        p2 = [W-1,H-1]
    else:
        p0, p1, p2 = pts
    
    Ax, Ay, Az = *p0, A[int(p0[1]), int(p0[0])]
    Bx, By, Bz = *p1, A[int(p1[1]), int(p1[0])]
    Cx, Cy, Cz = *p2, A[int(p2[1]), int(p2[0])]
    
    a = (By-Ay)*(Cz-Az) - (Cy-Ay)*(Bz-Az)
    b = (Bz-Az)*(Cx-Ax) - (Cz-Az)*(Bx-Ax)
    c = (Bx-Ax)*(Cy-Ay) - (Cx-Ax)*(By-Ay)
    d = -(a*Ax+b*Ay+c*Az)
    z = -(a*X+b*Y+d) / c
    return A - z

def lineSubtract(data, n=1, maskon=False, thres=4, M=4, normalize=True, colSubtract=False):
    '''
    Remove a polynomial background from the data line-by-line, with
    the option to skip pixels within certain distance away from
    impurities.  If the data is 3D (eg. 3ds) this does a 2D background
    subtract on each layer independently.  Input is a numpy array.

    Inputs:
        data    -   Required : A 1D, 2D or 3D numpy array.
        n       -   Optional : Degree of polynomial to subtract from each line.
                               (default : 1).
        maskon  -   Optional : Boolean flag to determine if the impurty areas are excluded.
        thres   -   Optional : Float number specifying the threshold to determine
                               if a pixel is impurity or bad pixels. Any pixels with intensity greater
                               than thres*std will be identified as bad points.
        M       -   Optional : Integer number specifying the box size where all pixels will be excluded
                               from poly fitting.
        normalize - Optional : Boolean flag to determine if the mean of a layer
                               is set to zero (True) or preserved (False).
                               (default : True)
        colSubtract - Optional : Boolean flag (False by default) to determine if polynomial background should also be subtracted column-wise

    Returns:
        subtractedData  -   Data after removing an n-degree polynomial
    '''
    # Polyfit for lineSubtraction excluding impurities.
    def filter_mask(data, thres, M, D):
        filtered = data.copy()
        if D == 1:
            temp = np.gradient(filtered)
            badPts = np.where(np.abs(temp-np.mean(temp))>thres*np.std(temp))
            for ix in badPts[0]:
                filtered[max(0, ix-M) : min(data.shape[0], ix+M+1)] = np.nan
            return filtered
        elif D == 2:
            temp = np.gradient(filtered)[1]
            badPts = np.where(np.abs(temp-np.mean(temp))>thres*np.std(temp))
            for ix, iy in zip(badPts[1], badPts[0]):
                filtered[max(0, iy-M) : min(data.shape[0], iy+M+1),
                                  max(0, ix-M) : min(data.shape[1], ix+M+1)] = np.nan
            return filtered

    def subtract_mask(data, n, thres, M, D):
        d = data.shape[0]
        x = np.linspace(0,data.shape[-1]-1,data.shape[-1])
        filtered = filter_mask(data, thres, M, D)
        output = data.copy()
        if D == 1:
            index = np.isfinite(filtered)
            try:
                popt = np.polyfit(x[index], data[index], n)
                output = data - np.polyval(popt, x)
            except TypeError:
                raise TypeError('Empty x-array encountered. Please use a larger thres value.')
            return output
        if D == 2:
            for i in range(d):
                index = np.isfinite(filtered[i])
                try:
                    popt = np.polyfit(x[index], data[i][index], n)
                    output[i] = data[i] - np.polyval(popt, x)
                except TypeError:
                    raise TypeError('Empty x-array encountered. Please use a larger thres value.')
            return output

    if maskon is not False:
        if len(data.shape) == 3:
            output = np.zeros_like(data)
            for ix, layer in enumerate(data):
                output[ix] = subtract_mask(layer, n, thres, M, 2)
            return output
        elif len(data.shape) == 2:
            return subtract_mask(data, n, thres, M, 2)
        elif len(data.shape) == 1:
            return subtract_mask(data, n, thres, M, 1)
        else:
            raise TypeError('Data must be 1D, 2D or 3D numpy array.')

    # The original lineSubtract code.
    def subtract_1D(data, n):
        x = np.linspace(0,1,len(data))
        popt = np.polyfit(x, data, n)
        return data - np.polyval(popt, x)
    def subtract_2D(data, n):
        if normalize:
            norm = 0
        else:
            norm = np.mean(data)
        output = np.zeros_like(data)
        for ix, line in enumerate(data):
            output[ix] = subtract_1D(line, n)
        if colSubtract:
            temp = np.zeros_like(data)
            for ix, line in enumerate(np.transpose(output)):
                temp[ix] = subtract_1D(line, n)
            output = np.transpose(temp)
        return output + norm

    if len(data.shape) == 3:
        output = np.zeros_like(data)
        for ix, layer in enumerate(data):
            output[ix] = subtract_2D(layer, n)
        return output
    elif len(data.shape) == 2:
        return subtract_2D(data, n)
    elif len(data.shape) == 1:
        return subtract_1D(data, n)
    else:
        raise TypeError('Data must be 1D, 2D or 3D numpy array.')
    

def fft(dataIn, window='None', output='absolute', zeroDC=False, beta=1.0,
        units='None'):
    '''
    Compute the fast Frouier transform of a data set with the option to add
    windowing.

    Inputs:
        dataIn    - Required : A 1D, 2D or 3D numpy array
        window  - Optional : String containing windowing function used to mask
                             data.  The options are: 'None' (or 'none'), 'bartlett',
                             'blackman', 'hamming', 'hanning' and 'kaiser'.
        output  - Optional : String containing desired form of output.  The
                             options are: 'absolute', 'real', 'imag', 'phase'
                             or 'complex'.
        zeroDC  - Optional : Boolean indicated if the centeral pixel of the
                                FFT will be set to zero.
        beta    - Optional : Float used to specify the kaiser window.  Only
                               used if window='kaiser'.
        units   - Optional : String containing desired units for the FFT.
                             Options: 'None', or 'amplitude' (in the future, I
                             might add "ASD" and "PSD".

    Returns:
        fftData - numpy array containing FFT of data

    Usage:
        fftData = fft(data, window='None', output='absolute', zeroDC=False,
                      beta=1.0)
    '''
    def ft2(data):
        ftData = np.fft.fft2(data)
        if zeroDC:
            ftData[0,0] = 0
        return np.fft.fftshift(ftData)

    outputFunctions = {'absolute':np.absolute, 'real':np.real,
                       'imag':np.imag, 'phase':np.angle, 'complex':(lambda x:x) }

    windowFunctions = {'None':(lambda x:np.ones(x)), 'none':(lambda x:np.ones(x)),
                       'bartlett':np.bartlett, 'blackman':np.blackman,
                       'hamming':np.hamming, 'hanning':np.hanning,
                       'kaiser':np.kaiser , 'sine':windows.cosine}

    outputFunction = outputFunctions[output]
    windowFunction = windowFunctions[window]

    data = dataIn.copy()
    if zeroDC:
        if len(data.shape) == 3:
            for ix, layer in enumerate(data):
                data[ix] -= np.mean(layer)
        else:
            data -= np.mean(data)

    if len(data.shape) != 1:
        if window == 'kaiser':
            wX = windowFunction(data.shape[-2], beta)[:,None]
            wY = windowFunction(data.shape[-1], beta)[None,:]
        else:
            wX = windowFunction(data.shape[-2])[:,None]
            wY = windowFunction(data.shape[-1])[None,:]
        W = wX * wY
        if len(data.shape) == 2:
            wData = data * W
            ftData = outputFunction(ft2(wData))
        elif len(data.shape) == 3:
            wTile = np.tile(W, (data.shape[0],1,1))
            wData = data * wTile
            if output == 'complex':
                ftData = np.zeros_like(data, dtype=np.complex)
            else:
                ftData = np.zeros_like(data)
            for ix, layer in enumerate(wData):
                ftData[ix] = outputFunction(ft2(layer))
        else:
            print('ERR: Input must be 1D, 2D or 3D numpy array')

    else:
        if window == 'kaiser':
            W = windowFunction(data.shape[0], beta)
        else:
            W = windowFunction(data.shape[0])
        wData = data * W
        ftD = np.fft.fft(wData)
        ftData = outputFunction(np.fft.fftshift(ftD))
    if units == 'amplitude':
        if len(data.shape) == 3:
            datashape = data[0].shape
        else:
            datashape = data.shape
        for size in datashape:
            ftData /= size
            if window == 'hanning':
                ftData *= 2
            elif window == 'None' or window == 'none':
                pass
            else:
                print('WARNING: The window function "%s" messes up the FT units' %
                        window)
    return ftData