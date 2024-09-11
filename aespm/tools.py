import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
from igor2 import binarywave as bw

from scipy.interpolate import interp1d
import scipy.optimize as opt
import scipy.ndimage as snd
from scipy.signal import butter, filtfilt, fftconvolve, hilbert, correlate, windows

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
    return IBWData(file, ss=ss)

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
    def __init__(self, path, ss=False):
        super(IBWData, self).__init__()

        self._load_ibw(path)

        # Spectroscopy files:
        if ss == True:
            self.mode = "Spec"
            try:
                self._load_ss()
            except IndexError:
                pass
        elif "ARDoIVCurve" in self.header:
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

    def _load_ss(self, nan=True, drop=0.1):

        if nan is True:
            bias_raw= self.data[-1]
            index_not_nan = np.where(~np.isnan(bias_raw))

            bias = bias_raw[index_not_nan]
            amp, phase1, phase2, freq = self.data[2][index_not_nan], self.data[3][index_not_nan], \
                        self.data[4][index_not_nan], self.data[5][index_not_nan]
        else:
            bias = self.data[-1]
            amp = self.data[2]
            phase1 = self.data[3]
            phase2 = self.data[4]
            freq = self.data[5]

        #Correcting first without offset to calculate the drive parameters
        phase1 = self._correct_phase_wrapping(phase1, offset_correction=False)
        phase2 = self._correct_phase_wrapping(phase2, offset_correction=False)

        df = self.header['DFRTFrequencyWidth']

        a_dr, ph_dr, q = self._calc_drive_params(amp, amp, phase1/180*np.pi, phase2/180*np.pi, freq, df)

        phase1 = self._correct_phase_wrapping(phase1)
        phase2 = self._correct_phase_wrapping(phase2)
        ph_dr  = self._correct_phase_wrapping(ph_dr/np.pi*180)

        # Let's count how many times the bias has changed (on->off, off->on)
        index_bp = np.where(np.diff(bias) != 0)[0] + 1
        # We define the width of applied voltage by the first non-zero voltage plateau
        index_delta = index_bp[1] - index_bp[0]
        # We drop the first segment of zero bias signals as this is the initial settling time
        index_bp = np.concatenate([[index_bp[0]-index_delta], index_bp])

        # Output array length
        length = len(index_bp) // 2

        bias_on, bias_off = np.zeros(length), np.zeros(length)

        phase1_on,phase1_off = np.zeros(length), np.zeros(length)
        phase2_on, phase2_off  = np.zeros(length), np.zeros(length)
        amp_on, amp_off = np.zeros(length), np.zeros(length)
        freq_on, freq_off = np.zeros(length), np.zeros(length)

        amp_dr_on, amp_dr_off = np.zeros(length), np.zeros(length)
        phase_dr_on, phase_dr_off = np.zeros(length), np.zeros(length)
        q_on, q_off = np.zeros(length), np.zeros(length)

        # We drop the first and last 10% data to avoid oscillation after bias change (10% settling time)
        skip = int(drop * index_delta)
        
        for i in range(length * 2-1):
            start = index_bp[i] + skip
            end = index_bp[i+1] - skip
            if i % 2 == 0: # bias off
                phase1_off[i//2] = np.mean(phase1[start:end])
                phase2_off[i//2] = np.mean(phase2[start:end])
                amp_off[i//2] = np.mean(amp[start:end])
                freq_off[i//2] = np.mean(freq[start:end])
                bias_off[i//2] = np.mean(bias[start:end])

                phase_dr_off[i // 2] = np.mean(ph_dr[start:end])
                q_off[i // 2] = np.mean(q[start:end])
                amp_dr_off[i // 2] = np.mean(a_dr[start:end])

            else:
                bias_on[i//2] = np.mean(bias[start:end])
                phase1_on[i//2] = np.mean(phase1[start:end])
                phase2_on[i//2] = np.mean(phase2[start:end])
                amp_on[i//2] = np.mean(amp[start:end])
                freq_on[i//2] = np.mean(freq[start:end])

                phase_dr_on[i // 2] = np.mean(ph_dr[start:end])
                q_on[i // 2] = np.mean(q[start:end])
                amp_dr_on[i // 2] = np.mean(a_dr[start:end])

        self.bias = bias_on
        self.phase1_on = phase1_on
        self.phase1_off = phase1_off
        self.phase2_on = phase2_on
        self.phase2_off = phase2_off
        self.freq_on = freq_on
        self.freq_off = freq_off
        self.amp_on = amp_on
        self.amp_off = amp_off
        self.x_on = amp_on * np.cos(phase1_on/180*np.pi)
        self.x_off = amp_off * np.cos(phase1_off / 180 * np.pi)

        self.amp_dr_on = amp_dr_on
        self.amp_dr_off = amp_dr_off
        self.x_dr_on = amp_dr_on * np.cos(phase_dr_on / 180 * np.pi)
        self.x_dr_off = amp_dr_off * np.cos(phase_dr_off / 180 * np.pi)
        self.phase_dr_on = phase_dr_on
        self.phase_dr_off = phase_dr_off
        self.q_on = q_on
        self.q_off = q_off

        # return bias[1:], amp_off[1:], phase1_off[1:], phase2_off[1:]

    def _correct_phase_wrapping(self, ph, lower=-90, upper=270, offset_correction=True):
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
        if offset_correction:
            ph_shift = ph - ph[-1]
        else:
            ph_shift = ph

        index_upper = np.where(ph_shift > upper)
        index_lower = np.where(ph_shift < lower)
        ph_shift[index_upper] -= 360
        ph_shift[index_lower] += 360

        return ph_shift

    @staticmethod
    def _calc_drive_params(_a1, _a2, _ph1, _ph2, _fc, _df):
        '''
        Calculate real Dart parameters from the observables.

        Input:
            _a1  - amplitude 1
            _a2  - amplitude 2
            _ph1 - phase 1
            _ph2 - phase 2
            _fc  - resonance frequency
            _df  - difference between freq 2 and freq 1
        Output:
            _a_drive  - drive amplitude
            _ph_drive - resonance phase
            _q        - resonanse quality factor
        '''

        epsilon = 1e-10  # a small adding for calculation stability
        _dph = _ph2 - _ph1
        _f1 = _fc - _df / 2
        _f2 = _fc + _df / 2

        _om = _f1 * _a1 / (_f2 * _a2)
        _fi = np.tan(_dph)

        _x1 = -(1 - np.sign(_fi) * np.sqrt(1 + np.square(_fi)) / _om) / (_fi + epsilon)
        _x2 = (1 - np.sign(_fi) * np.sqrt(1 + np.square(_fi)) * _om) / (_fi + epsilon)

        _q = np.sqrt(_f1 * _f2 * (_f2 * _x1 - _f1 * _x2) * (_f1 * _x1 - _f2 * _x2)) / (np.square(_f2) - np.square(_f1))
        _q[_q > 1000] = 1000
        _a_drive = _a1 * np.sqrt((_fc**2 - _f1**2)**2 +(_fc * _f1 / _q)**2) / np.square(_fc)
        _ph_drive = _ph1 - np.arctan(_fc * _f1 / (_q * (np.square(_fc) - np.square(_f1))))

        return _a_drive, _ph_drive, _q


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

def ifft(data, output='real', envelope=False):
    '''
    Compute the inverse Fourier transform with the option to detect envelope.

    Inputs:
        data    - Required : A 1D or 2D numpy array. (3D not yet supported)
        output  - Optional : String containing desired form of output.  The
                             options are: 'absolute', 'real', 'imag', 'phase'
                             or 'complex'.
        envelope - Optional : Boolen, when True applies the Hilbert transform
                              to detect the envelope of the IFT, which is the
                              absolute values.
        **kwarg - Optional : Passed to scipy.signal.hilbert()

    See docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    for more information about envelope function.

    Outputs:
        ift - A numpy array containing inverse Fourier transform.

    History:
        Adapted from stmpy
    '''
    outputFunctions = {'absolute':np.absolute, 'real':np.real,
                       'imag':np.imag, 'phase':np.angle, 'complex':(lambda x:x)}
    out = outputFunctions[output]
    if len(data.shape) == 2:
        ift = np.fft.ifft2(np.fft.ifftshift(data))
    elif len(data.shape) == 1:
        ift = np.fft.ifft(np.fft.ifftshift(data))
    if envelope:
        ift = hilbert(np.real(ift))
    return out(ift)


def fftfreq(px, nm):
    '''Get frequnecy bins for Fourier transform.'''
    freqs = np.fft.fftfreq(px, float(nm)/(px))
    return np.fft.fftshift(freqs)
    
def interp2d(x, y, z, kind='nearest', **kwargs):
    '''
    An extension for scipy.interpolate.interp2d() which adds a 'nearest'
    neighbor interpolation.

    See help(scipy.interpolate.interp2d) for details.

    Inputs:
        x       - Required : Array contining x values for data points.
        y       - Required : Array contining y values for data points.
        z       - Required : Array contining z values for data points.
        kind    - Optional : Sting for interpolation scheme. Options are:
                             'nearest', 'linear', 'cubic', 'quintic'.  Note
                             that 'linear', 'cubic', 'quintic' use spline.
        **kwargs - Optional : Keyword arguments passed to
                              scipy.interpolate.interp2d

    Returns:
        f(x,y) - Callable function which will return interpolated values.

    History:
        Adapted from stmpy.
    '''
    from scipy.interpolate import NearestNDInterpolator
    if kind is 'nearest':
        X, Y = np.meshgrid(x ,y)
        points = np.array([X.flatten(), Y.flatten()]).T
        values = z.flatten()
        fActual = NearestNDInterpolator(points, values)
        def fCall(x, y):
            if type(x) is not np.ndarray:
                lx = 1
            else:
                lx = x.shape[0]
            if type(y) is not np.ndarray:
                ly = 1
            else:
                ly = y.shape[0]
            X, Y = np.meshgrid(x ,y)
            points = np.array([X.flatten(), Y.flatten()]).T
            values = fActual(points)
            return values.reshape(lx, ly)
        return fCall
    else:
        from scipy.interpolate import interp2d as scinterp2d
        return scinterp2d(x, y, z, kind=kind, **kwargs)
    
def linecut(data, p0, p1, width=1, dl=1, dw=1, kind='linear',
                show=False, ax=None, **kwarg):
    '''Linecut tool for 2D or 3D data.

    Inputs:
        data    - Required : A 2D or 3D numpy array.
        p0      - Required : A tuple containing indicies for the start of the
                             linecut: p0=(x0,y0)
        p1      - Required : A tuple containing indicies for the end of the
                             linecut: p1=(x1,y1)
        width   - Optional : Float for perpendicular width to average over.
        dl      - Optional : Extra pixels for interpolation in the linecut
                             direction.
        dw      - Optional : Extra pixels for interpolation in the
                             perpendicular direction.
        kind    - Optional : Sting for interpolation scheme. Options are:
                             'nearest', 'linear', 'cubic', 'quintic'.  Note
                             that 'linear', 'cubic', 'quintic' use spline.
        show    - Optional : Boolean determining whether to plot where the
                             linecut was taken.
        ax      - Optional : Matplotlib axes instance to plot where linecut is
                             taken.  Note, if show=True you MUST provide and
                             axes instance as plotting is done using ax.plot().
        **kwarg - Optional : Additional keyword arguments passed to ax.plot().

    Returns:
        r   -   1D numpy array which goes from 0 to the length of the cut.
        cut -   1D or 2D numpy array containg the linecut.

    Usage:
        r, cut = linecut(data, (x0,y0), (x1,y1), width=1, dl=0, dw=0,
                         show=False, ax=None, **kwarg)

    History:
        Adapted from stmpy.
    '''
    def calc_length(p0, p1, dl):
        dx = float(p1[0]-p0[0])
        dy = float(p1[1]-p0[1])
        l = np.sqrt(dy**2 + dx**2)
        if dx == 0:
            theta = np.pi/2
        else:
            theta = np.arctan(dy / dx)
        xtot = np.linspace(p0[0], p1[0], int(np.ceil(l+dl)))
        ytot = np.linspace(p0[1], p1[1], int(np.ceil(l+dl)))
        return l, theta, xtot, ytot

    def get_perp_line(x, y, theta, w):
        wx0 = x - w/2.0*np.cos(np.pi/2 - theta)
        wx1 = x + w/2.0*np.cos(np.pi/2 - theta)
        wy0 = y + w/2.0*np.sin(np.pi/2 - theta)
        wy1 = y - w/2.0*np.sin(np.pi/2 - theta)
        return (wx0, wx1), (wy0, wy1)

    def cutter(F, p0, p1, dw):
        l, __, xtot, ytot = calc_length(p0, p1, dw)
        cut = np.zeros(int(np.ceil(l+dw)))
        for ix, (x,y) in enumerate(zip(xtot, ytot)):
            cut[ix] = F(x,y)
        return cut

    def linecut2D(layer, p0, p1, width, dl, dw):
        xAll, yAll = np.arange(layer.shape[1]), np.arange(layer.shape[0])
        F = interp2d(xAll, yAll, layer, kind=kind)
        l, theta, xtot, ytot = calc_length(p0, p1, dl)
        r = np.linspace(0, l, int(np.ceil(l+dl)))
        cut = np.zeros(int(np.ceil(l+dl)))
        for ix, (x,y) in enumerate(zip(xtot,ytot)):
            (wx0, wx1), (wy0, wy1) = get_perp_line(x, y, theta, width)
            wcut = cutter(F, (wx0,wy0), (wx1,wy1), dw)
            cut[ix] = np.mean(wcut)
        return r, cut

    if len(data.shape) == 2:
        r, cut = linecut2D(data, p0, p1, width, dl, dw)
    if len(data.shape) == 3:
        l, __, __, __ = calc_length(p0, p1, dl)
        cut = np.zeros([data.shape[0], int(np.ceil(l+dl))])
        for ix, layer in enumerate(data):
            r, cut[ix] = linecut2D(layer, p0, p1, width, dl, dw)
    if show:
        __, theta, __, __ = calc_length(p0, p1, dl)
        (wx00, wx01), (wy00, wy01) = get_perp_line(p0[0], p0[1], theta, width)
        (wx10, wx11), (wy10, wy11) = get_perp_line(p1[0], p1[1], theta, width)
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]], 'k--', **kwarg)
        ax.plot([wx00,wx01], [wy00,wy01], 'k:', **kwarg)
        ax.plot([wx10,wx11], [wy10,wy11], 'k:', **kwarg)
    return r, cut

# Help function to crop images into patches
from typing import Tuple, Optional, Dict, Union, List

def get_coord_grid(imgdata: np.ndarray, step: int,
                   return_dict: bool = True
                   ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Generate a square coordinate grid for every image in a stack. Returns coordinates
    in a dictionary format (same format as generated by atomnet.predictor)
    that can be used as an input for utility functions extracting subimages
    and atomstat.imlocal class (Adapted from atomai).

    Args:
        imgdata (numpy array): 2D or 3D numpy array
        step (int): distance between grid points
        return_dict (bool): returns coordiantes as a dictionary (same format as atomnet.predictor)

    Returns:
        Dictionary or numpy array with coordinates
    """
    if np.ndim(imgdata) == 2:
        imgdata = np.expand_dims(imgdata, axis=0)
    coord = []
    for i in range(0, imgdata.shape[1], step):
        for j in range(0, imgdata.shape[2], step):
            coord.append(np.array([i, j]))
    coord = np.array(coord)
    if return_dict:
        coord = np.concatenate((coord, np.zeros((coord.shape[0], 1))), axis=-1)
        coordinates_dict = {i: coord for i in range(imgdata.shape[0])}
        return coordinates_dict
    coordinates = [coord for _ in range(imgdata.shape[0])]
    return np.concatenate(coordinates, axis=0)
    
    
def get_imgstack(imgdata: np.ndarray,
                 coord: np.ndarray,
                 r: int) -> Tuple[np.ndarray]:
    """
    Extracts subimages centered at specified coordinates
    for a single image (Adapted from atomai).
    Args:
        imgdata (3D numpy array):
            Prediction of a neural network with dimensions
            :math:`height \\times width \\times n channels`
        coord (N x 2 numpy array):
            (x, y) coordinates
        r (int):
            Window size
    Returns:
        2-element tuple containing
        - Stack of subimages
        - (x, y) coordinates of their centers
    """
    img_cr_all = []
    com = []
    for c in coord:
        cx = int(np.around(c[0]))
        cy = int(np.around(c[1]))
        if r % 2 != 0:
            img_cr = np.copy(
                imgdata[cx-r//2:cx+r//2+1,
                        cy-r//2:cy+r//2+1])
        else:
            img_cr = np.copy(
                imgdata[cx-r//2:cx+r//2,
                        cy-r//2:cy+r//2])
        if img_cr.shape[0:2] == (int(r), int(r)) and not np.isnan(img_cr).any():
            img_cr_all.append(img_cr[None, ...])
            com.append(c[None, ...])
    if len(img_cr_all) == 0:
        return None, None
    img_cr_all = np.concatenate(img_cr_all, axis=0)
    com = np.concatenate(com, axis=0)
    return img_cr_all, com


def extract_subimages(imgdata: np.ndarray,
                      coordinates: Union[Dict[int, np.ndarray], np.ndarray],
                      window_size: int, coord_class: int = 0):

    """
    Extracts subimages centered at certain atom class/type
    (usually from a neural network output) (Adapted from atomai).

    Args:
        imgdata (numpy array):
            4D stack of images (n, height, width, channel).
            It is also possible to pass a single 2D image.
        coordinates (dict or N x 2 numpy arry): Prediction from atomnet.locator
            (can be from other source but must be in the same format)
            Each element is a :math:`N \\times 3` numpy array,
            where *N* is a number of detected atoms/defects,
            the first 2 columns are *xy* coordinates
            and the third columns is class (starts with 0).
            It is also possible to pass N x 2 numpy array if the corresponding
            imgdata is a single 2D image.
        window_size (int):
            Side of the square for subimage cropping
        coord_class (int):
            Class of atoms/defects around around which the subimages
            will be cropped (3rd column in the atomnet.locator output)

    Returns:
        3-element tuple containing

        - stack of subimages,
        - (x, y) coordinates of their centers,
        - frame number associated with each subimage
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = np.concatenate((
            coordinates, np.zeros((coordinates.shape[0], 1))), axis=-1)
        coordinates = {0: coordinates}
    if np.ndim(imgdata) == 2:
        imgdata = imgdata[None, ..., None]
    subimages_all, com_all, frames_all = [], [], []
    for i, (img, coord) in enumerate(
            zip(imgdata, coordinates.values())):
        coord_i = coord[np.where(coord[:, 2] == coord_class)][:, :2]
        stack_i, com_i = get_imgstack(img, coord_i, window_size)
        if stack_i is None:
            continue
        subimages_all.append(stack_i)
        com_all.append(com_i)
        frames_all.append(np.ones(len(com_i), int) * i)
    if len(subimages_all) > 0:
        subimages_all = np.concatenate(subimages_all, axis=0)
        com_all = np.concatenate(com_all, axis=0)
        frames_all = np.concatenate(frames_all, axis=0)

    return subimages_all, com_all, frames_all
    
    
def extract_patches_(lattice_im: np.ndarray, lattice_mask: np.ndarray,
                     patch_size: int, num_patches: int, **kwargs: int
                     ) -> Tuple[np.ndarray]:
    """
    Extracts subimages of the selected size from the 'mother" image and mask (Adapted from atomai).
    """
    rs = kwargs.get("random_state", 0)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    images = extract_patches_2d(
        lattice_im, patch_size, max_patches=num_patches, random_state=rs)
    labels = extract_patches_2d(
        lattice_mask, patch_size, max_patches=num_patches, random_state=rs)
    return images, labels


def extract_patches(images: np.ndarray, masks: np.ndarray,
                    patch_size: int, num_patches: int, **kwargs: int
                    ) -> Tuple[np.ndarray]:
    """
    Takes batch of images and batch of corresponding masks as an input
    and for each image-mask pair it extracts stack of subimages (patches)
    of the selected size (Adapted from atomai).
    """
    if np.ndim(images) == 2:
        images = images[None, ...]
    if np.ndim(masks) == 2:
        masks = masks[None, ...]
    images_aug, masks_aug = [], []
    for im, ma in zip(images, masks):
        im_aug, ma_aug = extract_patches_(
            im, ma, patch_size, num_patches, **kwargs)
        images_aug.append(im_aug)
        masks_aug.append(ma_aug)
    images_aug = np.concatenate(images_aug, axis=0)
    masks_aug = np.concatenate(masks_aug, axis=0)
    return images_aug, masks_aug


def extract_patches_and_spectra(hdata: np.ndarray, *args: np.ndarray,
                                coordinates: np.ndarray = None,
                                window_size: int = None,
                                avg_pool: int = 2,
                                **kwargs: Union[int, List[int]]
                                ) -> Tuple[np.ndarray]:
    """
    Extracts image patches and associated spectra
    (corresponding to patch centers) from hyperspectral dataset (Adapted from atomai).

    Args:
        hdata:
            3D or 4D hyperspectral data
        *args:
            2D image for patch extraction. If not provided, then
            patches will be extracted from hyperspectral data
            averaged over a specified band (range of "slices")
        coordinates:
            2D numpy array with xy coordinates
        window_size:
            Image patch size
        avg_pool:
            Kernel size and stride for average pooling in spectral dimension(s)
        **band:
            Range of slices in hyperspectral data to average over
            for producing a 2D image if the latter is not provided as a separate
            argument. For 3D data, it can be integer (use a single slice)
            or a 2-element list. For 4D data, it can be integer or a 4-element list.

        Returns:
            3-element tuple with image patches, associated spectra and coordinates
    """
    F = torch.nn.functional
    if hdata.ndim not in (3, 4):
        raise ValueError("Hyperspectral data must 3D or 4D")
    if len(args) > 0:
        img = args[0]
        if img.ndim != 2:
            raise ValueError("Image data must be 2D")
    else:
        band = kwargs.get("band", 0)
        if hdata.ndim == 3:
            if isinstance(band, int):
                band = [band, band+1]
            img = hdata[..., band[0]:band[1]].mean(-1)
        else:
            if isinstance(band, int):
                band = [band, band+1, band, band+1]
            elif isinstance(band, list) and len(band) == 2:
                band = [*band, *band]
            img = hdata[..., band[0]:band[1], band[2]:band[3]].mean((-2, -1))
    patches, coords, _ = extract_subimages(img, coordinates, window_size)
    patches = patches.squeeze()
    spectra = []
    for c in coords:
        spectra.append(hdata[int(c[0]), int(c[1])])
    avg_pool = 2*[avg_pool] if (isinstance(avg_pool, int) & hdata.ndim == 4) else avg_pool
    torch_pool = F.avg_pool1d if hdata.ndim == 3 else F.avg_pool2d
    spectra = torch.tensor(spectra).unsqueeze(1)
    spectra = torch_pool(spectra, avg_pool, avg_pool).squeeze().numpy()
    return patches, spectra, coords