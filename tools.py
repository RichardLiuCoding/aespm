import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import os
from igor2 import binarywave as bw


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

# User defined functions

# @title
class IBWData:
    def __init__(self, header=None, channels=None, data=None, size=None):
        self.header = header if header is not None else {}
        self.channels = channels if channels is not None else []
        self.data = data if data is not None else []

    def __repr__(self):
        return f"IBWData(header={self.header}, channels={self.channels}, data={self.data})"

def load_ibw(file, ss=False):
    '''
    Load the ibw file into an IBWData object.
    It automatically creates three default attributes:
      1. header: a dict contains all the setup information
      2. channels: a list of channel names for each image data
      3. data: an array of all the saved image data in this ibw file
    '''
    t = bw.load(file)
    wave = t.get('wave')

    # Decode the notes section to parse the header
    if isinstance(wave['note'], bytes):
        try:
            parsed_string = wave['note'].decode('utf-8').split('\r')
        except:
            parsed_string = wave['note'].decode('ISO-8859-1').split('\r')
    header = {}
    for item in parsed_string:
        try:
            key, value = item.split(':', 1)
            value = value.strip()  # Remove leading/trailing whitespace
        except ValueError:
            continue  # For items that do not split correctly

        # Determine the data type of the value and convert
        if '.' in value or 'e' in value:  # Floating point check
            try:
                header[key] = float(value)
            except ValueError:
                header[key] = value
        elif value.lstrip('-').isdigit():  # Integer check
            header[key] = int(value)
        else:
            header[key] = value

    # Transpose the data matrix
    data = wave['wData'].T
    # data = wave['wData']

    # Extract channel data types from the header
    channels = [header.get(f'Channel{i+1}DataType', 'Unknown') for i in range(np.shape(data)[0])]
    out = IBWData(header, channels, data)
    out.size = header['ScanSize']
    out.mode = header['ImagingMode']
    if out.mode == "PFM Mode":
        if len(out.channels) > 4:
            out.mode = "DART Mode"
            out.channels = ['Height', 'Amplitude1', 'Amplitude2', 'Phase1', 'Phase2', 'Frequency']
        else:
            out.channels = ['Height', 'Amplitude', 'Deflection', 'Phase']

    # Load the switching spectroscopy (hysteresis loop) data
    if ss is not False:
        bias_raw= data[-1]
        index_values = np.where(~np.isnan(bias_raw))

        bias = bias_raw[index_values]
        amp, phase1, phase2 = data[2][index_values], data[3][index_values], data[4][index_values]
        index_bp = np.where(np.diff(bias) != 0)[0] + 1
        index_bp = np.concatenate([[0], index_bp])

        phase1 = correct_phase_wrapping(phase1)
        phase2 = correct_phase_wrapping(phase2)

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
        out.bias = bias_on[1:]
        out.phase1_on = phase1_on[1:]
        out.phase1_off = phase1_off[1:]
        out.phase2_on = phase2_on[1:]
        out.phase2_off = phase2_off[1:]
        out.amp_on = amp_on[1:] * np.cos(phase1_off[1:]/180*np.pi)
        out.amp_off = amp_off[1:] * np.cos(phase1_off[1:]/180*np.pi)
        out.data = data
    # Return an IBWData object
    return out

# load functions
def ss_loop(A, nan=False):

    if nan is True:
        bias_raw= A[-1]
        index_values = np.where(~np.isnan(bias_raw))

        bias = bias_raw[index_values]
        amp, phase1, phase2 = A[2][index_values], A[3][index_values], A[4][index_values]
    else:
        bias = A[-1]
        amp = A[2]
        phase1 = A[3]
        phase2 = A[4]
    phase1 = correct_phase_wrapping(phase1)
    phase2 = correct_phase_wrapping(phase2)

    index_bp = np.where(np.diff(bias) != 0)[0] + 1
    index_bp = np.concatenate([[0], index_bp])

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
    bias = bias_on
    phase1_on = phase1_on
    phase1_off = phase1_off
    phase2_on = phase2_on
    phase2_off = phase2_off
    amp_on = amp_on * np.cos(phase2_off/180*np.pi)
    amp_off = amp_off * np.cos(phase1_off/180*np.pi)
    return bias[1:], amp_off[1:], phase1_off[1:], phase2_off[1:]


def correct_phase_wrapping(ph):
    ph_shift = ph - ph[-1]
    index1 = np.where(ph_shift > 270)
    index2  =np.where(ph_shift < -90)
    ph_shift[index1] -= 360
    ph_shift[index2] += 360
    return ph_shift