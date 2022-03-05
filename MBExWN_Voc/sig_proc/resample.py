# AUTHOR:  A.Roebel
# COPYRIGHT: Copyright(c) 2019 - 2022 IRCAM - Roebel

import numpy as np
import scipy.signal as ss

def resample(x, in_sr, out_sr, stop_att=70, axis=0, trans_width_normed=0.1, fir_filt=None) :
    """
     resampling with FIR based anti-aliasing (AA) filter

    :param x: signal
    :type x: np.array
    :param in_sr: sample rate x
    :type in_sr: Union[float, int]
    :param out_sr: desired sample rate after resampling
    :type out_sr: Union[float, int]
    :param stop_att: minimum stop band attenuation at nyquist (out_sr) in dB
    :type stop_att: Union[float, int]
    :param axis: axis along which the resampling will be performed
    :type axis: int
    :param trans_width_normed: transition width of the 6dB pass band to stop band frequency of the low pass AA filter
       this parameter will be used only if the fir_filt parameter is not given or is None
    :type trans_width_normed:  float
    :param fir_filt: FIR filter coefficients to be used for th antialiasing filter. If this parameter is gien the
       trans_width_normed parameter will be ignored
    :type fir_filt: Union[np.array, None]
    :return: 2-tupe containg the downsampled signal as well as the FIR AA filter that has been used
    :rtype: tuple(np.array, np.array)
    """

    in_sr =int(in_sr)
    out_sr = int(out_sr)
    gcd = np.math.gcd(in_sr, out_sr)

    up = out_sr/gcd
    down = in_sr/gcd

    if fir_filt is None:
        # Calculate the window parameters for kaiser window */
        if (stop_att >= 50) :
            mBeta = 0.1102*(stop_att-8.7)
        elif(stop_att >= 21) :
            mBeta = 0.5842*pow(stop_att-21.,0.4)+0.07886*(stop_att-21.)
        else :
            mBeta = 0.

        mTransWidth = 2*np.pi*np.fmin(1.,out_sr/in_sr)*trans_width_normed

        # radius for filter in orginal samplerate
        while True:
            mRadius = np.int(np.ceil((stop_att -8.)/2.285/mTransWidth/2))
            #print("resampler::mRadius {} cond {} stop_att {} beta {}\n".format(mRadius,2*mRadius>8000,stop_att,mBeta))
            if  ((2*mRadius>8000) and stop_att > 10) :
                stop_att -= 6
            else :
                break

        winlen = mRadius * 2 + 1
        if x.dtype == np.float32:
            filt_dtype = x.dtype
        else:
            filt_dtype = np.float64
        fir_filt= ss.firwin(winlen * up, cutoff=(1-trans_width_normed)/max(up, down), window=("kaiser", mBeta)).astype(filt_dtype, copy=False)
    return ss.resample_poly(x, up, down, axis=axis, window=fir_filt), fir_filt
    
    
if __name__ == "__main__":
    import pysndfile.sndio as sndio

    sig, rr, enc, fmt=sndio.read('~/snd/manna.aiff', return_format=True)
    ss96, ww96= resample(sig, rr, 96000)
    ss16, ww16= resample(sig, rr, 16000)
    ss8, ww8  = resample(sig, rr, 8000)
    sndio.write("mm8.aiff", ss8 / np.max(np.abs(ss8)), rate=8000)
    sndio.write("mm16.aiff", ss16 / np.max(np.abs(ss16)), rate=16000)
    sndio.write("mm96.aiff", ss96 / np.max(np.abs(ss96)), rate=96000)
    
