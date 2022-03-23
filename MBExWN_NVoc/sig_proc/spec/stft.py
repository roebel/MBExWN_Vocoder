from __future__ import division, absolute_import
import numpy as np

from ..Mwindows import Mhanning, window

local_fft   = np.fft.fft
local_ifft  = np.fft.ifft
local_rfft  = np.fft.rfft
local_irfft = np.fft.irfft

def get_stft_window(win_type, win_len, dtype):
    return window(win_type=win_type, winlen=win_len).astype(dtype=dtype)

def calc_stft(x, win_len, hop_len, fft_size, center=True, pad_mode='reflect',
              win_type="hann", axis=-1, do_mag=False, dtype=None, detrend_phase  = False, norm_window=False):
    """

    returns D the STFT of x calculated along axis axis
    All analysis frames will have their center within the signal.

    :param x: signal
    :type x: np.ndarray
    :param win_len: length of analysis window in samples (<= fft_len)
    :type win_len: int
    :param hop_len: step size in samples
    :type hop_len: int
    :param fft_size: fft_size in samples
    :type fft_size:
    :param center: If `True`, the signal `x` is padded so that frame
    :param pad_mode: padding mode for the case center = True (see np.pad for modes)
    :type pad_mode: str
    :param win_type: window type available from `sig_proc.windows`
    :type win_type: str
    :param axis: the axis in  along that the STFT should be calculated
    :type axis: int
    :param do_mag: if True the return value is the magnitude STFT
    :type do_mag: bool
    :param dtype: the datatype used for calculating the stft, if set to None the datatype is derived from the input signal
    :type dtype: np.dtype
    :return:
    :rtype:
    """
    axis = np.arange(x.ndim)[axis]
    win = get_stft_window(win_type=win_type, win_len=win_len, dtype=dtype)[tuple(slice(None) if ax == axis else np.newaxis
                                                                                 for ax in range(x.ndim))]

    if norm_window:
        win /= np.sum(win)

    if dtype is None:
        dtype = x.dtype


    if center:
        # Calculate number of frames
        num_frames = (x.shape[axis] // hop_len) + 1
        x = np.pad(x.astype(dtype, copy=False),
                       tuple((0,0) if ax != axis else (win_len//2, win_len)
                             for ax in range(x.ndim)),
                       mode=pad_mode)
    else:
        if x.shape[axis]< win_len:
            raise RuntimeError('calc_stft::error::cannot calculate STFT if signal is shorter than window')
        num_frames = ((x.shape[axis]-win_len) // hop_len) + 1
        x = x.astype(dtype, copy=False)


    out_shape = list(x.shape)
    out_shape[axis] = fft_size//2 +1
    out_shape.insert(axis, num_frames)
    if do_mag:
        res = np.empty(out_shape, dtype=dtype)
    else:
        res = np.empty(out_shape, dtype=_get_cplx_dtype(dtype))

    store_ind = list(None if ax == axis else slice(None)  for ax in range(len(out_shape)))
    x_ind = list(None if ax == axis else slice(None)  for ax in range(x.ndim))

    if not do_mag and detrend_phase:
        detrend_vec = np.exp(1j * np.pi * (win_len - 1)
                                  * np.arange(fft_size // 2 + 1) /fft_size).astype( _get_cplx_dtype(dtype))
        def fft_fun(_vec, _size):
            return local_rfft(_vec, _size) * detrend_vec
    else:
        fft_fun = local_rfft

    for ii in range(num_frames):
        store_ind[axis] = ii
        start = ii * hop_len
        x_ind[axis] = slice(start, start + win_len )
        if do_mag:
            res[tuple(store_ind)] = np.abs(fft_fun(win * x[tuple(x_ind)], fft_size))
        else:
            res[tuple(store_ind)] = fft_fun(win * x[tuple(x_ind)], fft_size)

    return res
