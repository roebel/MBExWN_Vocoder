# coding: utf-8
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  mel spectrogram generation and transformation

import os, sys

import numpy as np
from typing import List, Union, Any, Sequence, Dict, Tuple

import scipy.interpolate
from sig_proc.spec.stft import calc_stft, STFT, get_stft_window
import librosa

try:
    from librosa.core.convert import mel_frequencies as librosa_mel_frequencies
    from librosa.core.convert import hz_to_mel as librosa_hz_to_mel
    from librosa.core.convert import mel_to_hz as librosa_mel_to_hz
except ModuleNotFoundError:
    from librosa.core.time_frequency import mel_frequencies as librosa_mel_frequencies
    from librosa.core.time_frequency import hz_to_mel as librosa_hz_to_mel
    from librosa.core.time_frequency import mel_to_hz as librosa_mel_to_hz


from ...utils import nextpow2_val
from utils_find_1st import find_1st, cmp_larger
from sig_proc.resample import resample
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import convolve
from utils_bspline import bspline_c as bsp

try:
    librosa.feature.melspectrogram(y=np.zeros(2100),norm="slaney")
    librosa_use_norm_slaney = True
except librosa.ParameterError:
    librosa_use_norm_slaney = False

# avoid warning in fo mel bands bove sample rate
import warnings
warnings.filterwarnings("ignore", "(?s).*Empty filters detected in mel frequency basis.*", category=UserWarning)

from functools import lru_cache
@lru_cache(30, typed=True)
def get_mel_filter(sr, n_fft, n_mels, fmin, fmax, dtype=np.dtype('float32'), centered=False, norm =True):

    norm_arg = None
    if norm:
        norm_arg = 'slaney' if librosa_use_norm_slaney else 1
    # Build a Mel filter, librosa caching is disk based, we dont use that cache mechanism
    if centered :
        # centered == True means the bordering mel bands have their centers placed over fmin and fmax,
        # while for centering == False (the default fmin, fmax arre the border frequencies of the bordering mel bands
        mel_freqs = librosa_mel_frequencies(n_mels = n_mels, fmin = fmin, fmax= fmax, htk =False)
        lower_half_band = mel_freqs[1] - mel_freqs[0]
        upper_half_band = mel_freqs[-1]- mel_freqs[-2]
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                        fmin=fmin-lower_half_band,fmax=fmax + upper_half_band, htk=False,
                                        norm=norm_arg,
                                        dtype=dtype)
    else:
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                        fmin=fmin,fmax=fmax, htk=False,
                                        norm=norm_arg,
                                        dtype=dtype)

    return mel_basis

def get_filters(sr: Union[int, float], n_fft:int, n_bands: int,
                fmin : Union[int, float], fmax: Union[int, float],
                dtype=np.dtype('float32'), centered : bool =False):
    """
    Generate equally spaced triangular filter bank

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_bands    : int > 0 [scalar]
        number of  bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``

    centered: bool
        centered == True means the bordering mel bands have their centers placed over fmin and fmax,
        while for centering == False (the default) fmin, fmax are the border frequencies of the bordering bands

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.


    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
    """
    if fmax is None:
        fmax = sr/2

    if centered :
        frequency_positions = np.concatenate(([fmin], np.linspace(fmin, fmax, n_bands), [fmax]), axis=0)
    else:
        frequency_positions = np.linspace(fmin, fmax, n_bands + 2)

    bins = np.linspace(0, n_fft//2, n_fft//2+1)
    # move all filter centers to bin centers
    quantized_bin_positions = np.array([np.round(n_fft*ff/sr) for ff in frequency_positions])
    # force last support point to ly outside the axis because otherwise the bspline package generates a zero entry
    quantized_bin_positions[-1] *= 1.00000001
    filter_basis = []
    for ii in range(n_bands):
        filter_basis.append(bsp.generate_basis(quantized_bin_positions[ii:ii+3], bins).astype(dtype, copy=False)[0])

    return np.array(filter_basis)



log_to_db = 20 * np.log10(np.exp(1))


def scale_mel_spectrogram(mel_spectrogram, preprocess_config, forward = True, use_tf =False):
    #print(f"max mel spectrogram {np.max(mel_spectrogram)} mean mel spec {np.mean(mel_spectrogram)}")
    if use_tf:
        import tensorflow as tf
    lin_amp_scale = 1
    if ("lin_amp_scale" in preprocess_config) and (preprocess_config["lin_amp_scale"] != 1):
        lin_amp_scale = preprocess_config["lin_amp_scale"]

    lin_amp_off = 1.e-5
    if "lin_amp_off" in preprocess_config and (preprocess_config["lin_amp_off"] is not None):
        lin_amp_off = preprocess_config["lin_amp_off"]

    mel_amp_scale = 1
    if ("mel_amp_scale" in preprocess_config) and (preprocess_config["mel_amp_scale"] != 1):
        mel_amp_scale = preprocess_config["mel_amp_scale"]

    if forward :
        if use_tf:
            # if use_tf we suppose the input is an array or a tensor already.
            mel = mel_spectrogram * lin_amp_scale
        else:
            mel = np.array(mel_spectrogram) * lin_amp_scale

        if "use_max_limit" in preprocess_config and preprocess_config["use_max_limit"]:
            if use_tf :
                mel = mel_amp_scale * tf.math.log(tf.maximum(mel, lin_amp_off))
            else:
                mel = mel_amp_scale * np.log(np.fmax(mel, lin_amp_off)).astype(np.float32)
        else:
            if use_tf :
                mel = mel_amp_scale * tf.math.log(mel + lin_amp_off)
            else:
                mel = mel_amp_scale * np.log(mel + lin_amp_off).astype(np.float32)
    else:
        if use_tf :
            mel = tf.exp(mel_spectrogram / mel_amp_scale)
        else:
            mel = np.exp(mel_spectrogram / mel_amp_scale).astype(np.float32)
        if "use_max_limit" in preprocess_config and preprocess_config["use_max_limit"]:
            pass
        else:
             mel -= lin_amp_off

        mel /= lin_amp_scale

    return mel


def norm_mell(mell, preprocess_config, num_frames_per_seg=None, snd=None, de_normalize_snd=False, center=True,
              mell_is_log=True, norm_max=None, norm_max_fac= None, num_smooth_iters=None, smooth_win_scale=1., use_tf=False,
              return_snd_gain=False, normalize_compressor_exp=None, use_pinv=False, old_gain_smooth=False):
    """
    shift mell and scale corresponding sound such that median of max(mell) or rms(mell) over num_frames_per_seg is 0

    Parameters:
    :param mell: mell spectrogram to normalize
    :param hop_size: analysis hop_size
    :param num_frames_per_seg: number of mell frames that are used to calculate the median of the maximum
           this parameter allows weakening the normalization effect
    :param snd: snd signal corresponding to the mell sequence, if given and not None this sound will scaled
                to correspond to the normalized (shifted) mel spectrogram. If de_normalize_snd is True
                the sound is supposed have been normalized already and is denormalized to correspond to the
                input mell spectrum.
    :param de_normalize_snd: whether the input snd will normalized (de_normalize_snd = False) or
                de_normalized (de_normalize_snd = True)

    returns  mel_out
       mel_out: the shifted (normalized) mell spectrum

    in case snd is not None

    returns  mel_out, snd_out
       mel_out: the shifted (normalized) mell spectrum
       snd_out: if de_normalize_snd is False snd_out is the scaled snd that corresponds approximately to
                the shifted mel spectrum. If de_normalize_snd is True snd is supposed to contain
                a normalized sound signal and it is scaled to fit the mell spectrum

    """
    if not center:
        raise NotImplementedError("scale_median_mell::error:: scaling for non centered STFT is not yet implemented")

    if num_frames_per_seg is None:
        if "norm_mel" in preprocess_config:
            num_frames_per_seg = preprocess_config["norm_mel"]["num_frames_per_seg"]

    if num_frames_per_seg is not None:
        if num_frames_per_seg<= 0:
            if snd is not None:
                return mell, snd
            return mell
        if num_frames_per_seg != 1:
            raise NotImplementedError("norm_mel::norm_mel:num_frames_per_seg != 1 is no longer supported")

    hop_size = preprocess_config["hop_size"]
    win_size = preprocess_config["win_size"]
    if 4 * hop_size != win_size:
        raise RuntimeError("norm_mel:error: this function currently supports only the case where win_size {win_size} = 4 * hop_size {hop_size}")

    fft_size = preprocess_config["fft_size"]
    if norm_max is  None:
        if "norm_mel" in preprocess_config:
            norm_max = preprocess_config["norm_mel"]["norm_max_fac"]
        else:
            norm_max = False

    if num_smooth_iters is None:
        if "norm_mel" in preprocess_config:
            num_smooth_iters = preprocess_config["norm_mel"]["num_smooth_iters"]
        else:
            num_smooth_iters = 1

    if mell_is_log:
        mell_test = np.exp(mell)
    else:
        mell_test = mell
    rescale_mel = not norm_max
    if rescale_mel:

        # Here we approximately calculate the energy of the signal under two different assumptions:
        # For use_pinv == False
        # we assume that the signal spectrum is sparse and concentrated at the center of the melbands
        # The fact that all the energy is assumed to be concentrated in a single point will lead to an over-estimation
        # of the energy depending on the band width. The wider the band the more the energy will be over-estimated
        # Similarly the impact of noise energy that tends to be more spread than sinusoidal energy will also be
        # over-estimated. As a result the energy estimate will be to high which means that the normalization will
        # produce a signal with an energy smaller than 1. Notably th highr mel bands with larger bandwidth
        # will have a smaller effect on the estimated signal energy and by consequence on the normalization.
        # It turns out however that the signals maximum amplitude over time is generally more constant
        # compared to the more consistent energy estimation assuming a spread signal spectrum.
        # For use_pinv == True
        # we assume that the signal spectrum is rather smoothly distributed over all
        # melbands. For the ambiguous inversion the pseudo inverse produces the input signal with minimum energy
        # that can explain the output mel spectrogram, so in general the energy estimate is too low. Therefore
        # the normalization factor will be too high. Visual signal inspection reveals that notably for
        # fricative phonemes the normalized signal has amplitude values that are higher than the voiced segments.
        # This seems unfortunate because it will lead
        # to the fact that unvoiced segments will tend to have a larger impact on the error.
        if use_pinv:
            wnorm = np.sum(get_stft_window(win_type="hann", win_len=win_size, dtype="float32") **2)
            mel_basis = get_mel_filter(sr=preprocess_config['sample_rate'], n_fft=preprocess_config['fft_size'],
                                       n_mels=preprocess_config["mel_channels"],
                                       fmin=preprocess_config["fmin"], fmax=preprocess_config["fmax"], dtype="float32")
            mbi = np.linalg.pinv(mel_basis)
            mell_test = np.dot(mell_test, mbi.T)/np.sqrt(wnorm)
            wnorm = 1
        else:

            n_mels = preprocess_config['mel_channels']
            mel_f = librosa_mel_frequencies(n_mels=n_mels + 2, fmin=preprocess_config["fmin"], fmax=preprocess_config["fmax"])
            inv_enorm = ((mel_f[2 : n_mels + 2] - mel_f[:n_mels]) / 2.).astype(np.float32)
            mell_test = mell_test * inv_enorm
            wnorm=win_size

    # if snd is not None:
    #     snd_length = snd.shape[-1]
    # else:
    #     snd_length = mell.shape[1] * hop_size 
    
    # print(f"snd.shape {snd.shape} mell.shape {mell.shape} {mell_test.dtype} self.rms_norm_fact {2/fft_size/win_size} rms_mel_ampl {np.sqrt(2*np.sum(mell_test**2, axis=-1)/fft_size/win_size).
    # shape} {np.mean(np.sqrt(2*np.sum(mell_test**2, axis=-1)/fft_size/win_size))}")

    ana_win = get_stft_window("hann", win_len=win_size, dtype=np.dtype("float32"))[np.newaxis, :]
    gain_ana_win = ana_win / np.sum(ana_win)


    smooth_win_size = int(win_size * smooth_win_scale)
    smooth_syn_win = get_stft_window("hann", win_len=smooth_win_size, dtype=np.dtype("float32"))[np.newaxis, :]
    if old_gain_smooth:
        # while for calculating the effect of an individual window for the gain of a single frame
        # we need to square the window, for estimating the effect of the individual gains applied to individual frames
        # we shoud not square the window, so we swicth this off here.
        smooth_syn_win = smooth_syn_win**2
    n_frames = mell.shape[1]
    if num_smooth_iters:
        for it in range(num_smooth_iters):
            if norm_max:
                max_mel_ampl = np.fmax(np.finfo(mell.dtype).eps, np.max(mell_test, axis=-1))
                norm_fact_hop_grid = max_mel_ampl
            else:
                # approximate signal energy as a sum of squares of mel amplitudes
                if it == 0:
                    rms_mel_ampl = np.sqrt(2*np.sum(mell_test**2, axis=-1)/fft_size/wnorm)
                    #print(f"rms_mel_ampl: {np.sqrt(np.sum(rms_mel_ampl**2)/mell_test.shape[1])}")
                    #rms_mel_ampl = 2*np.sum(mell_test, axis=-1)/fft_size/win_size
                    norm_fact_hop_grid = rms_mel_ampl.astype(np.float32)
                else:
                    norm_fact_hop_grid = mell_test[...,0]

            if norm_max_fac :
                norm_fact_hop_grid = np.fmax(norm_fact_hop_grid, 1/norm_max_fac)

            if normalize_compressor_exp is not None :
                norm_fact_hop_grid = norm_fact_hop_grid ** normalize_compressor_exp

            if np.min(norm_fact_hop_grid) <0 :
                from IPython import embed
                embed()
            # transform energy estimate sequence into normalization function for signal,
            # signal gain is extended by 1 win_size left and 1 hop_size + 1 win_size right
            # energy estimates will be extrapolated by 2 hops left and right
            # using values equal to the first and last value of the energy sequence
            if use_tf:
                import tensorflow as tf
                gain_frames = tf.linalg.matmul(tf.concat((norm_fact_hop_grid[:,:1], norm_fact_hop_grid[:,:1],
                                                          norm_fact_hop_grid,
                                                          norm_fact_hop_grid[:,-1:], norm_fact_hop_grid[:,-1:]), axis=1)[:,np.newaxis,:],
                                               smooth_syn_win, transpose_a=True)
                norm_gain_frames = tf.linalg.matmul(tf.concat((tf.ones((1, 2), dtype=np.float32),
                                                               tf.ones(norm_fact_hop_grid.shape, dtype=np.float32),
                                                               tf.ones((1, 2), dtype=np.float32)), axis=1)[:,np.newaxis,:], smooth_syn_win, transpose_a=True)
                gain = tf.signal.overlap_and_add(gain_frames, hop_size)[:,smooth_win_size//2+2*hop_size-win_size//2:]
                norm_gain = tf.signal.overlap_and_add(norm_gain_frames, hop_size)[:,smooth_win_size//2+2*hop_size-win_size//2:]
                gain = gain/tf.maximum(tf.keras.backend.epsilon(), norm_gain)
                realizable_mel_gain=tf.nn.conv1d(tf.expand_dims(gain, axis=2), gain_ana_win.T[:,:,np.newaxis],
                                                 stride=[1, hop_size,1], padding="VALID", data_format="NWC")[:,:mell.shape[1]]
            else:
                gain = np.zeros((mell.shape[0], ((mell.shape[1] + 4) * hop_size + smooth_win_size)), dtype=mell.dtype)
                norm_gain = np.zeros((1, gain.shape[1]), dtype=mell.dtype)
                # that is the origin of the signal is positioned at sig_start_in_gain = win_size//2 + 2 * hop_size
                # and the last sample of the signal is positioned at sig_end_in_gain = sig_start_in_gain + (mell.shape[1] * hop_size)
                # but these two values are not used below
                start_ind = 0
                # extrapolate energy 2 two steps at the left and right
                for ii in range(-2, norm_fact_hop_grid.shape[-1] + 3):
                    ii_cut = np.fmin(np.fmax(ii, 0), norm_fact_hop_grid.shape[-1] - 1)
                    gain[:, start_ind:start_ind + smooth_win_size] += (smooth_syn_win[..., 0:np.fmin(smooth_win_size, gain.shape[-1] - start_ind)]
                                                                * (norm_fact_hop_grid[..., ii_cut:ii_cut + 1]))
                    norm_gain[:, start_ind:start_ind + smooth_win_size] += smooth_syn_win[..., 0:np.fmin(smooth_win_size, gain.shape[-1] - start_ind)]
                    start_ind += hop_size


                # cut to have starting point exactly winsize//2 before the first frame center
                gain = gain[:,smooth_win_size//2+2*hop_size-win_size//2:]
                norm_gain = norm_gain[:,smooth_win_size//2+2*hop_size-win_size//2:]

                #print(f"norm_gain.shape {norm_gain.shape} norm_gain[0,[{win_size//2-1}, {win_size//2},{win_size//2+snd.size},{win_size//2+snd.size+hop_size}]]"
                #          f"{norm_gain[0, [win_size//2-1, win_size//2,win_size//2+snd.size,win_size//2+snd.size+hop_size]]}")
                #print(f"gain.shape {gain.shape} gain[0,[{win_size//2-1}, {win_size//2},{win_size//2+snd.size},{win_size//2+snd.size+hop_size}]]"
                #          f"{gain[0, [win_size//2-1, win_size//2,win_size//2+snd.size,win_size//2+snd.size+hop_size]]}")

                # pass signal gain factor through analysis window to derive the effective gain that will be observed in spectral domain
                #print(f"gain {np.mean(gain)} norm_gain {np.mean(norm_gain)}")
                gain = gain/np.fmax(np.finfo(mell.dtype).eps, norm_gain)
                gain_frames = as_strided(gain, shape=gain.shape[:-1] + (n_frames, win_size),
                                         strides=gain.strides[:-1] + (hop_size * gain.itemsize, gain.itemsize))

                realizable_mel_gain = np.sum(gain_frames*gain_ana_win, axis=-1, keepdims=True)[:,:mell.shape[1]]
                #print(f"rms_melampl {np.mean(realizable_mel_gain)} rms_mel_shape {realizable_mel_gain.shape}")

            mell_test = realizable_mel_gain
            if np.min(mell_test) <0 :
                from IPython import embed
                embed()
            if normalize_compressor_exp is not None:
                mell_test = mell_test ** (1/normalize_compressor_exp)
            gain_off = int(win_size // 2)

    else:
        realizable_mel_gain = np.sqrt(2 * np.sum(mell_test ** 2, axis=-1) / fft_size / wnorm)[:,:,np.newaxis]
        gain = scipy.interpolate.interp1d(np.arange(realizable_mel_gain.shape[1])*hop_size,
                                          realizable_mel_gain, axis=1, fill_value="extrapolate"
                                          )(np.arange((realizable_mel_gain.shape[1]+2)*hop_size))[:,:,0]
        gain_off = 0

    if mell_is_log:
        mel_out = mell - np.log(np.fmax(np.finfo(mell.dtype).eps, realizable_mel_gain))
    else:
        mel_out = mell / np.fmax(np.finfo(mell.dtype).eps, realizable_mel_gain)

    if snd is not None:
        if use_tf:
            gain = gain.numpy()
        snd_gain = np.fmax(gain[...,gain_off:gain_off+snd.shape[-1]], np.finfo(gain.dtype).eps)

        print(gain.shape, snd_gain.shape)
        if False:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(mell.shape[1]), np.log(norm_fact_hop_grid[0]), "rx")
            plt.plot(np.arange(mell.shape[1]), np.log(realizable_mel_gain[0, :mell.shape[1], 0]), "go")
            plt.plot(np.arange(snd_gain.shape[1])/hop_size, np.log(snd_gain[0]), "b")
            plt.grid()
            plt.show()

        if de_normalize_snd:
            snd_out = snd * snd_gain
        else:
            snd_out = snd / snd_gain
        if return_snd_gain:
            return mel_out, snd_out.reshape(snd.shape), snd_gain
        else:
            return mel_out, snd_out.reshape(snd.shape)
    else:
        snd_gain = np.fmax(gain[..., gain_off: ], np.finfo(gain.dtype).eps)

    if return_snd_gain:
        return mel_out, snd_gain
    else:
        return mel_out


def get_mel_lin_interpol_params(preprocess_config, n_fft):

    # get norm factor that inverts the scaling in the librosa mel basis
    mel_basis = get_mel_filter(sr=preprocess_config['sample_rate'],
                               n_fft=preprocess_config["fft_size"],
                               n_mels=preprocess_config["mel_channels"],
                               fmin=preprocess_config["fmin"], fmax=preprocess_config["fmax"], dtype=np.dtype('float32'))

    gain_fac = np.sum(mel_basis, axis=1)

    # determine extended mel frequencies that keep the original band centers but extend the range such that the full
    # frequency range is covered
    mel_frequencies = librosa_mel_frequencies(n_mels=preprocess_config["mel_channels"] + 2,
                                              fmin=preprocess_config["fmin"],
                                              fmax=preprocess_config["fmax"])
    # extend mel frequencies to cover the full frequency range
    dmel = (librosa_hz_to_mel(mel_frequencies[-1]) - librosa_hz_to_mel(mel_frequencies[0])) / (
                preprocess_config["mel_channels"] - 1)


    ext_low_int = int(np.floor((librosa_hz_to_mel(mel_frequencies[0]) - (-dmel)) / dmel))
    ext_low_hz = librosa_mel_to_hz(librosa_hz_to_mel(mel_frequencies[0]) - ext_low_int * dmel)
    ext_high_int = int(np.ceil((librosa_hz_to_mel(dmel + preprocess_config['sample_rate'])
                                - librosa_hz_to_mel(mel_frequencies[-1])) / dmel))
    ext_high_hz = librosa_mel_to_hz(librosa_hz_to_mel(mel_frequencies[-1]) + ext_high_int * dmel)

    # get the unnormalized mel basis that can be used to do linear interpolation of the mel frequencies
    mel_interpolator = get_mel_filter(sr=preprocess_config['sample_rate'],
                               n_fft=n_fft,
                               n_mels=preprocess_config["mel_channels"] + ext_low_int + ext_high_int,
                               fmin=ext_low_hz, fmax=ext_high_hz, norm=False,
                               dtype=np.dtype('float32'))

    return gain_fac, mel_interpolator, ext_low_int, ext_high_int


def compute_mel_spectrogram_internal(sound, preprocess_config, dtype=np.dtype('float32'),
                                     force=False, band_limit=None, pad_mode = "reflect",
                                     center=True, return_STFT=False, do_post=True,
                                     return_band_limited_signal=True,
                                     return_band_limited_mel=True):
    '''
    Compute log amplitude mel spectrogram from sound array with dimension batch_size x time_dim
    or batch_size x channels x time_dimensions
    
    the output will have dimensions batch_size x time x mel_channels

    The band_lim parameter given as a three tuple of the form [band_lim_low_hz, band_lim_high_hz, band_stop_high_hz]
    allows filtering the input sound using an three point STFT domain filter that is constructed using a bpf given in form of
    (frequency, linear amplitude) tuples:
    ((0, 0), (band_lim_low_hz, 0), (band_lim_low_hz,1), (band_lim_low_hz,1), (band_stop_high_hz,  0)

    In case the band_limit paramter is not None two other flags control the generated output tuple

    return_band_limited_signal : bool (Default True)
       if set to true the output tuple is extended to include the signal corresponding to the band limited sound
       if False only the mel spectrogram together with its frequency is returned

    return_band_limited_mel : bool (Default True)
        if set the mel spectrogram is created after filtering with the band pass filter.
        otherwise the Mel spectrogram of the original sound is returned

    band limiting use cases:

      return_band_limited_signal=True, return_band_limited_mel=True

         this setup allows creating batches with arbitrarily band limited signals with the corresponding
         mel spectrogram.

      return_band_limited_signal=True, return_band_limited_mel=False

         can be used to provided band limited input signals for band enhacement in combiation with the mel spectrogram
         of the full band signal -> independently training stages in waveSGen

      return_band_limited_signal=False, return_band_limited_mel=True

         used for evaluating band enhancement with individual stages where the target signal is the band limited output
         of the given stage

      return_band_limited_signal=False, return_band_limited_mel=False

          the band limiting filter is not used, so this setup does not mke sense.


    :param sound: 2d input sound signal, time in the last dimension
    :type sound: np.array
    :param preprocess_config: dict containing all pre-processing parameters
    :type preprocess_config: dict[str, Union[str,int,float]]
    :param dtype:
    :type dtype: np.dtype
    :param rand_filt_len:
    :param rand_filt_amp:
    :param force:
    :param band_limit: a descriptor describing the band filter to be applied to the sound signal. The band filter
        is described using three values [band_lim_low_hz, band_lim_high_hz, band_stop_high_hz].
    :type band_limit: Union(None, tuple[float, float, float]]
    :param pad_mode:
    :param center:
    :param return_STFT:
    '''

    # from sig_proc.spec.stft import calc_stft
    if (not force) and (np.max(sound.shape) != sound.shape[-1]):
        raise RuntimeError('sound shape is not maximal in the last dimension, if you are sure the last dimension '
                           'is time you can force processing with the force argument')
    if sound.ndim == 1:
        sound =sound[np.newaxis, :]

    win_len = preprocess_config['fft_size']
    if 'win_size' in preprocess_config:
        win_len = preprocess_config['win_size']

    if band_limit is None:

        S = calc_stft(sound, win_len=win_len, hop_len=preprocess_config['hop_size'],
                      fft_size = preprocess_config['fft_size'], win_type = 'hann', center = center,
                      pad_mode = pad_mode, do_mag = True, axis = -1, dtype = dtype)
        #print(f"max spectrogram {np.max(np.abs(S))} mean spectrogram {np.mean(np.abs(S))} ")
    else:
        if band_limit :
            if len(band_limit) != 3:
                raise RuntimeError("compute_mel_spectrogram_internal::if the band_limit parameter is not None it needs "
                                   "to contain three values: band_lim_low_hz, band_lim_high_hz, band_stop_high_hz")

        spec = STFT(x=sound, win = get_stft_window("hann", win_len=win_len, dtype=dtype),
                    step=preprocess_config['hop_size'], fft_size=preprocess_config['fft_size'],
                    SR=preprocess_config['sample_rate'], extend_outside=center, pad_mode=pad_mode)
        binFreqs = spec.get_center_frequencies()
        filt = np.ones((1, binFreqs.size), dtype=spec.dtype)

        if band_limit[0]:
            ind = find_1st(binFreqs, band_limit[0], cmp_larger)
            if ind >= 0:
                filt[:,:ind] = 0
        if band_limit[1]:
            ind_high = find_1st(binFreqs, band_limit[1], cmp_larger)
            ind_stop = find_1st(binFreqs, band_limit[2], cmp_larger)

            if ind_high >= 0 and ind_stop > ind_high:
                filt[:,ind_high:ind_stop] = np.linspace(1,0, ind_stop- ind_high)
            if ind_stop > 0:
                filt[:,ind_stop:] = 0

        if not return_band_limited_mel:
            S = np.abs(spec.get_data())

        spec.stft_data *= filt
        sound = spec.resynthesize()

        if return_band_limited_mel:
            S = np.abs(spec.get_data())

    #print("compute_mel_spectrogram_internal::get mell basis", file=sys.stderr)
    #sys.stderr.flush()
    mel_basis = get_mel_filter(sr=preprocess_config['sample_rate'], n_fft=preprocess_config['fft_size'],
                               n_mels = preprocess_config["mel_channels"],
                               fmin = preprocess_config["fmin"], fmax = preprocess_config["fmax"], dtype = dtype)

    #print(f"sound dimension {sound.shape}, S dimension {S.shape} melbas {mel_basis.shape}")
    mel_spectrogram = np.dot(S, mel_basis.T)

    if do_post:
        if ("norm_mel" in preprocess_config) and preprocess_config["norm_mel"]:
            mel_spectrogram, sound = norm_mell(mel_spectrogram, preprocess_config=preprocess_config,
                                               snd=sound, mell_is_log=False, center=center)

        mell = scale_mel_spectrogram(mel_spectrogram=mel_spectrogram, preprocess_config=preprocess_config)
    else:
        mell = np.log(np.fmax(mel_spectrogram, np.finfo(mel_spectrogram.dtype).eps))

    #print(f"max lmel spectrogram {np.max(mell)} mean lmel spec {np.mean(mell)}")
    mel_srate = preprocess_config['sample_rate'] / preprocess_config['hop_size']

    if do_post and ((("norm_mel" in preprocess_config) and preprocess_config["norm_mel"])
                     or ((band_limit is not None) and return_band_limited_signal)):
        if return_STFT:
            return mell, mel_srate, S, sound
        else:
            return mell, mel_srate, sound

    if return_STFT:
        return mell, mel_srate, S

    return mell, mel_srate


if __name__ == "__main__":

    from argparse import ArgumentParser
    from sig_proc import db
    from sig_proc.spec.fft import rfft
    from matplotlib import pyplot as plt

    parser = ArgumentParser(description="test filter coefficients")
    parser.add_argument("--rand_amp", default=0.1, type=float, help="maximum amplitude of random filter coefficients (def: %(default)s)")
    parser.add_argument("--rand_len", default=2, type=int, help="number of random filter coefficients (def: %(default)s)")
    args = parser.parse_args()

    min_fil = None
    max_fil = None
    for _ in range(1000):
        fil = gen_filt(args.rand_len, args.rand_amp)
        fil_spec = rfft(fil, 1024)
        if min_fil is None:
            min_fil = db.lin2db(fil_spec)
            max_fil = db.lin2db(fil_spec)
        else:
            min_fil = np.fmin(min_fil, db.lin2db(fil_spec))
            max_fil = np.fmax(max_fil, db.lin2db(fil_spec))

    plt.plot(min_fil, label="min_fil")
    plt.plot(max_fil, label="max_fil")
    plt.grid(True)
    plt.legend()
    plt.show()
