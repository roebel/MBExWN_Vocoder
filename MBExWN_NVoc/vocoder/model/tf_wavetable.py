# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
# source wavetable implementation
#
# -----------------------------
# Acknowledgements
#
# PulseWaveTable.stable_cumsum_and_wraphas been copied and adapted from the function angular_cumsum in
# https://github.com/magenta/ddsp/blob/main/ddsp/core.py
# Copyright of the original version was:
# Copyright 2022 The DDSP Authors.
#
# See the related code sections below for changes.
# -----------------------------


import os
import numpy as np
import tensorflow as tf
from typing import List, Union, Any, Sequence, Dict, Tuple
import copy
from ...sig_proc.resample import resample
from ...glottis.FglotspecLF import FglotspecLF
from ...sig_proc import window as sig_window

from .preprocess import get_mel_filter, get_filters, get_mel_lin_interpol_params

import scipy.signal as ss

log_to_db = 20 * np.log10(np.exp(1))



def get_pulse_lowpass_kaiser(pass_band_edge, stop_att_db=70, trans_width_normed=0.1, dtype=np.float32):
        """
        Construct a FIR low pass filter having the first local minima (spectral zero) given by the pass_band edges.
        The 6dB will be located approximately at (cut_off - 0.5* trans_width_normed) for the upper end of the
        pass band and (cut_off + 0.5 * trans_width_normed) for the lower end of each pass band.
        Pass_band_edges neded o be separted by at least 2*trans_width_normed

        pass_band_edges are given as a 2D array with pass bands in the rows and the corresponding edges in the columns.
        Pass_band edges are given relative to the sample rate, that means the Nyquist frequency is 0.5.

        The transition between the -6db pass band edge and the stop band edge will have a width given by
        0.5*trans_width_normed*sample rate and the transition from the las time the pass band has
        0dB until the first time the stop band reaches stop_att_db will be (trans_width_normed*sample rate)

        cut_off frequencies are alternating between transitions to pass band and transitions to stop band.
        For constructing a low pass the pass_band_edges should start with 0
        """
        # Calculate the window parameters for kaiser window given the desired transition band width
        if(stop_att_db >= 50):
            mBeta = 0.1102 * (stop_att_db - 8.7)
        elif (stop_att_db >= 21):
            mBeta = 0.5842 * pow(stop_att_db - 21., 0.4) + 0.07886 * (stop_att_db - 21.)
        else:
            mBeta = 0.

        mTransWidth = 2* np.pi * trans_width_normed

        cut_off = []
        pass_zero = True
        cut_off = [pass_band_edge - 0.5 * trans_width_normed]

        # radius for filter
        while True:
            mRadius = np.int(np.ceil((stop_att_db - 8.) / 2.285 / mTransWidth / 2))
            # print("resampler::mRadius {} cond {} stop_att {} beta {}\n".format(mRadius,2*mRadius>8000,stop_att,mBeta))
            if ((2 * mRadius > 8000) and stop_att_db > 10):
                stop_att_db -= 6
            else:
                break

        winlen = mRadius * 2 + 1
        # prepare dimensions for conv2d
        #print(f"winlen {winlen}, pass_band_edge {pass_band_edge}, cut_off: {cut_off}")
        return ss.firwin(winlen, cutoff=cut_off, window=("kaiser", mBeta), pass_zero=pass_zero, fs=1.)

def get_min_phase_spectrum( log_magnitude):
    fft_size = log_magnitude.shape[-1] * 2 - 2
    real_cepst = np.fft.irfft(np.fmax(log_magnitude, np.finfo(log_magnitude.dtype).eps), n=fft_size)
    cepst_min_phase_mask = np.concatenate(([1.],
                                           2 * np.ones(fft_size // 2 - 1),
                                           [1.]), axis=0)
    log_spect = np.fft.rfft(real_cepst[:cepst_min_phase_mask.shape[0]] * cepst_min_phase_mask, n=fft_size)
    return np.exp(log_spect)



def get_LFpulse(n_wavetable, oq=0.5, am=0.7, rta=0.1, pul_bw=0.1,
                use_deriv=False, transition_width=0.1, quiet=False, norm=False,
                white_pulse=False):
    T0 = n_wavetable
    samp_rate_Hz = 1

    fft_size = 16
    while fft_size < n_wavetable :
        fft_size *= 2
    if not quiet and (n_wavetable != fft_size):
        print(f"adapt n_wavetable {n_wavetable}  to fft_size {fft_size}")

    fft_freq_hz = np.arange(fft_size // 2 + 1) * samp_rate_Hz / fft_size

    syn_pulse_spec = FglotspecLF(fft_freq_hz * T0, oq=oq, am=am, ta=rta * (1 - oq), get_derivative=use_deriv,
                                     orig=0)[0]

    if white_pulse :
        if not quiet:
            print(f" create whitened pulse")

        n_max_pulse_pos= np.argmax(syn_pulse_spec)
        n_max_white_pos = np.fmax(n_max_pulse_pos, int(fft_size * (pul_bw - 0.5*transition_width)))
        wfilt = np.ones(syn_pulse_spec.shape)
        if n_max_pulse_pos < n_max_white_pos:
            wfilt[n_max_pulse_pos:n_max_white_pos] = np.abs(syn_pulse_spec[n_max_pulse_pos])/np.abs(syn_pulse_spec[n_max_pulse_pos:n_max_white_pos])
            wfilt[n_max_white_pos:] =  np.abs(syn_pulse_spec[n_max_pulse_pos])/np.abs(syn_pulse_spec[n_max_white_pos])
            syn_pulse_spec *= get_min_phase_spectrum(np.log(wfilt))

    plot = False
    if plot:
        ori_pulse_spec = syn_pulse_spec

    fcoef = get_pulse_lowpass_kaiser(pul_bw, stop_att_db=70, trans_width_normed=np.fmin(pul_bw / 2., transition_width),
                                     dtype=np.float)
    filter_fftsize_factor = 1
    while fcoef.shape[0] > fft_size * filter_fftsize_factor:
        filter_fftsize_factor *= 2
    # frequency domain sub sampling leads to temporal aliasing which however does not harm too much for our quasi periodic use
    # of the waveforms in the wavetables
    filter_fft = np.fft.rfft(fcoef, fft_size*filter_fftsize_factor)[::filter_fftsize_factor]
    filter_fft[-1] = np.real(filter_fft[-1])
    syn_pulse_spec *= np.abs(filter_fft)


    #plt.plot(fft_freq_hz, db.lin2db(np.abs(syn_pulse_spec), minthresh=0.000001), label=f"{pul_bw}")
    #plt.legend()
    pp = np.fft.irfft(syn_pulse_spec, fft_size)

    if plot:
        from matplotlib import pyplot as plt
        plt.plot(fft_freq_hz,np.log10(np.fmax(np.abs(filter_fft), 1e-7))*20)
        plt.plot(fft_freq_hz,np.log10(np.fmax(np.abs(ori_pulse_spec), 1e-7))*20)
        plt.plot(fft_freq_hz,np.log10(np.fmax(np.abs(syn_pulse_spec), 1e-7))*20)
        plt.plot(fft_freq_hz,np.log10(np.fmax(np.abs(np.fft.rfft(pp, fft_size)), 1e-7))*20)
        plt.grid()
        plt.show()


    # plt.plot(fft_freq_hz, db.lin2db(np.abs(np.fft.rfft(pp,fft_size)), minthresh=0.000001))
    # plt.title("get_LFpulse")
    # plt.grid(True)

    if norm:
        if use_deriv:
            pp = -pp / np.min(pp)
        else:
            pp = pp / np.max(pp)

    return pp

def pad_axis(x, padding=(0, 0), axis=0, **pad_kwargs):
  """Pads only one axis of a tensor.

  Args:
    x: Input tensor.
    padding: Tuple of number of samples to pad (before, after).
    axis: Which axis to pad.
    **pad_kwargs: Other kwargs to pass to tf.pad.

  Returns:
    A tensor padded with padding along axis.
  """
  n_end_dims = len(x.shape) - axis - 1
  n_end_dims *= n_end_dims > 0
  paddings = [[0, 0]] * axis + [list(padding)] + [[0, 0]] * n_end_dims
  return tf.pad(x, paddings, **pad_kwargs)

class PulseWaveTable(tf.keras.layers.Layer):
    def __init__(self,
                 sample_rate: Union[int, float],
                 # this is the F0 that can be achieved with the given nominalBandwidth
                 # for smaller F0s the wavetable will be dilated (resampling with linear interpolation ) and therefore
                 # the effectiv bandwidth will be reduced!
                 # For higher F0s the wavetable content needs to be compressed and therefore band-width
                 # will increase leading to aliasing  here the resampled wavetables with lower bandwidth will be
                 # used to reduce the aliasing effect
                 #
                 nominalF0 : float,
                 nominalBandWidth : Union[float, None] = None,
                 Oq : float =0.5,
                 am : float = 0.8,
                 rta : float = 0.05,
                 use_radiation : bool = False,
                 # prepare dedicated wavetable entries for the Fundamentals in grid
                 # the only difference in the wavetable will be the band limitation
                 # to avoid strong oversampling for higher  F0s
                 F0GridFactor : float = 1.25,
                 # the grid will contain this many F0s, only used if maxF0 is not given.
                 numF0InGrid: int = 5,
                 # maxF0 - if given it will be used to calculate numF0InGrid the value of will then be replaced
                 maxF0: Union[float,None] = None,
                 wt_oversampling: int =2,
                 pulse_sync_gain_avg: bool = False,
                 no_interp : bool= False,
                 trainable : Union[bool, None]= None,
                 use_sinusoid: bool = False,
                 use_sinusoid_as_fun: bool = False,
                 use_white_pulse: bool = False,
                 add_subharm_chans: int = 0,
                 quiet= False,
                 name="PulseWavetTable"):

        super(PulseWaveTable, self).__init__(trainable=trainable, name=name)
        self.Oq= Oq
        self.am = am
        self.rta = rta
        default_bandwidth =  0.5/F0GridFactor
        if nominalBandWidth is not None and np.abs((nominalBandWidth - default_bandwidth)/default_bandwidth) > 0.0001:
            print(f"ATTENTION: Overwriting the default pulse bandwidth {default_bandwidth} with {nominalBandWidth} will most likely lead to suboptimal performance")
        self.nominalBandWidth = default_bandwidth if nominalBandWidth is None else nominalBandWidth
        self.use_radidation = use_radiation
        self.sample_rate = sample_rate
        self.nominalF0 = nominalF0
        self.numF0InGrid = numF0InGrid
        self.maxF0 = maxF0
        self.F0GridFactor = F0GridFactor
        self.wt_oversampling = wt_oversampling
        self.add_subharm_chans = add_subharm_chans
        self.wavetables = None
        # create the pulse for the nominal F0 of this pulse generator as well as all the versions for other
        # target F0 that contain the same pulse
        # with different bandwidth to prevent aliasing
        self.F0_list = []
        wavetable_list = []

        self.use_sinusoid = use_sinusoid or use_sinusoid_as_fun
        self.use_sinusoid_as_fun = use_sinusoid_as_fun
        self.use_white_pulse = use_white_pulse
        # try to generate a pulse table entry with extreme parameters (maximum bandWidthReductionFactor)
        # and use the realizable nominalF0 for the followig setup
        _, nominalF0 = self.create_normed_pulse(Oq, target_nominalF0=self.nominalF0,
                                                nominalBandWidth=0.5 / F0GridFactor,
                                                sample_rate=sample_rate, am=am, rta=rta,
                                                use_radiation=use_radiation, bandWidthReductionFactor=maxF0/nominalF0,
                                                wt_oversampling=wt_oversampling,
                                                return_nominal_f0=True, quiet=quiet,
                                                use_sinusoid=use_sinusoid, use_white_pulse=use_white_pulse)
        if not quiet:
            print(f"adapted nominal F0 from {self.nominalF0} to { nominalF0} avoid any effects of cutting the wavetable after frequency domain filtering")
        self.nominalF0 = nominalF0
        if not use_sinusoid:
            used_numF0InGrid = numF0InGrid
            if maxF0 is not None:
                used_numF0InGrid = np.cast[np.int32](np.ceil(np.log(maxF0/nominalF0) / np.log(F0GridFactor)))
        else:
            used_numF0InGrid = 0

        if not quiet:
            print("Target F0: ", end="")
        for ir in range(used_numF0InGrid+1):
            if ir > 0:
                rs = F0GridFactor ** ir
            else:
                rs = 1
            if not quiet:
                print(f" {rs*nominalF0:.2f}", end="")

            wavetable = self.create_normed_pulse(Oq, target_nominalF0= self.nominalF0,
                                                 nominalBandWidth = 0.5,
                                                 sample_rate=sample_rate, am = am, rta = rta,
                                                 use_radiation=use_radiation, bandWidthReductionFactor=rs,
                                                 wt_oversampling=wt_oversampling, use_sinusoid=use_sinusoid,
                                                 quiet=quiet, use_white_pulse=use_white_pulse).astype(np.float32)
            self.F0_list.append(self.nominalF0 * rs)
            # Add first sample to end of wavetable for smooth linear interpolation
            # between the last point in the wavetable and the first point.
            wavetable_list.append(np.concatenate([wavetable, wavetable[0:1]], axis=0)[:,np.newaxis])

        if not quiet:
            print(flush=True)
        self.minTranspositionFactorInGrid = tf.constant(np.min(self.F0_list)/self.nominalF0, tf.float32)
        self.maxTranspositionFactorInGrid = tf.constant(np.max(self.F0_list)/self.nominalF0, tf.float32)
        norm_factor = -np.min([wavetable_list])

        if trainable is None:
            self.wavetables = tf.concat([wl/norm_factor for wl in wavetable_list], axis=1)
        else:
            self.wavetables = tf.Variable(tf.concat([wl/norm_factor for wl in wavetable_list], axis=1), trainable=trainable)
        self.n_wavetable = self.wavetables.shape[0]
        self.n_period = int(self.wavetables.shape[0] - 1)
        if False:
            import matplotlib.pyplot as plt
            from sig_proc import db
            plt.figure()
            fft_size = wavetable_list[-1].shape[0] - 1

            for iw in range(self.wavetables.shape[1]):
                plt.plot(np.linspace(0, sample_rate/2 * wt_oversampling,fft_size//2+1),
                         db.lin2db(np.abs(np.fft.rfft(self.wavetables[:-1,iw], ))),
                         label=f"F0 {self.F0_list[iw]}")
        # Get a phase value for each point on the wavetable.
        self.grid_f0_diff_norm_factor = 1. / tf.math.log(F0GridFactor)
        self.no_interp = no_interp
        self.pulse_sync_gain_avg= pulse_sync_gain_avg

    @staticmethod
    def create_normed_pulse(Oq : float, target_nominalF0 : float, nominalBandWidth : float,
                            sample_rate: int, am:float=0.8, rta:float =0.1,
                            use_radiation:bool =False, bandWidthReductionFactor:float =1.,
                            wt_oversampling: int =1, return_nominal_f0=False,
                            quiet=False, use_sinusoid=False, use_white_pulse=False):
        """
        from sympy import *
        T0,oq,t=symbols('T0 oq t')
        a=1/(oq*oq*T0)
        b=1/(oq*oq*oq*T0*T0)
        f=a*t*t-b*t*t*t
        # nullstellen
        solve(f, t)
        # -> [0, T0*oq]
        # Minimum and Maximum
        solve(diff(f,t), t)
        # -> [0, 2*T0*oq/3] the first minimum is second order

        Theoretical considerations and relation between signal parameters
        =================================================================
        R: sample rate
        WS: wavetable size
        F0: The F0 generated with the wavetable

        The natural F0 that will be gerated with a wavetable of length WS and sample rate R is

        F0 = R/WS

        That means the WS for a given nominalF0 and R is  WS = R/F0!

        The optimal band limit of the pulse in the wavetable is R/2
        Now fo higher F0s the wavetable will be stepped through with step size > 1 which transposes the signal up
        and leads to aliasing if the badnwidth of the pulse is not reduced.  We handle ths effect by means of
        band limiting the pulse to R/F0GridFactor and create more wavetables in a logarithmic grid of F0s
        F0_i = F0 * F0GridFactor ** i

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ATTENTION: If we create wavetable of size M = 2 WS and want to play a sound with F0 then we need to go
        twice as fast through the wavetable which will compress the effective duration and therefore dilate the
        the spectrum (shift the envelope up) !!!

        That means that for playing a given entry in the wavetable without timbre distortion we need to play
        it with step size 1
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Initialization of WaveTable
        ===========================
        To avoid artifacts the Wavetable initialization needs to take into account that sampling a mathematical function creates
        aliasing due to the fact that the original function potentially has unlimited bandwidth. To avoid this effect the most
        robust strategy is to create it directly in the spectral domain so that the wavetable is coherently band limited.

        An alternative strategy that will work similarly if the signal has low-pass chaacter is to create it with a
        larger wavetable for higher sample rate and then resample with classical band limited interpolation

        1) Select wavetable size

        WS = R / nominalF0

        2) Band limitation

        Create a wavetable with length WS * O and downsample it to WS using band limited resampling.

        Initialization for other F0s
        ==============================
        bandWidthReductionFactor:

        The argument bandWidthReductionFactor indicates the amount of band limitation that should be
        applied to the wavetable waveform. This is required to avoid aliasing when the waveform is played back
        with higher pitch (so after sub sampling it).

        wt_oversampling: oversampling factor for the representation of the waveform in the wavetable.
        The waveform to be used for storing it in the wavtable will be upsamled by means of
        bandlimited interpolation. This allows reducing interpolation errors and
        therefore achieves better quality.

        """

        if use_sinusoid:
            period = int(wt_oversampling * np.floor(sample_rate / target_nominalF0))
            res = np.sin(np.arange(period)/period* np.pi * 2) * ss.hanning(period, sym=False)
            nominalF0 = wt_oversampling * sample_rate/ period
        else:
            res = get_LFpulse(int(np.ceil(wt_oversampling * sample_rate / target_nominalF0)), oq=Oq,
                              am=am, rta=rta, pul_bw=nominalBandWidth / (bandWidthReductionFactor*wt_oversampling),
                              transition_width = 0.1/wt_oversampling,
                              use_deriv=use_radiation, quiet=quiet, white_pulse=use_white_pulse)
            nominalF0 = wt_oversampling * sample_rate/ res.shape[0]
            if False and wt_oversampling > 1:
                resr = get_LFpulse(int(np.ceil(sample_rate / target_nominalF0)), oq=Oq,
                                   am=am, rta=rta, pul_bw=nominalBandWidth / bandWidthReductionFactor,
                                   transition_width = 0.1,
                                   use_deriv=use_radiation, quiet=quiet, white_pulse=use_white_pulse)
                resr = resample(resr, in_sr=1, out_sr=wt_oversampling)[0]
                from IPython import embed
                embed()
        #from IPython import embed
        #embed()
        if return_nominal_f0:
            return res, nominalF0

        return res

    @property
    def trainable_weights(self):
        if self.trainable:
            return [self.wavetables]
        else:
            return []


    def __call__(self, frequency : Union[tf.Tensor,Sequence[float]],
                 open_quotient: Union[None,Sequence[float]]= None,
                 pulse_gain_list: Union[None, List[Sequence[float]]]= None,
                 return_gain =False,
                 ):

        return self.call(frequency, open_quotient=open_quotient, pulse_gain_list=pulse_gain_list, return_gain=return_gain)


    def stable_cumsum_and_wrap(self, phase_velocity, chunk_size=1000):
        """Get phase by cumulative summation of phase velocity.

        Adapted from ddsp.core.angular_frequency
        - Copyright 2021 The DDSP Authors
        - http://www.apache.org/licenses/LICENSE-2.0

        Changes: variables have been renamed and outut rnage is now in 0,1 and n longer in [0, 2pi]

        Custom cumsum splits first axis into chunks to avoid accumulation error.
        Just taking tf.cumsum(phase_velocity) %1 overflows the float32 precision quite easily leading to
        strange frequency modulationsthat are audible for long segments or at high sample rates.

        Given that we are using the velocity to sample the wavetable in the range [0,1] only,
        we don't care about modulo 1 changes. This code chops the incoming frequency
        into chunks, applies cumsum to each chunk, takes mod 1, and then stitches
        them back together by adding the cumulative values of the final step of each
        chunk to the next chunk.

        Args:
          phase_velocity: period velocity. Shape [batch, time, ...].
            If there is no batch dimension, one will be temporarily added.
          chunk_size: Number of samples per a chunk.

        Returns:
          The accumulated phase modulo 1 in range [0, 1], shape [batch, time, ...].
        """

        n_batch = phase_velocity.shape[0]
        n_time = phase_velocity.shape[1]
        n_dims = len(phase_velocity.shape)
        n_ch_dims = n_dims - 2

        # Pad if needed.
        remainder = n_time % chunk_size
        if remainder:
            pad = chunk_size - remainder
            phase_velocity = pad_axis(phase_velocity, [0, pad], axis=1)

        # Split input into chunks.
        length = phase_velocity.shape[1]
        n_chunks = int(length / chunk_size)
        chunks = tf.reshape(phase_velocity,
                            [n_batch, n_chunks, chunk_size] + [-1] * n_ch_dims)
        phase = tf.cumsum(chunks, axis=2)

        # Add offsets.
        # Offset of the next row is the last entry of the previous row.
        offsets = phase[:, :, -1:, ...] % 1
        offsets = pad_axis(offsets, [1, 0], axis=1)
        offsets = offsets[:, :-1, ...]

        # Offset is cumulative among the rows.
        offsets = tf.cumsum(offsets, axis=1) % 1
        phase = phase + offsets

        # Put back in original shape and force target range
        phase = phase % 1
        phase = tf.reshape(phase, [n_batch, length] + [-1] * n_ch_dims)

        # Remove padding if added it.
        if remainder:
            phase = phase[:, :n_time]
        return phase

    #@tf.function
    def call(self, frequency : tf.Tensor,
             open_quotient: Union[None, tf.Tensor]= None,
             pulse_gain_list: Union[None, List[tf.Tensor]]= None, return_gain=False):
        """
        generates periodic signal from wavetable
        input:
           frequency target F0 for the periodic signal (B, T)

        pulse_gain_list : a list of gain vectors  that will be used to derive pulse gains depeding on the mode of the wavetable
               by means of either
               - sampling at the pulse start
               - averaging over the pulse duration
               - or sampled at a random position within the pulse


        output: periodic signal (B x T)

        """
        # convert frequency into phase_velocity
        phase_velocity = frequency / self.sample_rate

        # Note: Cumsum accumulates _very_ small errors at float32 precision.
        # On the order of milli-Hertz.
        wrapped_phase = self.stable_cumsum_and_wrap(phase_velocity)

        if self.use_sinusoid_as_fun or self.add_subharm_chans:
            wrapped_phase_2pi = wrapped_phase * 2 * tf.constant(np.pi)
        if self.use_sinusoid_as_fun:
            audio = (tf.sin(wrapped_phase_2pi) * 0.5 * (1. - tf.cos(wrapped_phase_2pi)))[:, :, tf.newaxis]
            if not self.add_subharm_chans:
                return audio
        else:

            # Synthesize with linear lookup.
            if self.pulse_sync_gain_avg:
                audio, pulsed_gains = self._linear_lookup_with_gain(phase = wrapped_phase,
                                                                   pulse_inst_gains = tf.stack(pulse_gain_list, axis=0))
            else:
                audio = self._linear_lookup(wrapped_phase)

            # create weighted interpolation to get wavetable entry with optimal band limitation
            # frequency is B x T, wavetable synthesis are B x T x R and target resampling factors are 1 x 1 x R

            # create the frequency ratio compared to nominal F0
            log_freq_nomF0_ratio = tf.math.log(tf.maximum(self.minTranspositionFactorInGrid,
                                                   tf.minimum(self.maxTranspositionFactorInGrid,
                                                              frequency/ tf.cast(self.nominalF0, tf.float32))))[:,:,tf.newaxis]

            freq_grid_diff = log_freq_nomF0_ratio * self.grid_f0_diff_norm_factor - tf.cast(tf.range(audio.shape[-1]), tf.float32)

            # thanks to ddsp mixing factors can be obtained by means of passing the differences through a relu
            if True:
                # This reduce_sum variant seems a little bit faster than matmul below.
                audio = tf.reduce_sum(audio * tf.maximum(1 - tf.abs(freq_grid_diff), 0), axis=2)[:,:,tf.newaxis]
            else:
                audio = tf.linalg.matmul(audio[:,:,tf.newaxis,:],
                                         tf.maximum(1 - tf.abs(freq_grid_diff), 0)[:,:,tf.newaxis, :], transpose_b=True)
                audio = audio[:,:,:1,0]

        if self.add_subharm_chans:
            audio_list = [audio]
            for ii in range(2, self.add_subharm_chans + 2):
                audio_list.append(tf.sin(wrapped_phase_2pi / ii)[:, :, tf.newaxis])

            audio = tf.concat(audio_list, axis=-1)

        # this will become a pulse synchronous gain
        if pulse_gain_list is not None:
            if self.pulse_sync_gain_avg:
                if return_gain:
                    audio_or_audio_list = audio
                    gain_list  = [pg  for pg in tf.unstack(pulsed_gains, axis=0)]
                else:
                    audio_or_audio_list = [audio * pg  for pg in tf.unstack(pulsed_gains, axis=0)]
            else:
                if return_gain:
                    audio_or_audio_list = audio
                    gain_list  = []
                else:
                    audio_or_audio_list = []

                for pulse_gain in pulse_gain_list:
                    if pulse_gain is None:
                        if return_gain:
                            gain_list.append(None)
                        else:
                            audio_or_audio_list.append(None)
                    else:
                        # BxT boolean mask for all pulse_starts (including the first)
                        gain_mask = tf.concat((tf.ones((pulse_gain.shape[0], 1), tf.bool),
                                   wrapped_phase[:, 1:] < wrapped_phase[:, :-1]), axis=1)
                        # sample gain at the start of each pulse, pulse starts are reshaped into a 1D vector with all pulse starts concatenated
                        pulse_gain_at_bounds  = pulse_gain[gain_mask]
                        # phase is the cumulative phase offset into the periods with all integers representing the start of a new period
                        # so floor(phase) gets the number of the period and is constant over the duration of the period
                        pulse_bound_gain_inds = tf.reshape(tf.cumsum(tf.reshape(tf.cast(gain_mask, tf.int64), (-1,))) - 1, gain_mask.shape)
                        full_gain = tf.gather(pulse_gain_at_bounds, tf.reshape(pulse_bound_gain_inds, pulse_gain.shape), axis=0, batch_dims=0)

                        if return_gain:
                            gain_list.append(full_gain)
                        else:
                            audio_or_audio_list.append(audio * full_gain)

            if return_gain:
                return audio_or_audio_list, gain_list
            return audio_or_audio_list

        return audio

    # Wavetable Synthesizer --------------------------------------------------------
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def _linear_lookup(self, phase: tf.Tensor) -> tf.Tensor:
        """Lookup from wavetables with linear interpolation.

        Args:
        phase: The instantaneous phase of the base oscillator, ranging from 0 to
          1.0. This gives the position to lookup in the wavetable.
          Shape [batch_size, n_samples] .

        Returns:
        The resulting audio from linearly interpolated lookup of the wavetables at
          each point in time. Shape [batch_size, n_samples].
        """

        # For ease of linear interpolation we increase phase steps such that a step from one sample to the next
        # in the wavetable has value 1.
        phase_wt = phase * self.n_period

        if self.no_interp:
            base_wt_inds = tf.cast(tf.round(phase_wt), tf.int32)

            # wavetable dimension is n_period x k (different versions with different band limits for
            # subsampling when high F0s are desired
            samples_quant= tf.gather(self.wavetables, base_wt_inds, axis=0, batch_dims=1)
            return samples_quant
        else:
            phase_wt_quant = tf.floor(phase_wt)
            phase_rem = (phase_wt - phase_wt_quant)[:,:,tf.newaxis,tf.newaxis]
            base_wt_inds = tf.cast(phase_wt_quant, tf.int32)[:,:,tf.newaxis]
            next_wt_inds = base_wt_inds + 1
            samples_quant = tf.gather(self.wavetables, tf.concat((base_wt_inds, next_wt_inds), axis=2), axis=0, batch_dims=1)

            #print(self.wavetables.shape, tf.concat((base_wt_inds, base_wt_inds+1), axis=2).shape, samples_quant.shape, phase_rem.shape)
            return tf.reduce_sum( samples_quant * tf.concat((1. -phase_rem, phase_rem), axis=2), axis=2)


    # Wavetable Synthesizer --------------------------------------------------------
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)))
    def _linear_lookup_with_gain(self, phase: tf.Tensor, pulse_inst_gains : tf.Tensor) -> tf.Tensor:
        """Lookup from wavetables with linear interpolation.

        Args:
        phase: The instantaneous phase of the base oscillator, ranging from 0 to
          1.0. This gives the position to lookup in the wavetable.
          Shape [batch_size, n_samples] .

        pulse_gains: a gain vector of dimension [N_pulse_gains, batch_size, n_samples]  that will be used to derive pulse gains
               by means of averaging over the pulse duration

            The first dimension holds the number of different gain factors and for each there will be generated a different
            pulse signal

        Returns:
        The resulting audio from linearly interpolated lookup of the wavetables at
          each point in time. Shape [N_pulses, batch_size, n_samples].
        """

        # For ease of linear interpolation we increase phase steps such that a step from one sample to the next
        # in the wavetable has value 1.
        phase_wt = phase * self.n_period
        phase_wt_quant = tf.floor(phase_wt)
        if self.no_interp:
            base_wt_inds = tf.cast(tf.round(phase_wt), tf.int32)

            # wavetable dimension is n_period x k (different versions with different band limits for
            # subsampling when high F0s are desired
            samples_quant = tf.gather(self.wavetables, base_wt_inds, axis=0, batch_dims=1)
        else:
            phase_rem = (phase_wt - phase_wt_quant)[:, :, tf.newaxis, tf.newaxis]
            base_wt_inds = tf.cast(phase_wt_quant, tf.int32)[:, :, tf.newaxis]
            next_wt_inds = base_wt_inds + 1
            samples_quant = tf.gather(self.wavetables, tf.concat((base_wt_inds, next_wt_inds), axis=2), axis=0,
                                      batch_dims=1)

            # print(self.wavetables.shape, tf.concat((base_wt_inds, base_wt_inds+1), axis=2).shape, samples_quant.shape, phase_rem.shape)
            samples_quant = tf.reduce_sum(samples_quant * tf.concat((1. - phase_rem, phase_rem), axis=2), axis=2)


        # pulse bounds is bool True at the end (last sample) of each pulse, pulse starts are forced at the last saple of each signal
        # the pulse ends position marers  are flattened over the batch
        flat_length = tf.shape(phase_wt)[0]*tf.shape(phase_wt)[1]
        pulse_bounds = tf.reshape(tf.concat((phase_wt[:,:-1] > phase_wt[:,1:], tf.ones((tf.shape(phase_wt)[0], 1), tf.bool)), axis=1),
                                  (flat_length,))

        # derive the indices of the pulse in sequence of the pulses over all batches
        # each pulse end increases the counter by one, and we shift the index changes one position by contatenaing a 0 at the start
        # the first pulse starts at the first position in the first entry
        pulse_gain_inds = tf.concat((tf.zeros(1, dtype=tf.int32),
                                     tf.cumsum(tf.cast(pulse_bounds[:-1], tf.int32), axis=0, exclusive=False)), axis=0)

        # calculate the start the positions in the flattened batch
        pulse_gain_nl = tf.range(flat_length, dtype=tf.int32)
        # generate indices at start/end of all pulses
        pulse_last_samp_pos = tf.boolean_mask(pulse_gain_nl, pulse_bounds, axis=0)
        # generate length of all pulses. The diff will not include the length of the first ulse so we add it explicitely
        # Note that given the construction of pulse_bounds all pulses including the last partial one are at least of length 1.
        pulse_lengths = tf.concat((pulse_last_samp_pos[:1]+1,
                                   pulse_last_samp_pos[1:] - pulse_last_samp_pos[:-1]), axis=0)

        # flatten all batches first dimension remains the different pulse gains
        c_gain = tf.cumsum(tf.reshape(pulse_inst_gains, (tf.shape(pulse_inst_gains)[0], -1)),
                           axis = -1, exclusive=False)
        c_gain_sums = tf.boolean_mask(c_gain, pulse_bounds, axis=1 )
        pulse_gains   = tf.concat((c_gain[:,pulse_last_samp_pos[0]-1:pulse_last_samp_pos[0]],
                                   c_gain_sums[:,1:] - c_gain_sums[:,:-1]), axis=1) / tf.cast(pulse_lengths, tf.float32)
        pulsed_gains = tf.gather(pulse_gains, pulse_gain_inds, axis=1, batch_dims=0)
        return samples_quant, tf.reshape(pulsed_gains, tf.shape(pulse_inst_gains))


    def get_config(self):
        config= super(PulseWaveTable, self).get_config()
        config.update(sample_rate=self.sample_rate)
        config.update(Oq=self.Oq)
        config.update(am=self.am)
        config.update(rta=self.rta)
        config.update(use_radiation=self.use_radidation)
        config.update(nominalF0=self.nominalF0)
        config.update(numF0InGrid=self.numF0InGrid)
        config.update(maxF0=self.maxF0)
        config.update(F0GridFactor=self.F0GridFactor)
        config.update(wt_oversampling=self.wt_oversampling)


