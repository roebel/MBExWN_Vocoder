# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
# Spectral Processing and PQMF
#
# -----------------------------
# Acknowledgements
#
# _design_prototype_filter and PQMF(layer) are extended versions of the
# corresponding functions  from
# https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/models/mb_melgan.py
# Copyright of the original version was:
# Copyright 2020 The Multi-band MelGAN Authors, Minh Nguyen (@dathudeptrai) and Tomoki Hayashi (@kan-bayashi)
#
# See the related code sections below for changes.
# -----------------------------
#

import numpy as np
import tensorflow as tf
import copy
from ...utils import nextpow2_val
import scipy.signal as ss
from typing import Union

from .preprocess import get_mel_filter, get_stft_window

def _design_prototype_filter(taps=62, cutoff_ratio=0.15, beta=9.0):
    """
    Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427

    This function is an extended version of design_prototype_filter from
    https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/models/mb_melgan.py
    - Copyright 2020 The Multi-band MelGAN Authors , Minh Nguyen (@dathudeptrai) and Tomoki Hayashi (@kan-bayashi)
    - Apache License, Version 2.0 (the "License")

    Changes: added implementation in tensorflow
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."


    if isinstance(cutoff_ratio, tf.Variable):
        # make initial filter
        omega_c = np.pi * cutoff_ratio
        h_i = tf.concat((tf.sin(omega_c * (np.arange(-(taps//2), 0))) / (np.pi * np.arange(-(taps//2), 0)),
                         tf.cos(tf.zeros(1, dtype=tf.float64)) * cutoff_ratio,
                         tf.sin(omega_c * (np.arange(1, taps//2+1))) / (np.pi * np.arange(1, taps//2+1))
                         ), axis=0)
        # apply kaiser window
        w = tf.signal.kaiser_window(taps + 1, beta, dtype=tf.float64)
        h = h_i * w
    else:
        # make initial filter
        omega_c = np.pi * cutoff_ratio
        with np.errstate(invalid="ignore"):
            h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
                np.pi * (np.arange(taps + 1) - 0.5 * taps)
            )
        # fix nan due to indeterminate form
        h_i[taps // 2] = np.cos(0) * cutoff_ratio

        # apply kaiser window
        w = ss.kaiser(taps + 1, beta)
        h = h_i * w

    return h


class TFPQMF(tf.keras.layers.Layer):
    """PQMF module."""

    def __init__(self, subbands : int, taps : int, cutoff_ratio : float, beta : Union[float, int],
                 dtype=tf.float32, max_band=None, name: str="pqmf",  do_synthesis: bool =True, **kwargs):
        """Initilize PQMF module.
        Slightly adapted from  https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/models/mb_melgan.py
        for theory see appendix A in https://arxiv.org/pdf/1909.01700.pdf

        For theory see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.18.2036&rep=rep1&type=pdf
        and http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.820.9304&rep=rep1&type=pdf
        Args:
            config (class): MultiBandMelGANGeneratorConfig

        This class is a modified version of design_prototype_filter from
        https://github.com/TensorSpeech/TensorFlowTTS/blob/master/tensorflow_tts/models/mb_melgan.py
         - Copyright 2020 The Multi-band MelGAN Authors , Minh Nguyen (@dathudeptrai) and Tomoki Hayashi (@kan-bayashi)
         - Apache License, Version 2.0 (the "License")

        Changes:
        - completed Keras layer interface (functions: build, compute_output_shape, and call; properties output_shape)
        - added interface for retrieving filter transfer function.
        """
        super().__init__( dtype=dtype, name=name,**kwargs)
        self.subbands =subbands
        self.taps = taps
        self.cutoff_ratio = cutoff_ratio
        self.beta = beta
        self.do_synthesis =  do_synthesis
        self.max_band =  max_band
        self.taps = taps

        self.used_subbands = subbands
        if max_band :
            self.used_subbands = max_band

        # define filter coefficient
        h_proto = _design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((self.subbands, len(h_proto)))
        h_synthesis = np.zeros((self.used_subbands, len(h_proto)))

        for k in range(self.subbands):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - (taps / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            if k<self.used_subbands:
                h_synthesis[k] = (
                    2
                    * h_proto
                    * np.cos(
                        (2 * k + 1)
                        * (np.pi / (2 * subbands))
                        * (np.arange(taps + 1) - (taps / 2))
                        - (-1) ** k * np.pi / 4
                    )
                )

        # [subbands, 1, taps + 1] == [filter_width, in_channels, out_channels]
        analysis_filter = np.expand_dims(h_analysis, 1)
        analysis_filter = np.transpose(analysis_filter, (2, 1, 0))

        synthesis_filter = np.expand_dims(h_synthesis, 0)
        synthesis_filter = np.transpose(synthesis_filter, (2, 1, 0))

        # filter for downsampling & upsampling
        updown_filter = np.zeros((subbands, subbands, subbands), dtype=np.float32)
        for k in range(subbands):
            updown_filter[0, k, k] = 1.0

        self.analysis_filter = analysis_filter.astype(np.float32)
        self.synthesis_filter = synthesis_filter.astype(np.float32)
        self.updown_filter = updown_filter.astype(np.float32)
        self._built_input_shape = None


    def lin2db(self, filter_spec):
        return 20. * tf.math.log(tf.abs(filter_spec)) / tf.math.log(tf.cast(10., filter_spec.dtype))

    def calculate_filter_spect_lin(self, filter_coef, fft_size):
        zp_filter_coef = tf.concat((filter_coef,
                                    tf.zeros((filter_coef.shape[0], fft_size - filter_coef.shape[1]),
                                             dtype=filter_coef.dtype)), axis=1)
        return tf.abs(tf.signal.fft(tf.cast(zp_filter_coef, dtype=tf.complex128)))


    @property
    def output_shape(self):
        if self._built_input_shape:
            return self.compute_output_shape(input_shape=self._built_input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (None if input_shape[1] is None else input_shape[1]*self.subbands, 1)

    def build(self, input_shape):
        self._built_input_shape = input_shape
        super().build(input_shape)


    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)],
    )
    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, T, 1).
        Returns:
            Tensor: Output tensor (B, T // subbands, subbands).
        """
        x = tf.pad(x, [[0, 0], [self.taps // 2, self.taps // 2], [0, 0]])
        x = tf.nn.conv1d(x, self.analysis_filter, stride=1, padding="VALID")
        x = tf.nn.conv1d(x, self.updown_filter, stride=self.subbands, padding="VALID")
        return x

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)],
    )
    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, T // subbands, subbands).
        Returns:
            Tensor: Output tensor (B, T, 1).
        """
        x = tf.nn.conv1d_transpose(
            x,
            self.updown_filter * self.subbands,
            strides=self.subbands,
            output_shape=(
                tf.shape(x)[0],
                tf.shape(x)[1] * self.subbands,
                self.subbands,
            ),
        )
        x = tf.pad(x[:,:, :self.used_subbands], [[0, 0], [self.taps // 2, self.taps // 2], [0, 0]])
        return tf.nn.conv1d(x, self.synthesis_filter, stride=1, padding="VALID")

    def call(self, inputs, **kwargs):
        if self.do_synthesis:
            return self.synthesis(inputs)
        else:
            return self.analysis(inputs)

    def get_config(self):
        config = super(TFPQMF, self).get_config()
        config.update(subbands= self.subbands)
        config.update(max_band= self.max_band)
        config.update(taps=self.taps)
        config.update(cutoff_ratio = self.cutoff_ratio)
        config.update(beta = self.beta)
        config.update(do_synthesis = self.do_synthesis)
        return config





class TFSpectProcessor(object):
    def __init__(self, STFT_config, srate, sync_fft_size=False):
        """
        TF preprocessor: generates time frequency representation of a signal and applies
        TF preporcessing and scaling
        Parameters
        ----------
        preproc_dict : dict
           configures preprocessing, should contain entries for
           win_len: int
           hop_size: int
           C : float
              scaling factor to be applied before sigmoid. If C<=0 the sigmoid compression
              will be replaced by means of a sqrt function applied individually to R and I channels without any scaling.

        """
        self.STFT_config = copy.deepcopy(STFT_config)
        self.srate = srate
        self.win_len = self.STFT_config['win_size']
        self.hop_size = self.STFT_config['hop_size']
        self.rel_lin_amp_off = self.STFT_config.get("rel_lin_amp_off", False)
        # if set the spectrum will not be converted to log but compressed by means of applying the parameter as exponent
        self.magnitude_compression = self.STFT_config.get("magnitude_compression", None)
        self.use_lin_amp_off_for_mc = self.STFT_config.get("use_lin_amp_off_for_mc", False)
        self.sync_fft_size = sync_fft_size
        if not hasattr(self.win_len, "__getitem__"):
            self.win_len = [self.win_len]
        self.win_len = [int(wl * srate) for wl in self.win_len]

        if not hasattr(self.hop_size, "__getitem__"):
            self.hop_size = [self.hop_size]
        self.hop_size = [int(hs * srate) for hs in self.hop_size]

        if "fft_size" in  self.STFT_config:
            raise RuntimeError("TFSpectProcessor::error::STFT config key fft_size is no longer suported, use fft_over now")
        elif "fft_over" in self.STFT_config:
            fft_over = self.STFT_config['fft_over']
            if not hasattr(fft_over, "__getitem__"):
                fft_over = [fft_over]
            if len(self.win_len) != len(fft_over):
                if len(fft_over) == 1:
                    fft_over *= len(self.win_len)
                else:
                    raise RuntimeError("TFSpectProcessor::error::if len(fft_over) > 1 then the number of entries for win_size and fft_over needs to be the same")
            self.fft_size = []
            for wl, fo in zip(self.win_len, fft_over):
                self.fft_size.append(nextpow2_val(wl)* (2 ** fo) )
        else:
            self.fft_size = []
            for wl in self.win_len:
                self.fft_size.append(nextpow2_val(wl))

        if len(self.win_len) != len(self.hop_size):
            raise RuntimeError("TFSpectProcessor::error::number of entries for win_size and hop_size needs to be the same")

        if sync_fft_size:
            max_fft_size = tf.reduce_max(self.fft_size)
            self.fft_size = [max_fft_size for _ in self.fft_size]
        self.log_2_db = 20 * np.log10(np.exp(1))
        self.lin_amp_off = 1e-5
        if ("lin_amp_off" in self.STFT_config) and (self.STFT_config["lin_amp_off"] is not None):
            self.lin_amp_off = STFT_config["lin_amp_off"]


    def generate_RIC_weights(self, n_sets, RIC_band_width, rseed=1):
        """
        generate a reproducible set of random weights for convolving with the RI spectrogram
        """
        r = np.random.RandomState(rseed)
        weights = []
        for fs in self.fft_size:
            weights.append(r.uniform(-1, 1, ( 1, int(fs * RIC_band_width/self.srate), 2, n_sets)))
        return weights


    def get_window(self, win_len, **kwargs):
        """
        get Hanning window that is normalized to sum to 1
        """
        ww = tf.signal.hann_window(win_len, periodic=False, dtype=tf.float32)
        return ww/tf.reduce_sum(ww)

    def generate_stft(self, tf_signal):
        """
        produce time frequency representation of input signal


        Parameters
        ----------
        tf_signal: tensorflow.Tensor


        Returns: returns list of into batch x time x freq representations
        -------
        """

        # returns into batch x time x freq representation
        specs = []
        for wl, hs, fft_size in zip(self.win_len, self.hop_size, self.fft_size):
            specs.append(tf.signal.stft(tf.pad(tf_signal, ((0,0),) * (len(tf_signal.shape)-1) + ((wl//2, wl//2 + hs + 1),), mode= "REFLECT"),
                                         wl, hs, fft_length=fft_size, window_fn=self.get_window, pad_end=False))
        return specs

    def scale_spec(self, spec):
        amp_spec = tf.abs(spec)
        if self.magnitude_compression is not None:
            if self.use_lin_amp_off_for_mc:
                if self.rel_lin_amp_off:
                    return tf.pow(amp_spec + tf.reduce_max(amp_spec, axis=[-2,-1], keepdims=True) * self.lin_amp_off, self.magnitude_compression)
                else:
                    return tf.pow(amp_spec + self.lin_amp_off, self.magnitude_compression)
            else:
                return tf.pow(amp_spec, self.magnitude_compression)
        elif self.rel_lin_amp_off:
            return self.log_2_db * tf.math.log(amp_spec + tf.reduce_max(amp_spec, axis=[-2,-1], keepdims=True) * self.lin_amp_off)
        else:
            return self.log_2_db * tf.math.log(amp_spec + self.lin_amp_off)

    def scale_spec_man_select(self, spec, magnitude_exponent=None):
        amp_spec = tf.abs(spec) + self.lin_amp_off
        if magnitude_exponent is not None:
            if magnitude_exponent == 1:
                return amp_spec
            elif magnitude_exponent==2:
                return tf.square(amp_spec)
            else:
                return tf.pow(amp_spec, magnitude_exponent)
        else:
            return self.log_2_db * tf.math.log(amp_spec)


    def __call__(self, tf_signal):
        """
        produce time frequency representation of input signal

        Parameters
        ----------
        tf_signal: tensorflow.Tensor


        Returns: tuple(array, float)
        -------
        """

        # returns into time x freq representation
        specs = [ ]
        for spec in zip(self.generate_stft(tf_signal)):
            specs.append(self.scale_spec(spec))

        return specs



class TFMelProcessor(object):
    def __init__(self, preprocess_config):
        """
        TF preprocessor: generates time frequency representation of a signal and applies
        TF preporcessing and scaling
        Parameters
        ----------
        preproc_dict : dict
           configures preprocessing, should contain entries for
           win_len: int
           hop_size: int
           C : float
              scaling factor to be applied before sigmoid. If C<=0 the sigmoid compression
              will be replaced by means of a sqrt function applied individually to R and I channels without any scaling.

        """
        self.preprocess_config = copy.deepcopy(preprocess_config)

        self.win_len = self.preprocess_config['win_size']
        self.sample_rate =  self.preprocess_config['sample_rate']
        self.mel_channels = self.preprocess_config['mel_channels']
        # number of intermediate steps that need to be derived
        self.hop_size = self.preprocess_config['hop_size']
        self.fft_size = self.preprocess_config['fft_size']
        self.lin_amp_off = 1e-5
        if ("lin_amp_off" in self.preprocess_config) and (self.preprocess_config["lin_amp_off"] is not None):
            self.lin_amp_off = preprocess_config["lin_amp_off"]
        self.lin_amp_scale = 1
        if ("lin_amp_scale" in preprocess_config) and (preprocess_config["lin_amp_scale"] != 1):
            self.lin_amp_scale = preprocess_config["lin_amp_scale"]
        mel_amp_scale = 1
        if ("mel_amp_scale" in preprocess_config) and (preprocess_config["mel_amp_scale"] != 1):
            mel_amp_scale = preprocess_config["mel_amp_scale"]
        self.mel_amp_scale = tf.convert_to_tensor(mel_amp_scale, dtype=tf.float32)

        self.stft_window = tf.convert_to_tensor(self.lin_amp_scale * get_stft_window(win_type="hann", win_len=self.win_len, dtype=np.float32))
        # determine sound segment length
        mel_basis = get_mel_filter(sr=self.sample_rate, n_fft=self.fft_size, n_mels = self.mel_channels,
                                   fmin = self.preprocess_config["fmin"], fmax = self.preprocess_config["fmax"],
                                   dtype = np.float32)
        self.tf_f_bank = tf.convert_to_tensor(mel_basis.T, dtype=tf.float32)
        self.sub_sampled = None
        self.stft_filter = None

    def get_window(self, *args, **kwargs):
        return self.stft_window

    def __call__(self, tf_signal):
        """
        produce time frequency representation of input signal


        Parameters
        ----------
        tf_signal: tensorflow.Tensor


        Returns: tuple(array, float)
        -------
        """

        # returns into time x freq representation
        temp = tf.abs(tf.signal.stft(tf.pad(tf_signal, ((0,0), (self.win_len//2, self.win_len//2 + self.hop_size + 1)),
                                            mode= "REFLECT"),
                                     self.win_len, self.hop_size, fft_length=self.fft_size,
                                     window_fn=self.get_window, pad_end=False))

        if self.stft_filter is not None:
            temp = temp * self.stft_filter
        melspec = tf.linalg.matmul(temp, self.tf_f_bank)
        mellspec = self.mel_amp_scale * tf.math.log(melspec + self.lin_amp_off)

        return mellspec
