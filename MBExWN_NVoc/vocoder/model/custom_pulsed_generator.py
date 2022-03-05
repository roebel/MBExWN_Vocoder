# coding: utf-8
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
# MBExWN layer

import os
import sys
import numpy as np
import tensorflow as tf
from typing import List, Union, Any, Sequence, Dict, Tuple
import copy

from .custom_layers import TFPad1d
from .tf2_components.layers.conv_layers import TF2C_Conv1DWeightNorm, TF2C_Conv1DUpDownSample
from .tf2_components.layers.support_layers import TF2C_LinInterpLayer as LinInterpLayer

from .custom_AE_layers import  ActivationLayer, WaveNetAEBlock
from .tf_preprocess import TFPQMF
from .tf_wavetable import PulseWaveTable
from .training_utils import ParamSchedule
from .tf2_components.layers.tf2c_base_layer import TF2C_BasePretrainableLayer

log_to_db = 20 * np.log10(np.exp(1))


def get_missing_upsampling_factor(target_ups, total_ups, base_name):
    up = target_ups // total_ups
    # print("subnet gen", target_ups, total_ups, int(target_ups / total_ups),target_ups // total_ups)
    if total_ups * up != target_ups:
        raise RuntimeError(f"get_missing_upsamling_factor::error:: Upsampling to target "
                           f"upsampling factor {target_ups} from {total_ups} is not possible for subnet {base_name}")
    return up


def generate_subnet_from_specs(specs, base_name, activation,
                               final_n_channels, final_nks, final_activation, weight_init_scale=0.02,
                               target_ups=None, force_causal =False,
                               pad_to_valid=False, remove_inactive_pad_layers=False,
                               use_tf25_compatible_implementation=False,
                               **activation_kwargs):
    # For the moment the Pulse parameter layer will output the F0 values only, over time there may be a pulse shape parameter as well.
    total_ups = 1
    layers = []
    if use_tf25_compatible_implementation:
        Conv1D_Layer = TF2C_Conv1DWeightNorm
        Conv1DUpDown_Layer = TF2C_Conv1DUpDownSample
    else:
        raise NotImplementedError("generate_subnet_from_specs::error::implmentations not selecting use_tf25_compatible_implementation are not supported")

    default_padding = "CAUSAL" if force_causal else "SAME"

    if specs:
        for ii, spec in enumerate(specs):
            if spec[0] == "L":
                up= spec[1]
                layers.append(LinInterpLayer(upsampling_factor=up, num_pad_end=1, drop_last=True,
                                                  name=base_name + f"_LinUpLayer_{ii}",))
            else:
                ks = spec[0]
                nf = spec[1]
                linear_up= False
                up = 1
                if len(spec) > 2:
                    if isinstance(spec[2], str):
                        if spec[2][0] == "L":
                            linear_up = True
                        up = int(spec[2][1:])
                    else:
                        up = spec[2]

                if linear_up:
                    if (not remove_inactive_pad_layers) or ((ks - 1) // 2 + ((ks-1) % 2)) > 0:
                        if force_causal:
                            layers.append(TFPad1d(padding_size=((ks - 1) // 2 + ((ks - 1) % 2) + (ks - 1) // 2, 0),
                                                  padding_type="EDGE" if pad_to_valid else "SYMMETRIC",
                                                  name=base_name + f"_Pad_{ii}"))
                        else:
                            layers.append(TFPad1d(padding_size=((ks - 1) // 2 + ((ks-1) % 2), (ks - 1) // 2),
                                              padding_type="EDGE" if pad_to_valid else "SYMMETRIC", name=base_name + f"_Pad_{ii}"))
                    layers.append(Conv1D_Layer(nf, kernel_size=ks,
                                                       padding="VALID", use_weight_norm=True,
                                                       name=base_name + f"_Layer_{ii}",
                                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,
                                                                                                             stddev=weight_init_scale)))
                    layers.append(LinInterpLayer(upsampling_factor=up, num_pad_end=1, drop_last=True,
                                                 name=base_name + f"_LinUpLayer_{ii}"))
                else:
                    if up > 1:
                        if pad_to_valid and (((ks - 1) // 2 + ((ks-1) % 2)) > 0):
                            if force_causal:
                                layers.append(TFPad1d(padding_size=((ks - 1) // 2 + ((ks - 1) % 2) + (ks - 1) // 2, 0),
                                                      padding_type="EDGE", name=base_name + f"_Pad_{ii}"))
                            else:
                                layers.append(TFPad1d(padding_size=((ks - 1) // 2 + ((ks-1) % 2), (ks - 1) // 2),
                                                  padding_type="EDGE", name=base_name + f"_Pad_{ii}"))
                        layers.append(Conv1DUpDown_Layer(nf, kernel_size=ks,
                                                         padding="VALID" if pad_to_valid else default_padding,
                                                         use_weight_norm=True,
                                                         name=base_name + f"_Layer_{ii}",
                                                         factor=up,
                                                         up_sample=True,
                                                         use_checkerboard_free_init=True,
                                                         kernel_initializer=tf.keras.initializers.RandomNormal(
                                                             mean=0,
                                                             stddev=weight_init_scale)))
                    else:
                        if (not remove_inactive_pad_layers) or ((ks - 1) // 2 + ((ks - 1) % 2)) > 0:
                            if force_causal:
                                layers.append(TFPad1d(padding_size=((ks - 1) // 2 + ((ks-1) % 2) + (ks - 1) // 2, 0),
                                                      padding_type="EDGE" if pad_to_valid else "SYMMETRIC", name=base_name + f"_Pad_{ii}"))
                            else:
                                layers.append(TFPad1d(padding_size=((ks - 1) // 2 + ((ks-1) % 2), (ks - 1) // 2),
                                                      padding_type="EDGE" if pad_to_valid else "SYMMETRIC", name=base_name + f"_Pad_{ii}"))

                        layers.append(Conv1D_Layer(nf, kernel_size=ks,
                                                       padding="VALID", use_weight_norm=True,
                                                       name=base_name + f"_Layer_{ii}",
                                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,
                                                                                                             stddev=weight_init_scale)))
                layers.append(activation(**activation_kwargs, name=base_name + f"_ActLayer_{ii}"))
                total_ups *= up

        if final_nks is not None:
            if pad_to_valid and (((final_nks - 1) // 2 + ((final_nks - 1) % 2)) > 0) :
                if force_causal:
                    layers.append(TFPad1d(padding_size=((final_nks - 1) // 2 + ((final_nks - 1) % 2) + (final_nks - 1) // 2, 0),
                                          padding_type="EDGE", name=base_name + f"_Pad_{ii}"))
                else:
                    layers.append(TFPad1d(padding_size=((final_nks - 1) // 2 + ((final_nks - 1) % 2), (final_nks - 1) // 2),
                                          padding_type="EDGE", name=base_name + f"_Pad_{ii}"))
            layers.append(Conv1D_Layer(final_n_channels, kernel_size=final_nks,
                                           padding="VALID" if pad_to_valid else default_padding,
                                           use_weight_norm=True, name=base_name + "_Layer_final",
                                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,
                                                                                                 stddev=weight_init_scale)))

            if (target_ups is not None) and total_ups != target_ups:
                up = get_missing_upsampling_factor(target_ups=target_ups, total_ups=total_ups, base_name=base_name)
                layers.append(LinInterpLayer(upsampling_factor=up,
                                                  num_pad_end=1, drop_last=True, name=base_name + "_linear_interp"))
                total_ups *= up
            if len(layers) and final_activation is not None:
                layers.append(ActivationLayer(activation_function=final_activation, name=base_name + "_Layer_finalAct"))

    return layers, total_ups


class MBExWN(TF2C_BasePretrainableLayer):
    """Synthesize audio from a series of wavetables."""


    def __init__(self,
                 preprocess_config,
                 # pulse frequency
                 pp_subnet : Union[List[Union[Tuple[int, int], Tuple[int, int, int]]], None],
                 # pulse filter
                 ps_subnet : List[Union[Tuple[int, int], Tuple[int, int, int]]],

                 # a wavenet that is used for pulse modification
                 pp_mod_subnet: Dict,
                 # upsamling ratios of the residual blocks
                 pp_mod_subnet_upsampling_factors : List[int],
                 # channel factors of the residual blocks
                 pp_mod_subnet_channel_factors: List[int],
                 # multi band
                 multi_band_config: Union[None, Dict],

                 # pulse frequency
                 pp_min_frequency: Union[float, int] = 40.,
                 pp_max_frequency: Union[float, int] = 600.,
                 pp_teacher_forcing_schedule: Union[Dict, None] = None,
                 pp_F0_pred_loss_limits_ms: float = 0.,
                 pp_F0_rec_loss_limits_ms: float = 0.,
                 pp_activation: str = "soft_sigmoid",
                 pp_F0_loss_weight: Union[Dict, None] = None,
                 pp_F0_loss_method: str = "L1",
                 # F0 loss weight for unvoiced segments
                 pp_F0_UV_loss_weight: Union[float, None] = None,
                 pp_mod_subnet_noise_channel_sigma: float = 0.5,
                 pp_mod_subnet_use_pqmf : bool = True,
                 pp_subnet_use_valid_padding: bool = False,
                 pp_subnet_training_only: bool = False,
                 pp_subnet_exclude_from_pretrain: bool = False,
                 pp_subnet_suppress_uv_gradient : bool = False,
                 # pulse filter spectrum
                 ps_max_ceps_coefs : int = 120,
                 ps_env_order_scale: Union[float, None] = None,
                 ps_subnet_use_valid_padding: bool = False,
                 ps_use_stft : bool = True,
                 ps_off : bool = False,
                 # gain limits for ps and ns filters
                 filter_max_db_range: Union[float, None] = None,

                 # if not set to None or 0 all filters will have mean log amplitude == 0
                 # if set to a value > 0 pulse and noise filters will be penalized if they affect the soure signal energy
                 # which is supposed to be controlled by the sources gain.
                 psns_gain_loss_weight : Union[float, None] = None,
                 #
                 psns_use_cepstral_loss_constraint : bool =False,
                 psns_cepstral_loss_weight: Union[float, None] = 0.5,
                 spect_filters_preserve_energy : bool = False,
                 stft_coh_loss_weight: Union[float, None, Dict] = None,
                 remove_inactive_pad_layers: bool = False,
                 use_prelu : bool = True,
                 pulse_rate_factor : int = 2,
                 pulse_channels : int = 8,
                 pulse_channels_use_pqmf : bool = False,
                 pulse_channels_multi_band_config: Union[None, Dict] = None,

                 # force all CNN layers to use causal padding
                 # which in principle will allow real time applications with small latency
                 # equal to a single STFT frame. Note that this would require a dedicated implmentation
                 # of the convolution op
                 force_causal: bool = False,
                 wavetable_config : Dict = None,
                 alpha : float = 0.2,
                 dump_controls = False,
                 # noise foor to be added to pulse signal to avoid creating spectral domain value close to
                 # magnitude zero which will create NaN in the gradient calculations. The noise floor will
                 # be used only during training and can be disabled by means of selecting None
                 pulse_noise_floor_db: Union[float, None] = -90,
                 internal_win_size_s: Union[float, None] = None,
                 internal_fft_over: int = 0,

                 name : str ='MBExWNGen',
                 use_tf25_compatible_implementation: bool = False,
                 quiet : bool = False):

        super().__init__(name=name)

        self.preprocess_config = copy.deepcopy(preprocess_config)
        self.sample_rate = preprocess_config["sample_rate"]
        self.spect_hop_size = preprocess_config["hop_size"]
        self.mel_channels =  preprocess_config["mel_channels"]

        self.F0_loss = None
        self.PS_gain_loss = None
        self.PS_cepstral_loss = None
        self.stft_coh_loss = None

        self.force_causal = force_causal
        self.use_prelu = use_prelu
        self.alpha = alpha
        if use_prelu:
            activation = tf.keras.layers.PReLU
            activation_kwargs = {"alpha_initializer": tf.keras.initializers.Constant(self.alpha),
                                 "shared_axes": [1]}
        else:
            activation = tf.keras.layers.LeakyReLU
            activation_kwargs = {"alpha": self.alpha}
        self.remove_inactive_pad_layers = remove_inactive_pad_layers

        # internal sample rates.
        self.multi_band_config = copy.deepcopy(multi_band_config)
        self.mb_factor = self.multi_band_config["subbands"]
        self.pulse_rate_factor = pulse_rate_factor
        self.pulse_rate = self.sample_rate/self.pulse_rate_factor
        self.pulse_channels = pulse_channels
        self.pp_mod_subnet_use_pqmf = pp_mod_subnet_use_pqmf
        self.pulse_channels_use_pqmf = pulse_channels_use_pqmf
        self.pulse_channels_multi_band_config = pulse_channels_multi_band_config
        self.spect_to_subband_upsampling_factor = self.spect_hop_size // self.mb_factor
        self.spect_to_pulse_upsampling_factor = (self.spect_to_subband_upsampling_factor * pulse_channels)//np.prod(pp_mod_subnet_upsampling_factors)
        self.F0_down_sampling_factor = int(self.sample_rate // self.pulse_rate)

        if use_tf25_compatible_implementation:
            Conv1D_Layer = TF2C_Conv1DWeightNorm
        else:
            raise NotImplementedError(
                "MBExWN::error::implmentations not selecting use_tf25_compatible_implementation are not supported")

        # generator for pulse wavetable parameters

        self.pp_subnet = copy.deepcopy(pp_subnet)
        self.pp_subnet_layers = []
        self.pp_layers = []
        pp_subnet_layers = None
        self.pp_min_frequency = pp_min_frequency
        self.pp_max_frequency = pp_max_frequency
        self.pp_activation = pp_activation
        self.pp_F0_loss_method = pp_F0_loss_method
        self.pp_F0_UV_loss_weight = pp_F0_UV_loss_weight
        self.pp_subnet_training_only = pp_subnet_training_only
        self.pp_subnet_exclude_from_pretrain = pp_subnet_exclude_from_pretrain

        self.pp_subnet_suppress_uv_gradient = pp_subnet_suppress_uv_gradient
        if pp_F0_loss_weight is None:
            self.pp_F0_loss_weight = None
        else:
            self.pp_F0_loss_weight = ParamSchedule(name="pp_F0_loss_weight", quiet=quiet, **pp_F0_loss_weight)

        self.pp_F0_pred_loss_limits_ms = pp_F0_pred_loss_limits_ms
        self.pp_F0_rec_loss_limits_ms = pp_F0_rec_loss_limits_ms

        self.unvoiced_right_extender_kernel = tf.ones((int((pp_F0_pred_loss_limits_ms * self.pulse_rate ) // 1000) + 1, 1 , 1))
        self.unvoiced_left_extender_kernel = self.unvoiced_right_extender_kernel
        if pp_F0_rec_loss_limits_ms>= 0:
            self.voiced_extender_kernel = tf.ones((int((pp_F0_rec_loss_limits_ms * self.pulse_rate ) // 1000) + 1, 1 , 1))




        # For the moment the Pulse parameter layer will output the F0 values only, over time there may be a pulse shape parameter as well.
        self.pp_subnet_use_valid_padding = pp_subnet_use_valid_padding
        if self.pp_subnet:
            self.pp_subnet_layers, _ = generate_subnet_from_specs(
                self.pp_subnet,
                base_name="PulsPar",
                activation=activation,
                final_n_channels=1,
                final_nks=1,
                final_activation=pp_activation,
                force_causal =self.force_causal,
                pad_to_valid=self.pp_subnet_use_valid_padding,
                target_ups=self.spect_to_pulse_upsampling_factor,
                remove_inactive_pad_layers= self.remove_inactive_pad_layers,
                use_tf25_compatible_implementation=use_tf25_compatible_implementation,
                **activation_kwargs
            )


        # here we define all the subnets that are necessary for the full MBExWN
        if not self.pp_subnet_training_only:

            self.wavetable_config = copy.deepcopy(wavetable_config)

            self.pp_mod_subnet_upsampling_factors = copy.deepcopy(pp_mod_subnet_upsampling_factors)
            self.pp_mod_subnet_channel_factors = pp_mod_subnet_channel_factors


            # internal rates
            # sample_rate= SR
            # pulse_rate = PR = SR / pulse_rate_factor
            # WN_in_rate = PR / pulse_channels
            # WN_out_rate = PR / pulse_channels * np.prod(pp_mod_subnet_upsampling_factors)
            # SR = WN_out_rate * mb_factor
            # spect_rate = SR / spect_hop_size
            # max_cond_rate = SR / mb_factor
            # first_cond_rate = SR / (mb_factor * np.prod(pp_mod_subnet_upsampling_factors[:-1))
            # spect_to_f0_uprate =
            if self.pulse_rate/pulse_channels * np.prod(pp_mod_subnet_upsampling_factors) * self.mb_factor != self.sample_rate:
                raise RuntimeError(
                    f"MBExWN::config_error::the generated "
                    f"sample rate {self.pulse_rate / pulse_channels * np.prod(pp_mod_subnet_upsampling_factors) * self.mb_factor} "
                    f"!= {self.sample_rate}\n"
                    f"pulse_rate:{self.pulse_rate}, "
                    f"WN_in_rate:{self.pulse_rate/pulse_channels}, "
                    f"WN_out_rate:{self.pulse_rate / pulse_channels* np.prod(pp_mod_subnet_upsampling_factors)}, "
                    f"generator_rate:{self.pulse_rate / pulse_channels * np.prod(pp_mod_subnet_upsampling_factors) * self.mb_factor}"
                )

            self.pulse_generator = PulseWaveTable(**wavetable_config, sample_rate=self.pulse_rate, quiet=quiet)

            if pp_teacher_forcing_schedule is None:
                self.pp_teacher_forcing_schedule = None
            else:
                self.pp_teacher_forcing_schedule = ParamSchedule(name="pp_teacher_forcing_schedule", quiet=quiet, **pp_teacher_forcing_schedule)


            # generator for pulse filter spectrum
            self.ps_subnet = copy.deepcopy(ps_subnet)
            self.ps_subnet_layers = []
            self.ps_max_ceps_coefs  = ps_max_ceps_coefs
            self.ps_env_order_scale = ps_env_order_scale
            self.ps_use_stft = ps_use_stft
            self.ps_off = ps_off
            self.filter_max_db_range = filter_max_db_range
            self.filter_max_log_range = filter_max_db_range / log_to_db if filter_max_db_range is not None else None


            self.dump_controls = dump_controls
            self.pulse_noise_floor_db = pulse_noise_floor_db
            self.pulse_noise_floor_mag = None
            if pulse_noise_floor_db is not None:
                # ensure negative threshold value in case parameter is missunderstood
                self.pulse_noise_floor_db = -np.abs(pulse_noise_floor_db)
                self.pulse_noise_floor_mag = 10 ** (self.pulse_noise_floor_db / 20)

            self.psns_gain_loss_weight = psns_gain_loss_weight
            self.psns_use_cepstral_loss_constraint = psns_use_cepstral_loss_constraint
            self.psns_cepstral_loss_weight = psns_cepstral_loss_weight
            self.spect_filters_preserve_energy = spect_filters_preserve_energy
            self.stft_coh_loss_weight = stft_coh_loss_weight

            self.stft_win = tf.signal.hann_window


            self.internal_win_size_s = internal_win_size_s
            self.internal_fft_over = internal_fft_over
            if self.internal_win_size_s :
                self.stft_win_size = int(self.internal_win_size_s * self.sample_rate)
            else:
                self.stft_win_size = 4 * self.spect_hop_size
            fft_size = 16
            while fft_size < self.stft_win_size:
                fft_size *= 2
            self.fft_size = fft_size * (2 ** self.internal_fft_over )


            # this kernel is used to smooth the f0 controu for selection of the cepstral envelope windows
            # we want the window without zeros at the ends
            smooth_win = np.bartlett(2 * self.spect_hop_size + 3)[1:-1, np.newaxis, np.newaxis]
            self.frequency_smoothing_kernel = tf.constant(smooth_win / np.sum(smooth_win), dtype=tf.float32)

            pp_mod_subnet = copy.deepcopy(pp_mod_subnet)
            self.pp_mod_subnet = copy.deepcopy(pp_mod_subnet)
            self.pp_mod_subnet_noise_channel_sigma = pp_mod_subnet_noise_channel_sigma

            if not ps_off:

                self.ps_cepstral_windows = []
                self.ps_subnet_use_valid_padding = ps_subnet_use_valid_padding
                self.ps_subnet_layers, ps_total_ups = generate_subnet_from_specs(
                    self.ps_subnet,
                    base_name="PS",
                    activation=activation,
                    final_nks=1,
                    final_n_channels=ps_max_ceps_coefs if self.ps_use_stft else multi_band_config["subbands"] ,
                    final_activation=None,
                    pad_to_valid=self.ps_subnet_use_valid_padding,
                    force_causal =self.force_causal,
                    remove_inactive_pad_layers=self.remove_inactive_pad_layers,
                    weight_init_scale = 0.01,
                    use_tf25_compatible_implementation=use_tf25_compatible_implementation,
                    **activation_kwargs
                )

                if self.ps_use_stft:
                    self.ps_gain_interpolator = None

                    if ps_env_order_scale:
                        cepstral_windows = []
                        cepstral_windows_log10f0 = []

                        for f0 in np.logspace(np.log10(pp_min_frequency), np.log10(pp_max_frequency), 30):
                            win_len = int(ps_env_order_scale * 0.5 * self.sample_rate / f0)
                            if (win_len // 2) * 2 == win_len:
                                win_len += 1
                            cepstral_windows_log10f0.append(np.log10(f0))
                            if win_len // 2 + 1 > ps_max_ceps_coefs:
                                cepstral_windows.append(np.hamming(win_len)[win_len // 2:][:ps_max_ceps_coefs])
                            else:
                                cepstral_windows.append(np.concatenate((np.hamming(win_len)[win_len // 2:],
                                                                        np.zeros(ps_max_ceps_coefs - 1 - (win_len // 2))),
                                                                       axis=0))
                        self.ps_cepstral_windows_log10f0 = tf.constant(cepstral_windows_log10f0, dtype=tf.float32)
                        self.ps_cepstral_windows = tf.constant(cepstral_windows, dtype=tf.float32)

                else:
                    self.ps_gain_interpolator = LinInterpLayer(upsampling_factor=self.spect_hop_size, num_pad_end=1)


            self.pp_waveNetBlocks = []
            self.wn_post_net = []

            self.pp_mod_subnet_num_channels = pp_mod_subnet.pop("n_channels")
            self.pp_mod_subnet_cond_lin_upsampling = pp_mod_subnet.pop("cond_lin_upsampling", 16)
            self.pp_mod_subnet_cond_kernel_size = pp_mod_subnet.pop("cond_kernel_size", 3)
            curr_pulse_rate = self.pulse_rate/self.pulse_channels
            spect_rate  = self.sample_rate /self.spect_hop_size
            for iwn, (ups, chan_fac) in enumerate(zip(self.pp_mod_subnet_upsampling_factors, self.pp_mod_subnet_channel_factors)):
                if not quiet:
                    print(f"configure wavenet block: in pulse rate folded {curr_pulse_rate}, spect_rate={spect_rate} "
                          f"cond_conv_upsampling={ curr_pulse_rate / (spect_rate * self.pp_mod_subnet_cond_lin_upsampling)} "
                          f"cond_lin_upsampling={self.pp_mod_subnet_cond_lin_upsampling}", file=sys.stderr)
                if curr_pulse_rate != (curr_pulse_rate//(spect_rate*self.pp_mod_subnet_cond_lin_upsampling))*spect_rate*self.pp_mod_subnet_cond_lin_upsampling:
                    raise RuntimeError(f"MBExWN::config_error:: cannot achieve conditioning rate {curr_pulse_rate} by "
                                       f"means of integer usampling of spectrum rate {spect_rate} with "
                                       f"linear up {self.pp_mod_subnet_cond_lin_upsampling}")

                if self.force_causal:
                    pp_mod_subnet["padding"] = "CAUSAL"

                self.pp_waveNetBlocks.append(
                    WaveNetAEBlock(**pp_mod_subnet,
                                   n_channels=int(self.pp_mod_subnet_num_channels * chan_fac),
                                   dtype=self.dtype,
                                   up_sample=None if ups <= 1 else True,
                                   up_down_factor=ups,
                                   cond_kernel_size = self.pp_mod_subnet_cond_kernel_size,
                                   cond_conv_upsampling = int(curr_pulse_rate // (spect_rate * self.pp_mod_subnet_cond_lin_upsampling)),
                                   cond_lin_upsampling = self.pp_mod_subnet_cond_lin_upsampling,
                                   use_tf25_compatible_implementation=use_tf25_compatible_implementation,
                                   name="PP_waveNetBlock_ups{}_{}".format(ups, iwn)))
                curr_pulse_rate *= ups

            self.wn_post_net = [
                Conv1D_Layer(self.mb_factor, kernel_size=1, use_weight_norm=True,
                                 name=self.name+"_PaNMPulseWaveNet_Post"),
            ]

            self.pqmf = None
            if self.pp_mod_subnet_use_pqmf:
                self.pqmf = TFPQMF(**multi_band_config, do_synthesis=True, name="PQMFilterBank")

            self.pulse_pqmf = None
            if self.pulse_channels_use_pqmf:
                self.pulse_pqmf = TFPQMF(**pulse_channels_multi_band_config, do_synthesis=False, name="PC_PQMFilterBank")

        self.log_to_log10 = 1 / np.log(10)
        self._gain_layer_output = 0


    def _get_cepstral_windows(self, f0, cepstral_windows_log10f0, cepstral_windows, smooth_stride):
        # print(f0.shape, self.frequency_smoothing_kernel.shape, cepstral_windows.shape)
        f0_padded = tf.concat((tf.tile(f0[:, :1], (1, self.frequency_smoothing_kernel.shape[0] // 2)),
                               f0, tf.tile(f0[:, -1:], (1, self.frequency_smoothing_kernel.shape[0] // 2))),
                              axis=1)

        smooth_log10f0 = tf.minimum(tf.maximum(self.log_to_log10 *
                                               tf.math.log(tf.nn.conv1d(f0_padded[:,:,tf.newaxis], self.frequency_smoothing_kernel,
                                                   stride=smooth_stride,
                                                   dilations=1, padding="VALID")),
                                               cepstral_windows_log10f0[0]),
                                    cepstral_windows_log10f0[-1])[:, :, 0]

        ratio = (smooth_log10f0 - cepstral_windows_log10f0[0]) / (cepstral_windows_log10f0[-1] - cepstral_windows_log10f0[0])

        csmooth_index = tf.cast(tf.round(ratio * (cepstral_windows_log10f0.shape[0] - 1)), tf.int32)
        smooth_windows = tf.gather(cepstral_windows, csmooth_index, axis=0, batch_dims=0)
        #print(smooth_windows.shape,smooth_windows.dtype, csmooth_index.shape)
        return tf.stop_gradient(smooth_windows)


    def get_F0_pred_loss_mask (self, target_F0):
        """
        Generate a boolean mask covering only the segments of the batch to be taken into account for the F0 loss.

        """

        unvoiced_mask = tf.cast(target_F0 == 0, tf.float32)
        unvoiced_mask_left_extended = tf.nn.conv1d(tf.pad(unvoiced_mask, ((0,0), (self.unvoiced_left_extender_kernel.shape[0]-1, 0)))[:,:,tf.newaxis],
                                                  filters= self.unvoiced_left_extender_kernel, stride=1, dilations=1, padding='VALID')
        unvoiced_mask_extended = tf.nn.conv1d(tf.pad(unvoiced_mask_left_extended, ((0,0), (0,self.unvoiced_right_extender_kernel.shape[0]-1), (0,0))),
                                             filters= self.unvoiced_right_extender_kernel, stride=1, dilations=1, padding='VALID')
        return tf.cast(unvoiced_mask_extended == 0, tf.float32)[:,:,0]

    def get_F0_rec_loss_mask (self, target_F0):
        """
        Generate a boolean mask covering extended voiced segments

        """

        voiced_mask = tf.cast(target_F0 != 0, tf.float32)
        voiced_mask_left_extended = tf.nn.conv1d(tf.pad(voiced_mask, ((0,0), (self.voiced_extender_kernel.shape[0]-1, 0)))[:,:,tf.newaxis],
                                                  filters= self.voiced_extender_kernel, stride=1, dilations=1, padding='VALID')
        voiced_mask_extended = tf.nn.conv1d(tf.pad(voiced_mask_left_extended, ((0,0), (0,self.voiced_extender_kernel.shape[0]-1), (0,0))),
                                             filters= self.voiced_extender_kernel, stride=1, dilations=1, padding='VALID')
        return tf.cast(voiced_mask_extended != 0, tf.float32)[:,:,0]



    def call(self, input, F0=None, training=False, return_PP=False, return_components=False, test_grad = None,
             **kwargs) -> tf.Tensor:
        """
        input are batched mel-frequency spectrograms
        """


        # PulseParameterGenerator
        mel = input
        batch_size = mel.shape[0]

        pulse_frequency = self.generate_f0(mel)
        padded_excitation_signal = None
        excitation_signal = None
        source_filter_stft = None
        self.PS_cepstral_loss = None
        self.PS_gain_loss= None

        if training:
            step = tf.summary.experimental.get_step()
            F0 = F0[:, ::self.F0_down_sampling_factor]
            # prediction loss mask extends unvoiced segments into voiced segments to retain only thos segments
            # that are very likely to be vocied
            pred_loss_F0_mask = self.get_F0_pred_loss_mask(F0[:, :, 0])

            # reconstruction loss mask covers segments that adapts F0 with other losses than F0 prediction loss
            if self.pp_F0_rec_loss_limits_ms >=0:
                # reconstruction loss mask extends voiced segments into unvoiced segments to hopefully covr all those segments
                # that contain glottal pulses
                rec_loss_F0_mask = self.get_F0_rec_loss_mask(F0[:, :, 0])
            else:
                rec_loss_F0_mask = self.get_F0_rec_loss_mask(F0[:, :, 0])

            if (self.pp_F0_loss_weight is not None) :
                diff_F0 = tf.maximum(F0[:,:,0], self.pp_min_frequency) - pulse_frequency[:,:F0.shape[1]]

                # do consider prediction loss only in the inner parts of the voiced segments
                # pp_F0_UV_loss_weight can be None
                if self.pp_F0_UV_loss_weight:
                    prediction_loss_mask = pred_loss_F0_mask + (1 - rec_loss_F0_mask)
                    diff_F0 = diff_F0 * prediction_loss_mask
                    F0_prediction_weight = tf.maximum(pred_loss_F0_mask, self.pp_F0_UV_loss_weight*prediction_loss_mask)
                    max_lim = 0.
                else:
                    # strangly binary masking will lead to NaN in Adam  even if division by 0 is prevented using epsilon
                    #F0_prediction_weight = tf.maximum(pred_loss_F0_mask, prediction_loss_mask/(F0.shape[0]*F0.shape[1]))
                    F0_prediction_weight = pred_loss_F0_mask
                    max_lim =1.
                # calculate the weighted loss
                if self.pp_F0_loss_method in ["L2", "l2", "least_squares"]:
                    self.F0_loss = tf.reduce_sum(tf.square(diff_F0 * F0_prediction_weight))/tf.maximum(tf.reduce_sum(F0_prediction_weight),
                                                                                                       max_lim)
                elif self.pp_F0_loss_method in ["L1", "l1"]:
                    self.F0_loss = tf.reduce_sum(tf.abs(diff_F0 * F0_prediction_weight))/tf.maximum(tf.reduce_sum(F0_prediction_weight),
                                                                                                       max_lim)
                elif self.pp_F0_loss_method in ["RMSE", "rmse"]:
                    self.F0_loss = tf.sqrt(tf.reduce_sum(tf.square(diff_F0) * F0_prediction_weight)/tf.maximum(tf.reduce_sum(F0_prediction_weight),
                                                                                                               max_lim))

                # tf.print(tf.reduce_mean (F0_prediction_weight), self.F0_loss)
                # if tf.reduce_sum(F0_prediction_weight) == 0:
                #     import matplotlib.pyplot as plt
                #     plt.figure()
                #     plt.clf()
                #
                #     plt.plot(tf.cast(F0[:,:,0]> 0, tf.float32) . numpy()[0,:]*100, "b", linewidth=3)
                #     plt.plot(F0_prediction_weight.numpy()[0, :]*100, "m", linewidth=2)
                #     plt.plot(F0[:,:,0].numpy()[0, :], "r", linewidth=1)
                #     plt.plot( pulse_frequency[:,:F0.shape[1]].numpy()[0, :], "g", linewidth=1)
                #     plt.grid()
                #     plt.show()

        if self.pp_subnet_training_only:
            if return_PP:
                returned_PP = [["F0", pulse_frequency]]
            else:
                returned_PP = []

            returned_signals= [tf.zeros(shape=(pulse_frequency.shape[0], pulse_frequency.shape[1]*self.pulse_rate_factor)), ]
            return returned_signals, returned_PP

        # F0 handling during training
        #
        # F0 prediction loss is used only for those segments that are relatively certainly voiced and where we can be
        # rather sure that the externally predicted F0 is correct and well suited for reconstruction
        # F0 reconstruction loss is used everywhere else
        # teacher forcing is applied only for the segments that are used for prediction loss

        if training and (self.pp_teacher_forcing_schedule is not None) and (F0 is not None):
            # assemble the F0 contour to be used for teacher forcing
            # For the strongly voiced segments that are evaluated in the F0 prediction loss we use the target F0
            # for the other segments we use the predicted F0 (so this finally means in those segments teacher
            # forcing is not used)
            extF0 = F0[:, :, 0] * pred_loss_F0_mask  + (1 - pred_loss_F0_mask) * pulse_frequency[:,:F0.shape[1]]
            extF0 = tf.concat((extF0[:, :], extF0[:,-1:] * tf.ones((extF0.shape[0], pulse_frequency.shape[1] - extF0.shape[1]))),
                              axis = 1)
            teacher_weight = self.pp_teacher_forcing_schedule(step)
            pulse_frequency_ = pulse_frequency * (1 - teacher_weight) +  extF0 * teacher_weight
            if self.pp_subnet_suppress_uv_gradient :
                rec_loss_F0_mask_ext = tf.concat((rec_loss_F0_mask,
                                                  tf.zeros((rec_loss_F0_mask.shape[0],
                                                            pulse_frequency_.shape[1]- rec_loss_F0_mask.shape[1]),
                                                           dtype=rec_loss_F0_mask.dtype)),
                                                  axis = 1)
                pulse_frequency_ = rec_loss_F0_mask_ext * pulse_frequency_ + tf.stop_gradient( (1 - rec_loss_F0_mask_ext) * pulse_frequency_)


        else:
            pulse_frequency_ = pulse_frequency

        if (not self.ps_use_stft) or self.ps_off:
            # PulseFilterSpectrumGenerator
            if not self.ps_off:
                multi_band_gain = self.generate_multiband_gain(mel=mel, training=training)
                multi_band_interpolated_gain = self.ps_gain_interpolator(multi_band_gain)
            else:
                multi_band_interpolated_gain = None
            signal = self.generate_excitation(mel, pulse_frequency=pulse_frequency_,
                                                     mb_gain=multi_band_interpolated_gain)
        else:
            excitation_signal = self.generate_excitation(mel, pulse_frequency=pulse_frequency_)


        # print(f"source wn post netshape {source_.shape}")
        if self.ps_use_stft and (not self.ps_off):
            padded_excitation_signal = tf.pad(excitation_signal,  ((0,0), (self.stft_win_size//2,
                                                                self.stft_win_size//2 + self.spect_hop_size + 1)))

            if training and (self.pulse_noise_floor_mag is not None):
                # The STFT analysis resynthesis can pass through amplitudes that are so small that the gradient calculation then
                # produces NaN for these values which subsequently stop the training process. This is very common if
                # self.ng_forced_gain == 0 but can also happen in other situations.
                # Adding a very small noise to the source signals helps avoiding these situations.
                padded_excitation_signal += self.pulse_noise_floor_mag * tf.random.uniform(minval=-1, maxval=1,
                                                                                           shape=padded_excitation_signal.shape)

            source_stft = tf.signal.stft(padded_excitation_signal,
                                        frame_length=self.stft_win_size, frame_step=self.spect_hop_size,
                                        fft_length=self.fft_size, pad_end=False, name="source_stft")[:, :mel.shape[1]]

            # PulseFilterSpectrumGenerator
            if training:
                source_filter_stft = self.generate_specenv(
                    mel=mel,
                    pulse_frequency=tf.stop_gradient(pulse_frequency_),
                    training=training
                )
            else:
                source_filter_stft = self.generate_specenv(mel=mel, pulse_frequency=pulse_frequency_, training=training)

            if self.dump_controls:
                from fileio.iovar import save_var
                data = {"pulse_frequency": pulse_frequency.numpy(),
                        "pulse_signal": excitation_signal.numpy(),
                        "PulseFilterSpectrum": np.abs(source_filter_stft.numpy()),
                        }

                save_var("mbexiwn.p", data)

            signal_stft = source_stft * source_filter_stft
            signal = tf.signal.inverse_stft(signal_stft, frame_length=self.stft_win_size,
                                            frame_step=self.spect_hop_size, fft_length=self.fft_size,
                                            window_fn=tf.signal.inverse_stft_window_fn(
                                                frame_step=self.spect_hop_size,
                                                forward_window_fn=self.stft_win),
                                            name="PaN_iFFT")[:, self.stft_win_size // 2:self.stft_win_size // 2
                                                                                        + pulse_frequency.shape[
                                                                                            1] * int(self.sample_rate
                                                                                                     // self.pulse_rate)]
            if training and self.stft_coh_loss_weight:
                signal_stft_coh_test = tf.stop_gradient(source_stft) * source_filter_stft
                signal_coh_test = tf.signal.inverse_stft(signal_stft_coh_test, frame_length=self.stft_win_size,
                                                         frame_step=self.spect_hop_size, fft_length=self.fft_size,
                                                         window_fn=tf.signal.inverse_stft_window_fn(
                                                             frame_step=self.spect_hop_size,
                                                             forward_window_fn=self.stft_win),
                                                         name="PaN_coh_testiFFT")

                signal_stft_coh_test_inv = tf.signal.stft(signal_coh_test, frame_length=self.stft_win_size,
                                                          frame_step=self.spect_hop_size, fft_length=self.fft_size,
                                                          window_fn=self.stft_win, name="PaN_coh_testFFT")
                self.stft_coh_loss = (tf.reduce_mean(tf.square(tf.abs(signal_stft_coh_test)
                                                               - tf.abs(signal_stft_coh_test_inv)))
                                      / tf.reduce_mean(tf.square(tf.abs(signal_stft_coh_test))))

        else:
            if self.dump_controls:
                from fileio.iovar import save_var
                if self.ps_off:
                    excitation_signal = signal
                else:
                    excitation_signal = self.generate_excitation(mel, pulse_frequency=pulse_frequency_)
                data = {"pulse_frequency": pulse_frequency.numpy(),
                        "pulse_signal": excitation_signal.numpy(),
                        "multi_band_gain": multi_band_interpolated_gain.numpy(),
                        }

                save_var("mbexiwn.p", data)


        if return_PP:
            returned_PP = [["F0", pulse_frequency[:, :signal.shape[1]:int(self.sample_rate//self.pulse_rate)]]]

            if excitation_signal is not None:
                returned_PP.append(["PSig", excitation_signal[:, :signal.shape[1]]])
            elif padded_excitation_signal is not None :
                returned_PP.append([ "PSig", padded_excitation_signal[:, self.stft_win_size//2:self.stft_win_size//2+signal.shape[1]]])

            if source_filter_stft is not None:
                returned_PP.append([ "PS", tf.abs(source_filter_stft)])
        else:
            returned_PP = []

        returned_signals = [signal, ]

        return returned_signals, returned_PP

    def generate_f0(self, mel) -> tf.Tensor:
        """
        input are batched mel-frequency spectrograms
        """

        # PulseParameterGenerator
        if self.pp_subnet_layers:
            x = mel
            for ii, ll in enumerate(self.pp_subnet_layers):
                #print(f"layer {ii} x min {tf.reduce_min(x)} max {tf.reduce_max(x)} mean {tf.reduce_mean(x)}")
                x = ll(x)
                # print("out ",ii, "yy", y.shape)

            pulse_frequency =  x[:,:,0]*(self.pp_max_frequency - self.pp_min_frequency) + self.pp_min_frequency
            pulse_frequency = pulse_frequency[:,: mel.shape[1] * self.spect_to_pulse_upsampling_factor]
        else:
            pulse_frequency = tf.ones((mel.shape[0],
                                       mel.shape[1] * self.spect_to_pulse_upsampling_factor)) * self.pp_max_frequency
        return pulse_frequency

    def generate_specenv(self, mel, pulse_frequency, training=False):
        # PulseFilterSpectrumGenerator
        x = mel
        for ii, ll in enumerate(self.ps_subnet_layers):
            # print("in ",ii, "yy", y.shape)
            x = ll(x)
            # print("out ",ii, "yy", y.shape)

        if self.ps_env_order_scale:
            if training or not self.psns_use_cepstral_loss_constraint:
                cepstral_windows = self._get_cepstral_windows(pulse_frequency,
                                                              cepstral_windows_log10f0=self.ps_cepstral_windows_log10f0,
                                                              cepstral_windows=self.ps_cepstral_windows,
                                                              smooth_stride=self.spect_to_pulse_upsampling_factor)
                tf.assert_equal(cepstral_windows[:,:,0], 1., "problems with generated cepstral windows")
            if self.psns_use_cepstral_loss_constraint:
                smoothed_cepstrum = x
                if training:
                    self.PS_cepstral_loss = tf.reduce_mean(tf.abs(x*(1-cepstral_windows)))
            else:
                smoothed_cepstrum = x * cepstral_windows
        else:
            smoothed_cepstrum = x

        if not self.spect_filters_preserve_energy:
            # remove the first gain cepstral coefficient that is taken care of by the noise gain
            # complete missing part of the cepstrum to get the full DFFT
            source_filter_cepstrum = tf.pad(smoothed_cepstrum[:, :, 1:],
                                           ((0, 0), (0, 0), (1, self.fft_size - smoothed_cepstrum.shape[2])))
        else:
            # complete missing part of the cepstrum to get the full DFFT
            source_filter_cepstrum = tf.pad(smoothed_cepstrum,
                                           ((0, 0), (0, 0), (0, self.fft_size - smoothed_cepstrum.shape[2])))



        filter_log_amp_phase = tf.signal.rfft(source_filter_cepstrum)

        if self.filter_max_log_range:
            source_filter_stft = tf.exp(tf.complex(self.filter_max_log_range
                                                  * tf.math.tanh(tf.math.real(filter_log_amp_phase)),
                                                  tf.math.imag(filter_log_amp_phase)))
        else:
            source_filter_stft = tf.exp(filter_log_amp_phase)

        if self.spect_filters_preserve_energy:
            filter_gain = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(source_filter_stft)), axis=-1, keepdims=True))
            #if not tf.reduce_all(tf.math.is_finite(filter_gain)):
            #    from IPython import embed
            #    embed()
            source_filter_stft /= tf.cast(filter_gain, tf.complex64)
            #if not tf.reduce_all(tf.math.is_finite(tf.abs(pulse_filter_stft))):
            #    from IPython import embed
            #    embed()


        if self.spect_filters_preserve_energy:
            if self.psns_gain_loss_weight and training:
                # this loss function is using (x - 1 / x) has the nice property to by fully symmetric to 1
                # and to be 0 for the filter gain == 1
                self.PS_gain_loss = tf.reduce_mean(tf.square(filter_gain[:mel.shape[0]] - 1 / (filter_gain[:mel.shape[0]] + 0.001)))

        return source_filter_stft

    def generate_multiband_gain(self, mel, training=False):
        # PulseFilterSpectrumGenerator
        x = mel
        for ii, ll in enumerate(self.ps_subnet_layers):
            # print("in ",ii, "yy", y.shape)
            x = ll(x)
            # print("out ",ii, "yy", y.shape)

        channel_log_gain = x

        if self.spect_filters_preserve_energy:
            mean_channel_log_gain = tf.reduce_mean(channel_log_gain, axis=-1, keepdims=True)
            #if not tf.reduce_all(tf.math.is_finite(filter_gain)):
            #    from IPython import embed
            #    embed()
            channel_log_gain -= mean_channel_log_gain
            #if not tf.reduce_all(tf.math.is_finite(tf.abs(pulse_filter_stft))):
            #    from IPython import embed
            #    embed()


        if self.spect_filters_preserve_energy:
            if self.psns_gain_loss_weight and training:
                # this loss function is using (x - 1 / x) has the nice property to by fully symmetric to 1
                # and to be 0 for the filter gain == 1
                self.PS_gain_loss = tf.reduce_mean(tf.abs(mean_channel_log_gain[:mel.shape[0]] ))

        return tf.exp(channel_log_gain)

    def generate_excitation(self, mel, pulse_frequency, mb_gain=None):

        # print("pulse_frequency ", pulse_frequency_.shape)
        pulse_signal = self.pulse_generator(pulse_frequency)
        # print("pulse_signal ", pulse_signal.shape)

        if self.pulse_pqmf is None:
            x = tf.reshape(pulse_signal, (pulse_signal.shape[0], -1, self.pulse_channels*(1+self.pulse_generator.add_subharm_chans)))
        else:
            x = self.pulse_pqmf.analysis(tf.reshape(pulse_signal[:,:,0], (pulse_signal.shape[0], pulse_signal.shape[1], 1)))
            if self.pulse_generator.add_subharm_chans:
                x = tf.concat( [x,
                                tf.reshape(pulse_signal[:,:,1:],
                                           (pulse_signal.shape[0],
                                            -1, self.pulse_channels*self.pulse_generator.add_subharm_chans))], axis=-1)
        # print("pulse_signal folded ", x.shape)


        # Pulse/Noise Source Prozessor
        if self.pp_mod_subnet_noise_channel_sigma:
            x = tf.concat((x, self.pp_mod_subnet_noise_channel_sigma * tf.random.normal(x.shape[:-1]+(1,))), axis=-1)

        for ii, bl in enumerate(self.pp_waveNetBlocks):
            # print("in ",ii, "x", x.shape, "cond full", cond_input.shape, "sampled", cond_input_sampled.shape, "cond_step", cond_step)
            x = bl((x, mel))

        # convert to subband output
        for ll in self.wn_post_net:
            x = ll(x)

        if mb_gain is not None:
            x = x * mb_gain[:, :x.shape[1]]

        # print(f"wn post netshape {x.shape}")
        if self.pqmf is not None:
            source_signal = self.pqmf.synthesis(x)[:,:,0]
        else:
            source_signal = tf.reshape(x,(x.shape[0], x.shape[1]*x.shape[2]))

        return source_signal


    @property
    def pretrainable_weights(self) -> List[tf.Tensor]:
        # only the weights not reached through sub_layers need to be returned here
        return []

    @property
    def pretrain_activations(self):
        return self._pretrain_activations

    @pretrain_activations.setter
    def pretrain_activations(self, onoff):
        self._pretrain_activations = onoff

        for ll in self._flatten(recursive=False, predicate=self._filter_pretrainable_layers):
            if self.pp_subnet_training_only:
                if ll in self.pp_subnet_layers:
                    ll.pretrain_activations = onoff
            elif self.pp_subnet_exclude_from_pretrain:
                if ll not in self.pp_subnet_layers:
                    ll.pretrain_activations = onoff
            else:
                ll.pretrain_activations = onoff

    def _convert_signal_shape_into_stft_shape(self, signal_shape):
        num_samps = signal_shape[1]
        if num_samps is None:
            return (signal_shape[0], num_samps, self.fft_size // 2 + 1)
        num_frames = (num_samps // self.spect_hop_size) + 1
        return (signal_shape[0], num_samps, self.fft_size // 2 + 1)

    def build_or_compute_output_shape(self, input_shape, do_build=False) -> Union[None, Tuple]:

        # PulseParameterGenerator
        mel_shape = input_shape
        curr_shape = mel_shape
        if self.pp_subnet_layers:
            for ii, ll in enumerate(self.pp_subnet_layers):
                # print("in ",ii, "yy", y.shape)
                if do_build:
                    ll.build(curr_shape)
                curr_shape = ll.compute_output_shape(curr_shape)
        else:
            curr_shape = (mel_shape[0],
                          None if mel_shape[1] is None else mel_shape[1] *  self.spect_to_pulse_upsampling_factor)

        if self.pp_subnet_training_only:
           if do_build:
               return
           else:
               return curr_shape

        pulse_signal_shape = (curr_shape[0], None if  curr_shape[1] is None else curr_shape[1]//self.pulse_channels)

        if self.pp_waveNetBlocks:

            if self.pp_mod_subnet_noise_channel_sigma:
                curr_shape = pulse_signal_shape + (self.pulse_channels * (1 + self.pulse_generator.add_subharm_chans) + 1,)
            else:
                curr_shape = pulse_signal_shape + (self.pulse_channels * (1 + self.pulse_generator.add_subharm_chans),)
            # if mixing noise with pulses
            #curr_shape = pulse_signal_shape[:-1] + (pulse_signal_shape[-1] * 2)
            # wavenet group
            for ii, bl in enumerate(self.pp_waveNetBlocks):
                # print("in ",ii, "yy", y.shape)
                curr_shape = (curr_shape, mel_shape)
                if do_build:
                    bl.build(curr_shape)
                curr_shape = bl.compute_output_shape(curr_shape)

            for ll in self.wn_post_net:
                if do_build:
                    ll.build(curr_shape)
                curr_shape = ll.compute_output_shape(curr_shape)

        if self.pqmf is not None:
            if do_build :
                self.pqmf.build(curr_shape)
            out_shape = self.pqmf.compute_output_shape(curr_shape)
        else:
            if curr_shape[1] is not None:
                out_shape = (curr_shape[0], curr_shape[1] * curr_shape[2])
            else:
                out_shape = (curr_shape[0], curr_shape[1])

        # PulseFilterGenerator_TS
        if not self.ps_off:
            curr_shape = mel_shape
            for ii, ll in enumerate(self.ps_subnet_layers):
                if do_build:
                    ll.build(curr_shape)
                curr_shape = ll.compute_output_shape(curr_shape)

            if self.ps_gain_interpolator:
                self.ps_gain_interpolator.build(curr_shape)


        if not do_build:
            return out_shape



    def format_loss(self, losses):
        return "".join(["{}:{:6.3g} ".format(ff, ll) for ff, ll in zip(["F0_loss", "PSG_loss",
                                                                        "PSNSC_loss", "STC_loss"], losses[-4:])
                        if ll is not None])


    def total_loss(self, outputs, inputs=None, step=0):
        cepstral_loss = None
        if self.PS_cepstral_loss is not None:
            cepstral_loss = self.PS_cepstral_loss

        if self.F0_loss is not None:
            tf.summary.scalar(name='F0_loss', data=self.F0_loss)
        if self.stft_coh_loss is not None:
            tf.summary.scalar(name='stft_coh_loss', data=self.stft_coh_loss)
        if self.PS_gain_loss is not None:
            tf.summary.scalar(name='PS_gain_loss', data=self.PS_gain_loss)

        return ((self.F0_loss, self.pp_F0_loss_weight(step) if self.pp_F0_loss_weight is not None else None),
                (self.PS_gain_loss, self.psns_gain_loss_weight if self.PS_gain_loss is not None else None),
                (cepstral_loss , self.psns_cepstral_loss_weight if cepstral_loss is not None else None),
                (self.stft_coh_loss, self.stft_coh_loss_weight if self.stft_coh_loss is not None else None),
                )


    def summary(self, print_fn=print):
        input_shape = self._built_input_shape
        print_fn(f"SubModel {self.name}\n---------------------------------------")
        print_fn(f"{'Input':28s} -> {input_shape}")

        curr_shape = tuple(input_shape)
        if self.pp_subnet_layers:
            print_fn(f"{'PulseParameterGenerator':28s} input -> {curr_shape}")
            tot_num_weights = 0
            for ii, ll in enumerate(self.pp_subnet_layers):
                curr_shape = ll.compute_output_shape(curr_shape)
                num_weights=int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
                print_fn(f"  {ll.name:28s} -> {str(curr_shape):20s} ## {num_weights:d}")
                tot_num_weights += num_weights
            print_fn(f"  sub net params -> {tot_num_weights}")
        else:
            curr_shape = input_shape[0], input_shape[1] * self.spect_to_pulse_upsampling_factor

        if not self.pp_subnet_training_only:

            # pulse_frequency is generated without channels
            mel_shape = tuple(input_shape)
            curr_shape = (curr_shape[0], None if curr_shape[1] is None else curr_shape[1]//self.pulse_channels)

            if self.pp_waveNetBlocks:
                if self.pp_mod_subnet_noise_channel_sigma:
                    curr_shape = curr_shape + (self.pulse_channels * (1 + self.pulse_generator.add_subharm_chans) + 1,)
                else:
                    curr_shape = curr_shape + (self.pulse_channels * (1 + self.pulse_generator.add_subharm_chans),)
                print_fn(f"{'PulseFormGenerator':28s} input -> {curr_shape}")

                tot_num_weights = 0
                for ii, bl in enumerate(self.pp_waveNetBlocks):
                    curr_shape = (curr_shape, mel_shape)
                    curr_shape = bl.compute_output_shape(curr_shape)
                    num_weights = int(np.sum([np.prod(w.shape) for w in bl.trainable_weights]))
                    print_fn(f"  {bl.name:28s} -> {str(curr_shape):20s} ## {num_weights:d}")
                    if hasattr(bl, "summary"):
                        bl.summary(print_fn=print_fn, indent="  ")
                    tot_num_weights += num_weights
                print_fn(f"  sub net params -> {tot_num_weights}")

                tot_num_weights = 0
                print_fn(f"{'PulsePostNet':28s} input -> {curr_shape}")
                for ll in self.wn_post_net:
                    curr_shape = ll.compute_output_shape(curr_shape)
                    num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
                    print_fn(f"  {ll.name:28s} -> {str(curr_shape):20s} ## {num_weights:d}")
                    tot_num_weights += num_weights
                print_fn(f"  sub net params -> {tot_num_weights}")

            if self.pqmf is not None:
                pulse_signal_shape = self.pqmf.compute_output_shape(curr_shape)
            else:
                if  curr_shape[1] is not None:
                    pulse_signal_shape = (curr_shape[0], curr_shape[1] * curr_shape[2])
                else:
                    pulse_signal_shape = (curr_shape[0], curr_shape[1])

            if not self.ps_off:
                print_fn(f"{'PulseSignal':28s} -> {pulse_signal_shape}")
                curr_shape = input_shape
                if self.ps_use_stft:
                    pulse_stft_shape = self._convert_signal_shape_into_stft_shape(pulse_signal_shape)
                    print_fn(f"{'PulseSpectrum':28s} -> {pulse_stft_shape}")
                    print_fn(f"{'PulseSpectrumGenerator input':28s} -> {curr_shape}")
                else:
                    print_fn(f"{'PulseGainGenerator input':28s} -> {curr_shape}")

                tot_num_weights = 0
                for ii, ll in enumerate(self.ps_subnet_layers):
                    # print("in ",ii, "yy", y.shape)
                    curr_shape = ll.compute_output_shape(curr_shape)
                    num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
                    print_fn(f"  {ll.name:28s} -> {str(curr_shape):20s} ## {num_weights:d}")
                    tot_num_weights += num_weights

                if self.ps_gain_interpolator:
                    ll = self.ps_gain_interpolator
                    curr_shape = ll.compute_output_shape(curr_shape)
                    num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
                    print_fn(f"  {ll.name:28s} -> {str(curr_shape):20s} ## {num_weights:d}")
                    tot_num_weights += num_weights

                print_fn(f"  sub net params -> {tot_num_weights}")

            print_fn(f"{'Signal output':28s} -> {pulse_signal_shape[:2]}")
        print_fn("---------------------------------------")

    def get_config(self):
        config = super().get_config()
        config.update(spect_hop_size =self.spect_hop_size)
        config.update(preprocess_config =self.preprocess_config)
        config.update(pp_subnet =self.pp_subnet)
        config.update(pg_subnet =self.pg_subnet)
        config.update(ng_subnet =self.ng_subnet)
        config.update(ps_subnet =self.ps_subnet)
        config.update(ns_subnet =self.ns_subnet)
        config.update(ps_env_order_scale =self.ps_env_order_scale)
        config.update(ns_sigma =self.ns_sigma)
        config.update(use_prelu =self.use_prelu)
        config.update(alpha =self.alpha)
        config.update(pp_min_frequency =self.pp_min_frequency)
        config.update(pp_max_frequency =self.pp_max_frequency)
        config.update(wavetable_config =self.wavetable_config)
        config.update(pp_UV_teacher_forcing_limits_ms = self.pp_F0_pred_loss_limits_ms)

        return config






