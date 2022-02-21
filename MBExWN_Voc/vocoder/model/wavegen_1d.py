#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy
from typing import Dict, Union, Tuple
from utils import nextpow2_val
import tensorflow as tf

from sig_proc.spec.stft import get_stft_window

try:
    from librosa.core.convert import mel_frequencies as librosa_mel_frequencies
except ModuleNotFoundError:
    from librosa.core.time_frequency import mel_frequencies as librosa_mel_frequencies

from .preprocess import get_mel_filter
from .custom_layers import LinInterpLayer
from .custom_AE_layers import Conv1DUpDownSample
from .custom_pulsed_generator import MBWavePaN, MBExWN
from .config_utils import  check_spect_loss_config
from .tf_preprocess import TFSpectProcessor, TFMelProcessor
from .training_utils import ParamSchedule

from .tf2_components.tf2c_base_model import TF2C_BasePretrainableModel

class SpectLossComponents(tf.Module):
    spect_loss_bit = 1
    MCCTP_loss_bit = 2
    MODSPEC_loss_bit = 4
    MCCT_loss_bit = 8
    NLL_loss_bit = 16
    NPOW_loss_bit = 32
    PP_loss_bit = 64
    BC_loss_bit = 128

    def __init__(self, training_config, preprocess_config, quiet=False,
                 train_with_avg=None, sub_sample_facts=None):
        super().__init__()

        self.preprocess_config = copy.deepcopy(preprocess_config)
        self.training_config = copy.deepcopy(training_config)

        self.sub_sample_fact_of_stage_output = sub_sample_facts
        self.sample_rate = self.preprocess_config["sample_rate"]
        spect_loss_config = copy.deepcopy(self.training_config["spect_loss_config"])
        check_spect_loss_config(spect_loss_config)
        self.stage = self.training_config.get("stage", None)

        self.train_with_avg = train_with_avg

        self.MCCTP_loss_n = None
        self.MCCTS_loss_n = None
        self.spect_loss_n = None
        self.PP_loss_n = None
        self.BC_loss_n = None
        self.MODSPEC_loss_n = None
        self.NLL_loss_n = None
        self.NPOW_loss_n = None
        self.NLL_min_var = np.square(spect_loss_config.get("NLL_min_std", 0.01), dtype=np.float32)
        if "spect_loss_weight" in spect_loss_config:
            raise RuntimeError(
                "spect_loss_config::error::spect_loss_weight is no longer supported for MelGen generators files."
                "Use spect_loss_schedule instead.")
        if "spect_loss_schedule" in spect_loss_config:
            if spect_loss_config["spect_loss_schedule"] is None:
                self.spect_loss_weight = None
            else:
                self.spect_loss_weight = ParamSchedule(name="spect_loss_weight", quiet=quiet,
                                                       **spect_loss_config.get("spect_loss_schedule"))
        else:
            self.spect_loss_weight = ParamSchedule(name="spect_loss_weight", initial=1., quiet=quiet)


        self.MCCTP_loss_weight = spect_loss_config.get("MCCTP_loss_weight", 0)
        self.MCCTS_loss_weight = spect_loss_config.get("MCCTS_loss_weight", 0)
        self.PP_loss_weight = spect_loss_config.get("PP_loss_weight", 0)
        self.BC_loss_weight = spect_loss_config.get("BC_loss_weight", 0)
        self.MODSPEC_loss_weight = spect_loss_config.get("MODSPEC_loss_weight", 0)
        self.MCCT_loss_weight = spect_loss_config.get("MCCT_loss_weight", 0)
        self.NLL_loss_weight = spect_loss_config.get("NLL_loss_weight", 0)
        self.NPOW_loss_weight = spect_loss_config.get("NPOW_loss_weight", 0)

        self.PP_band_width_Hz = spect_loss_config.get("PP_band_width_Hz", 500)
        self.PP_segment_size_s = spect_loss_config.get("PP_segment_size_s", 0.025)
        self.PP_loss_method = spect_loss_config.get("PP_loss_method", "L1")

        self.BC_segment_size_s = spect_loss_config.get("BC_segment_size_s", 0.025)
        self.BC_loss_method = spect_loss_config.get("BC_loss_method", "L1")
        self.BC_max_off_Hz = spect_loss_config.get("BC_max_off_Hz", 2000.)
        self.MODSPEC_loss_method = spect_loss_config.get("MODSPEC_loss_method", "L1")

        MCC_segment_size_s = spect_loss_config.get("MCC_segment_size_s", 0.05)
        MCC_pad_size_s = spect_loss_config.get("MCC_pad_size_s", 0.02)
        self.MCCTS_segment_size = None
        self.MCCTS_pad_size = None
        self.MCC_segment_size = None
        self.MCC_pad_size = None
        self.PP_band_width_bins = None
        self.PP_segment_size = None

        self.spect_error_gain = spect_loss_config.get("spect_error_gain", 1)

        self.remove_mean_hz = spect_loss_config.get("remove_mean_hz", None)
        self.masking_noise_level = spect_loss_config.get("masking_noise_std", 0)
        self.rel_masking_noise_atten_db = spect_loss_config.get("rel_masking_noise_atten_db", None)
        self.rel_masking_noise_level = None
        if self.rel_masking_noise_atten_db is not None:
            self.rel_masking_noise_level = 10.**(-np.abs(self.rel_masking_noise_atten_db)/20.)

        self.mean_smoothing_win = None
        if self.remove_mean_hz:
            self.mean_smoothing_win = get_stft_window("nuttall4_6db",
                                                      win_len=int(4 * self.preprocess_config["sample_rate"]
                                                                  / self.remove_mean_hz),
                                                      dtype=np.float32)[:, np.newaxis, np.newaxis]
            self.mean_smoothing_win /= np.sum(self.mean_smoothing_win)

        if self.MCCTS_loss_weight:
            if not (MCC_segment_size_s and MCC_pad_size_s):
                raise RuntimeError("WaveSGen::error:: MCC_segment_size_s and MCC_pad_size_s must be set to " 
                                   "a valid duration if MCCTS_loss_weight is not 0")

            self.MCCTS_segment_size = int(MCC_segment_size_s * self.sample_rate)
            self.MCCTS_pad_size = int(MCC_pad_size_s * self.sample_rate)

        if "loss_type" in spect_loss_config:

            spect_loss_config["win_size"] = [wl for lt, wl in zip(spect_loss_config["loss_type"],
                                                                  spect_loss_config["win_size"]) if lt]
            spect_loss_config["hop_size"] = [hl for lt, hl in zip(spect_loss_config["loss_type"],
                                                                  spect_loss_config["hop_size"]) if lt]
            try:
                if ("fft_over" in spect_loss_config) and len(spect_loss_config):
                    spect_loss_config["fft_over"] = [fo for lt, fo in zip(spect_loss_config["loss_type"],
                                                                          spect_loss_config["fft_over"]) if lt]
            except TypeError:
                # scalar parameters do not need to adapt the number of element if indvidual loss types are 0
                pass

        self.stft_processor = None
        if ((self.spect_loss_weight is not None)
                or (self.NPOW_loss_weight > 0)
                or (self.MCCTP_loss_weight > 0) or (self.PP_loss_weight > 0)or (self.BC_loss_weight > 0)):
            self.stft_processor = TFSpectProcessor(spect_loss_config, self.sample_rate)

        if "loss_type" in spect_loss_config:
            self.PP_segment_size = [int((self.PP_segment_size_s + hs) / hs) if self.PP_segment_size_s else None
                                     for lt, hs in zip(spect_loss_config["loss_type"],
                                                       spect_loss_config["hop_size"]) if lt]
            self.PP_band_width_bins = [int((ffs * self.PP_band_width_Hz / self.sample_rate + 0.5)) if self.PP_band_width_Hz else None
                                       for ffs in self.stft_processor.fft_size]

            self.BC_segment_size = [int((self.BC_segment_size_s + hs) / hs) if self.BC_segment_size_s else None
                                     for lt, hs in zip(spect_loss_config["loss_type"],
                                                       spect_loss_config["hop_size"]) if lt]

            self.BC_max_off = [int(fft_size * self.BC_max_off_Hz/self.sample_rate)+1 if self.BC_max_off_Hz else None
                                     for lt, fft_size in zip(spect_loss_config["loss_type"],
                                                       self.stft_processor.fft_size) if lt]


            self.MCC_segment_size = [int((MCC_segment_size_s + hs) / hs)
                                     for lt, hs in zip(spect_loss_config["loss_type"],
                                                       spect_loss_config["hop_size"]) if lt]

            self.MCC_pad_size = [int((MCC_pad_size_s + hs) / hs)
                                 for lt, hs in zip(spect_loss_config["loss_type"],
                                                   spect_loss_config["hop_size"]) if lt]
            spect_loss_config["loss_type"] = [lt for lt in spect_loss_config["loss_type"] if lt]

            self.spect_loss_type = spect_loss_config["loss_type"]
            if not [1 for lt in self.spect_loss_type if (lt & self.MCCTP_loss_bit)]:
                self.MCCTP_loss_weight = 0
            if not [1 for lt in self.spect_loss_type if (lt & self.PP_loss_bit)]:
                self.PP_loss_weight = 0
            if not [1 for lt in self.spect_loss_type if (lt & self.BC_loss_bit)]:
                self.BC_loss_weight = 0
            if not [1 for lt in self.spect_loss_type if (lt & self.MODSPEC_loss_bit)]:
                self.MODSPEC_loss_weight = 0
            if not [1 for lt in self.spect_loss_type if (lt & self.MCCT_loss_bit)]:
                self.MCCT_loss_weight = 0
            if not [1 for lt in self.spect_loss_type if (lt & self.NLL_loss_bit)]:
                self.NLL_loss_weight = 0

        # frequency dependent weight
        self.low_band_extra_weight = spect_loss_config.get("low_band_extra_weight", 0)
        self.low_band_extra_weight_limit_Hz = spect_loss_config.get("low_band_extra_weight_limit_Hz", 0)
        self.low_band_extra_weight_transition_Hz = spect_loss_config.get("low_band_extra_weight_transition_Hz", 500)
        self.low_band_extra_weight = None
        if self.stft_processor is not None:
            if self.low_band_extra_weight :
                self.low_band_extra_weight = [self.low_band_extra_weight_generator(extra_weight=self.low_band_extra_weight,
                                                                                   transition=fs* self.low_band_extra_weight_transition_Hz/self.sample_rate,
                                                                                   position = fs * self.low_band_extra_weight_limit_Hz/ self.sample_rate,
                                                                                   length=fs//2+1) for fs in self.stft_processor.fft_size]
            else:
                self.low_band_extra_weight = [1 for _ in self.stft_processor.fft_size]

        self.mel_processor = None
        self.mel_loss_n = None
        self.mell_loss_weight = 0
        if ("mell_loss_weight" in spect_loss_config) and spect_loss_config["mell_loss_weight"] > 0:
            self.mel_processor = TFMelProcessor(self.preprocess_config,
                                                sub_sampled=(None
                                                             if self.stage is None
                                                             else self.sub_sample_fact_of_stage_output[self.stage]))
            self.mell_loss_weight = spect_loss_config["mell_loss_weight"]
            self.mell_loss_ign_attn_db = 40
            if ("lin_amp_off" in self.preprocess_config) and self.preprocess_config["lin_amp_off"] > 0:
                self.mell_loss_ign_attn_db = 0
        self.last_plot = 0


    def low_band_extra_weight_generator(self, extra_weight, transition, position, length):
        """
        spectal domain low frequency band extra weight generator
        generates a simple algebraic sigmoid weighting function W(f)
        the function transitions from
        W(0) = 1 + low_band_extra_weight,
        W(low_band_extra_weight_limit_Hz) = 1 + low_band_extra_weight * 0.5
        W(oo) = 1
        and achieves 90% and 10% of the transition at frequencies
        low_band_extra_weight_limit_Hz  +- 0.5 * low_band_extra_weight_transition_Hz
        """
        x = 4 * (-np.arange(length).astype(np.float32) + position) *2 / transition
        return (extra_weight * (0.5 + 0.5 * x / (1 + np.abs(x))) + 1)[np.newaxis, np.newaxis, :]


    def calc_spectral_error(self, ref_audio, gen_audio):
        spect_error = 0 if (self.spect_loss_weight is not None) else None
        MCCTP_error = 0 if self.MCCTP_loss_weight else None
        PP_error = 0 if self.PP_loss_weight else None
        BC_error = 0 if self.BC_loss_weight else None
        MODSPEC_error = 0 if self.MODSPEC_loss_weight else None
        MCCT_error = 0 if self.MCCT_loss_weight else None
        NLL_error = 0 if self.NLL_loss_weight else None
        NPOW_error = 0 if self.NPOW_loss_weight else None

        # print("calc_spectral_error::audio in:", ref_audio.shape, gen_audio.shape)

        spect_cnt = 0
        MCCTP_cnt = 0
        PP_cnt = 0
        BC_cnt = 0
        MODSPEC_cnt = 0
        MCCT_cnt = 0
        NLL_cnt = 0
        NPOW_cnt = 0
        if len(ref_audio.shape) == 3:
            in_spec_list = self.stft_processor.generate_stft(ref_audio[:, :, 0])
        elif len(ref_audio.shape) == 2:
            in_spec_list = self.stft_processor.generate_stft(ref_audio)
        else:
            in_spec_list = self.stft_processor.generate_stft(tf.reshape(ref_audio, (1, -1)))

        if self.train_with_avg is None or self.train_with_avg <= 1:
            if self.train_with_avg == 1:
                # remove variants dimension
                gen_audio = gen_audio[:, 0]
            if len(gen_audio.shape) == 3:
                syn_spec_list = self.stft_processor.generate_stft(gen_audio[:, :, 0])
            elif len(gen_audio.shape) == 2:
                syn_spec_list = self.stft_processor.generate_stft(gen_audio)
            else:
                syn_spec_list = self.stft_processor.generate_stft(tf.reshape(gen_audio, (1, -1)))
        else:
            if len(gen_audio.shape) == 4:
                syn_spec_list = self.stft_processor.generate_stft(gen_audio[:, :, :, 0])
            elif len(gen_audio.shape) == 3:
                syn_spec_list = self.stft_processor.generate_stft(gen_audio)
            else:
                syn_spec_list = self.stft_processor.generate_stft(tf.reshape(gen_audio, (1, gen_audio.shape[1], -1)))

        ri_ref_sp = None
        ri_syn_sp = None
        for i_spec, (_ref_sp, _syn_sp, lt, lbew) in enumerate(zip(in_spec_list, syn_spec_list, self.spect_loss_type, self.low_band_extra_weight)):
            # print(i_spec, lt, _ref_sp.shape, _syn_sp.shape)
            # print(i_spec, lt, ref_sp.shape, syn_sp.shape)
            if (self.stage is not None) and self.sub_sample_fact_of_stage_output[self.stage] != 1:
                # create TP filter corresponding to stage
                band_stop_ind = int((_syn_sp.shape[-1] - 1) // self.sub_sample_fact_of_stage_output[self.stage])
                band_pass_ind = int(0.95 * band_stop_ind)
                tp = tf.reshape(tf.concat((tf.ones(band_pass_ind, tf.float32),
                                           tf.linspace(1., 0., band_stop_ind - band_pass_ind),
                                           tf.zeros(_syn_sp.shape[-1] - band_stop_ind, tf.float32)
                                           ), axis=0), (1, 1, _syn_sp.shape[-1]))
                # print(f" tp.shape {tp.shape}  syn_sp.shape {_syn_sp.shape} ref_sp.shape {_ref_sp.shape}")
                # plt.figure()
                # print("refaudio:",ref_audio.shape)
                # plt.plot(ref_audio[0,: ,0].numpy())
                # plt.figure()
                # print("tp",tp.shape, tp.numpy())
                # plt.plot(tf.abs(tp[0,0]).numpy())
                # plt.figure()
                # plt.imshow(tf.abs(_ref_sp[0]).numpy().T)
                # plt.colorbar()
                # for the spectral loss we band limit only the input sound file
                _ref_sp = tf.cast(tp, _ref_sp.dtype) * _ref_sp
                # plt.figure()
                # plt.imshow(tf.abs(_ref_sp[0]).numpy().T)


            ref_sp = self.stft_processor.scale_spec(_ref_sp)
            syn_sp = self.stft_processor.scale_spec(_syn_sp)

            # produce average spectrum to reduce variance in the noise component which should avoid using pulses
            # for representing the mean spectrum of the noise level.
            syn_sp_std = None
            syn_sp_mean = None
            if self.train_with_avg is not None and self.train_with_avg > 1:
                # we slightly offset the estimated variance to ensure it does not become to small so that gradients explode. We do not do
                #    use tf.maximum here to keep the gradients working
                syn_sp_mean, syn_sp_var = tf.nn.moments(syn_sp, axes=[1])
                syn_sp_std = tf.sqrt(syn_sp_var + self.NLL_min_var)
                tf.debugging.assert_all_finite(syn_sp_std, "tf.reduce_any(syn_sp_std) is nan")
                syn_sp = syn_sp_mean

            if (self.spect_loss_weight is not None) and (lt & self.spect_loss_bit):
                # if self.last_plot > 100:
                #    import matplotlib.pyplot as plt
                #    plt.figure()
                #    plt.plot(ref_audio.numpy()[0])
                #    plt.plot(gen_audio.numpy()[0])
                #    plt.figure()
                #    plt.imshow(ref_sp[0].numpy().T)
                #    plt.colorbar()
                #    plt.figure()
                #    plt.imshow(syn_sp[0].numpy().T)
                #    plt.colorbar()
                #    plt.figure()
                #    plt.imshow((syn_sp[0]-ref_sp[0]).numpy().T)
                #    plt.colorbar()
                #    plt.show()
                spect_error += tf.reduce_mean(tf.abs(syn_sp - ref_sp) * lbew)
                spect_cnt += 1
            ref_sp_p = None
            syn_sp_p = None
            if (self.NPOW_loss_weight is not None) and (lt & self.NPOW_loss_bit):
                # if self.last_plot > 100:
                #    import matplotlib.pyplot as plt
                #    plt.figure()
                #    plt.plot(ref_audio.numpy()[0])
                #    plt.plot(gen_audio.numpy()[0])
                #    plt.figure()
                #    plt.imshow(ref_sp[0].numpy().T)
                #    plt.colorbar()
                #    plt.figure()
                #    plt.imshow(syn_sp[0].numpy().T)
                #    plt.colorbar()
                #    plt.figure()
                #    plt.imshow((syn_sp[0]-ref_sp[0]).numpy().T)
                #    plt.colorbar()
                #    plt.show()

                ref_sp_p = self.stft_processor.scale_spec_man_select(_ref_sp, magnitude_exponent=1)
                syn_sp_p = self.stft_processor.scale_spec_man_select(_syn_sp, magnitude_exponent=1)

                if False:
                    NPOW_error += tf.reduce_mean(tf.norm(syn_sp_p - ref_sp_p, ord="fro", axis=(1, 2))
                                                 / tf.norm(ref_sp_p, ord="fro", axis=(1, 2)))
                else:
                    NPOW_error += tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(syn_sp_p - ref_sp_p) * lbew, axis=[1,2]))
                                                 /tf.sqrt(tf.reduce_sum(tf.square(ref_sp_p) * lbew, axis=[1,2])
                                                   +tf.keras.backend.epsilon()*tf.keras.backend.epsilon()))
                NPOW_cnt += 1



        spect_loss_n = None
        MCCTP_loss_n = None
        PP_loss_n = None
        BC_loss_n = None
        MODSPEC_loss_n = None
        MCCT_loss_n = None
        NLL_loss_n = None
        NPOW_loss_n = None

        if spect_cnt:
            spect_loss_n = self.spect_error_gain * spect_error / spect_cnt
        if MCCTP_cnt:
            MCCTP_loss_n = MCCTP_error / MCCTP_cnt
        if BC_cnt:
            BC_loss_n = BC_error / BC_cnt

        if MODSPEC_cnt:
            MODSPEC_loss_n = MODSPEC_error / MODSPEC_cnt
        if MCCT_cnt:
            MCCT_loss_n = MCCT_error / MCCT_cnt
        if NLL_cnt:
            NLL_loss_n = NLL_error / NLL_cnt
        if NPOW_cnt:
            NPOW_loss_n = NPOW_error / NPOW_cnt

        return spect_loss_n, PP_loss_n, BC_loss_n, MCCT_loss_n, MCCTP_loss_n, NLL_loss_n, NPOW_loss_n, MODSPEC_loss_n

    def calc_losses(self, in_audio, outputs):

        mel_loss_n = None
        spect_loss_n = None
        PP_loss_n = None
        BC_loss_n = None
        MODSPEC_loss_n = None
        MCCT_loss_n = None
        MCCTP_loss_n = None
        MCCTS_loss_n = None
        NLL_loss_n = None
        NPOW_loss_n = None

        if self.mean_smoothing_win is not None:
            in_audio = in_audio - tf.nn.conv1d(in_audio, self.mean_smoothing_win, padding="SAME", stride=[1, 1, 1],
                                               data_format="NWC")
        if self.rel_masking_noise_level:
            in_audio = in_audio + tf.stop_gradient(tf.math.sqrt(tf.reduce_mean(tf.square(in_audio), axis=1, keepdims=True)) \
                                                   * self.rel_masking_noise_level * tf.random.normal(shape=in_audio.shape))
        if self.masking_noise_level:
            in_audio = in_audio + self.masking_noise_level * tf.random.normal(shape=in_audio.shape)

        if self.stft_processor:
            (spect_loss_n,
             PP_loss_n,
             BC_loss_n,
             MCCT_loss_n,
             MCCTP_loss_n,
             NLL_loss_n,
             NPOW_loss_n,
             MODSPEC_loss_n) = self.calc_spectral_error(in_audio, outputs)

        if self.mel_processor:
            mel_spect_syn = self.mel_processor(tf.reshape(outputs, outputs.shape[:2]))
            mel_spect_in = self.mel_processor(tf.reshape(in_audio, in_audio.shape[:2]))
            if self.mell_loss_ign_attn_db > 0:
                spect_min = (tf.reduce_max(tf.reduce_max(mel_spect_in, keepdims=True, axis=1),
                                           keepdims=True, axis=2)
                             - self.mell_loss_ign_attn_db / self.log_db_fac)
            else:
                spect_min = -100

            mel_loss_n = self.log_db_fac * tf.reduce_mean(
                tf.abs(tf.maximum(mel_spect_syn[:, :mel_spect_in.shape[1]], spect_min)
                       - tf.maximum(mel_spect_in, spect_min)))
            # if self.last_plot > 100:
            #    import matplotlib.pyplot as plt
            #    self.last_plot = 0
            #    plt.figure()
            #    plt.imshow( self.log_db_fac *mel_spect_in.numpy()[0].T)
            #    plt.colorbar()
            #    plt.figure()
            #    plt.imshow( self.log_db_fac *mel_spect_syn.numpy()[0].T)
            #    plt.colorbar()
            #    plt.figure()
            #    plt.imshow( self.log_db_fac *(mel_spect_in.numpy()[0].T - mel_spect_syn.numpy()[0].T))
            #    plt.colorbar()
            #    plt.show()
            # else:
            #    self.last_plot += 1


        return mel_loss_n, spect_loss_n, PP_loss_n, BC_loss_n, MCCT_loss_n, MCCTP_loss_n, MCCTS_loss_n, NLL_loss_n, NPOW_loss_n, MODSPEC_loss_n


class WaveGenerator(SpectLossComponents, TF2C_BasePretrainableModel):
    def __init__(self, model_config: Dict, training_config, preprocess_config,
                 sub_sample_facts=None, quiet=False, **kwargs):

        # cannot use super() here as the various base classes are
        tf.keras.Model.__init__(self, dtype=training_config['ftype'], **kwargs)
        self.sub_sample_facts = sub_sample_facts
        SpectLossComponents.__init__(self, training_config, preprocess_config, quiet=quiet,
                                     sub_sample_facts=self.sub_sample_facts,
                                     train_with_avg=model_config.get("ns_train_with_avg", 1))

        self.model_config = copy.deepcopy(model_config)
        self.training_config = copy.deepcopy(training_config)
        self.preprocess_config = copy.deepcopy(preprocess_config)
        self.norm_mel_components = None
        if self.model_config.get("normalize_rms_from_mell", False):
            self.norm_mel_components = NormMelComponents(preprocess_config=preprocess_config, dtype=self.dtype,
                                                         **model_config)

        self.win_size = self.preprocess_config['win_size']
        self.sample_rate = self.preprocess_config['sample_rate']
        self.sigma = self.model_config.get('sigma', None)
        self.mel_channels = self.preprocess_config['mel_channels']
        self.segment_length = self.preprocess_config['segment_length']
        self.spect_hop_size = self.preprocess_config['hop_size']

        self.log_db_fac = 20 * np.log10(2) / np.log(2)

        self.mel_loss_n = None
        self.spect_loss_n = None
        self.NLL_loss_n = None
        self.PP_loss_n = None
        self.MCCT_loss_n = None
        self.MCCTP_loss_n = None
        self.MCCTS_loss_n = None
        self.TD_loss_win = None
        self.TD_loss_weight = training_config.get("TD_loss_weight", 0.)
        TD_loss_win_len = training_config.get("TD_loss_win_len", 0)
        if TD_loss_win_len and self.TD_loss_weight > 0:
            self.TD_loss_win = np.hanning(TD_loss_win_len)[np.newaxis]
            self.TD_loss_win = self.TD_loss_win / np.sum(self.TD_loss_win)


    @property
    def has_components(self):
        """
        property for distinguishing models with signal components (PaN) and others that don't support components
        models that support components have an additional inference argument named "return_components"
        that will return the final mixed result and the individual components.
        """
        return False

    def get_config(self):
        config = super().get_config()
        config.update(model_config=self.model_config)
        config.update(training_config=self.training_config)
        config.update(preprocess_config=self.preprocess_config)
        config.update(sub_sample_facts=self.sub_sample_facts)

        return config


    def format_loss(self, losses):
        return "".join(
            ["{}:{:6.3f} ".format(ff, ll) if ll is not "SP_loss" else "{}:{:6.3g} ".format(ff, ll)
             for ff, ll in zip(["tot_loss", "SP_loss", "mel_loss",
                                "MCCTP_loss", "PP_loss", "BC_loss", "MCCT_loss", "MCCTS_loss",
                                "NLL_loss", "NPOW_loss",
                                "TD_rms", "MS_loss"], losses)
             if ll is not None])

    def summary(self, line_length=None, positions=None, print_fn=None, short=False):
        super().summary(line_length=line_length, positions=positions, print_fn=print_fn)
        if short:
            return

        print_fn(f"Model {self.name}")
        for ll in self.layers:
            print_fn(f"ll.name {ll.name} ll._built_input_shape {ll._built_input_shape}")
            ll.summary(print_fn=print_fn)

    def total_loss(self, outputs, inputs=None, step=0):

        (self.mel_loss_n, self.spect_loss_n, self.PP_loss_n, self.BC_loss_n,
         self.MCCT_loss_n, self.MCCTP_loss_n, self.MCCTS_loss_n,
         self.NLL_loss_n, self.NPOW_loss_n, self.MODSPEC_loss_n) = self.calc_losses(self.in_audio, outputs)

        total_loss_n = tf.constant(0, dtype=tf.float32)
        if self.spect_loss_n is not None:
            tf.summary.scalar(name='spec_loss_n', data=self.spect_loss_n)
            sp_loss_weight = self.spect_loss_weight(step)
            if sp_loss_weight > 0:
                total_loss_n += self.spect_loss_n * sp_loss_weight
        if self.mel_loss_n is not None:
            total_loss_n += self.mel_loss_n * self.mell_loss_weight
            tf.summary.scalar(name='mel_loss_n', data=self.mel_loss_n)
        if self.NLL_loss_n is not None:
            total_loss_n += self.NLL_loss_n * self.NLL_loss_weight
            tf.summary.scalar(name='NLL_loss_n', data=self.NLL_loss_n)
        if self.NPOW_loss_n is not None:
            total_loss_n += self.NPOW_loss_n * self.NPOW_loss_weight
            tf.summary.scalar(name='NPOW_loss_n', data=self.NPOW_loss_n)
        if self.MCCT_loss_n is not None:
            total_loss_n += self.MCCT_loss_n * self.MCCT_loss_weight
            tf.summary.scalar(name='MCCT_loss_n', data=self.MCCT_loss_n)
        if self.MCCTP_loss_n is not None:
            total_loss_n += self.MCCTP_loss_n * self.MCCTP_loss_weight
            tf.summary.scalar(name='MCCTP_loss_n', data=self.MCCTP_loss_n)
        if self.MCCTS_loss_n is not None:
            total_loss_n += self.MCCTS_loss_n * self.MCCTS_loss_weight
            tf.summary.scalar(name='MCCTS_loss_n', data=self.MCCTS_loss_n)
        if self.PP_loss_n is not None:
            total_loss_n += self.PP_loss_n * self.PP_loss_weight
            tf.summary.scalar(name='PP_loss_n', data=self.PP_loss_n)
        if self.BC_loss_n is not None:
            total_loss_n += self.BC_loss_n * self.BC_loss_weight
            tf.summary.scalar(name='BC_loss_n', data=self.BC_loss_n)
        if self.MODSPEC_loss_n is not None:
            total_loss_n += self.MODSPEC_loss_n * self.MODSPEC_loss_weight
            tf.summary.scalar(name='MODSPEC_loss_n', data=self.MODSPEC_loss_n)
        TD_loss_n = None
        if self.TD_loss_win is not None:
            print("ATTENTION::apparently the position of the DTLoss window at the sart of the segment is unstable ?"
                  " It appears the model is not able to reduce the TD loss at that position ")
            TD_loss_n = tf.sqrt(tf.reduce_sum(tf.square((tf.reshape(self.in_audio,
                                                                    self.in_audio.shape[:2])[:,
                                                         :self.TD_loss_win.shape[1]]
                                                         - tf.reshape(outputs,
                                                                      outputs.shape[:2])[:, :self.TD_loss_win.shape[1]])
                                                        * self.TD_loss_win)))
            total_loss_n += TD_loss_n * self.TD_loss_weight
            tf.summary.scalar(name='TD_loss_n', data=TD_loss_n)

        tf.summary.scalar(name='total_loss_n', data=total_loss_n)
        return (total_loss_n, self.spect_loss_n, self.mel_loss_n, self.MCCTP_loss_n, self.PP_loss_n, self.BC_loss_n, self.MCCT_loss_n,
                self.MCCTS_loss_n, self.NLL_loss_n, self.NPOW_loss_n, TD_loss_n, self.MODSPEC_loss_n)


class PaNWaveNet(WaveGenerator):
    def __init__(self, model_config, training_config, preprocess_config, quiet=False, use_tf25_compatible_implementation=False, **kwargs):

        mc_sub = copy.deepcopy(model_config)
        mc_sub.update(ns_train_with_avg=None)
        super().__init__(mc_sub, training_config, preprocess_config,
                         sub_sample_facts=None, quiet=quiet, **kwargs)

        self._pretrain_activations = False

        model_config_nonorm = copy.deepcopy(model_config)
        model_config_nonorm.pop("normalize_rms_from_mell", None)
        model_config_nonorm.pop("normalize_rms_num_smooth_iters", None)
        model_config_nonorm.pop("normalize_compressor_exp", None)
        model_config_nonorm.pop("normalize_smooth_win_scale", None)
        model_config_nonorm.pop("normalize_smooth_with_squared_win", None)
        model_config_nonorm.pop("normalize_use_pinv", None)

        if "ps_max_db_range" in model_config_nonorm:
            # map deprecated config name
            model_config_nonorm["filter_max_db_range"] = model_config_nonorm.pop("ps_max_db_range")
            if model_config_nonorm["ns_max_db_range"] != model_config_nonorm["filter_max_db_range"]:
                raise RuntimeError(f'MPWPaN::error::setting ns_max_log_range {model_config_nonorm["ns_max_log_range"]} != '
                                   f'ps_max_db_range {model_config_nonorm["filter_max_db_range"]} is no longer supported')
            model_config_nonorm.pop("ns_max_db_range")

        if "pulse_rate_factor" in model_config_nonorm:
            self.block = MBExWN(**model_config_nonorm, preprocess_config=preprocess_config, quiet=quiet,
                                use_tf25_compatible_implementation=use_tf25_compatible_implementation)
        else:
            self.block = MBWavePaN(**model_config_nonorm, preprocess_config=preprocess_config, quiet=quiet)
        self.pulse_frequency = None
        self.n_group = 1

    def build_model(self, variable_time_dim=False):
        if variable_time_dim:
            input_shape = (None, None, self.mel_channels)
        else:
            spect_steps = self.segment_length // self.spect_hop_size + 1
            if self.spect_hop_size * spect_steps < self.segment_length:
                spect_steps += 1
            input_shape = (None, spect_steps, self.mel_channels)

        self.build(input_shape=input_shape)

    def build_or_compute_output_shape(self, input_shape, do_build=False) -> Union[None, Tuple]:
        if do_build:
            self.block.build(input_shape=input_shape)
        else :
            return self.block.compute_output_shape(input_shape=input_shape)

    @property
    def has_components(self):
        """
        property for distinguishing models with signal components (PaN) and others that don't support components
        models that support components have an additional inference argument named "return_components"
        that will return the final mixed result and the individual components.
        """
        return True

    def call(self, inputs, training=None, test_grad=None, *args, **kwargs):
        """
        Evaluate model against inputs

        if training is false simply return the output of the infer method,
        which effectively run through the layers backward and invert them.
        Otherwise run the network in the training "direction".
        """

        spect = None
        F0 = None
        if len(inputs) == 2:
            self.in_audio, spect = inputs
        elif len(inputs) == 3:
            self.in_audio, spect, F0 = inputs

        audio = self.infer(spect, synth_length=self.in_audio.shape[1], F0=F0,
                           training=training if (training is not None) else True, test_grad=test_grad)
        return audio

    def infer(self, spect, sigma=None, z_in=None, synth_length=0, F0=None, return_F0=False, return_components=False,
              training=False, test_grad=None, **_):
        """
        Push inputs through network in reverse direction.
        Two key aspects:
        Layers in reverse order.
        Layers are inverted through exposed training boolean.
        """

        synth_length = synth_length if synth_length else self.segment_length
        if spect.shape[1] * self.spect_hop_size < synth_length:
            spect = tf.concat((spect, spect[:, -1:]), axis=1)

        if self.norm_mel_components is not None:
            _, in_mell, upsampled_rms = self.norm_mel_components.normalize_inputs_by_rms(None, spect,
                                                                                         synth_length=synth_length)
        else:
            in_mell = spect
            upsampled_rms = None

        signals, PP = self.block(in_mell, F0, training=training, return_PP=return_F0,
                                 return_components=return_components, test_grad=test_grad)
        # tf.debugging.check_numerics( audio, "audio is NaN", name=None)

        for ii in range(len(signals)):
            if signals[ii] is not None:
                if self.norm_mel_components is not None:
                    tmp = signals[ii][:, :synth_length] * upsampled_rms[:, :, 0]
                else:
                    tmp = signals[ii][:, :synth_length]

                # remove the dimensions that should not be seen by the caller
                signals[ii] = tmp

        # self.add_loss(tf.reduce_mean(tf.abs(tf.exp(mel_spec[:, 2:spect.shape[1] - 2]) - tf.exp(spect[:, 2:-2]))))
        if return_F0:
            # parameters are list of lists each containing a string and an array
            for ii in range(len(PP)):
                PP[ii][1] = PP[ii][1][:, :synth_length]
            if return_components:
                return signals, PP
            else:
                return signals[0], PP

        if return_components:
            return signals
        return signals[0]

    def infer_components(self, spect, synth_length=0, F0=None, transposition_factor=None):
        """
        Push inputs through network in reverse direction.
        Two key aspects:
        Layers in reverse order.
        Layers are inverted through exposed training boolean.
        """

        synth_length = synth_length if F0 is None else F0.shape[1]
        if spect.shape[1] * self.spect_hop_size < synth_length:
            spect = tf.concat((spect, spect[:, -1:]), axis=1)

        if self.norm_mel_components is not None:
            _, in_mell, upsampled_rms = self.norm_mel_components.normalize_inputs_by_rms(None, spect,
                                                                                         synth_length=synth_length)
            upsampled_rms = upsampled_rms[:, :, 0]
        else:
            in_mell = spect
            upsampled_rms = None

        if F0 is None:
            F0 = self.block.generate_f0(in_mell)

        if transposition_factor:
            F0 = transposition_factor * F0
        excitation_signal = self.block.generate_excitation(in_mell, F0)

        specenv = self.block.generate_specenv(in_mell, pulse_frequency=F0, training=False)
        # tf.debugging.check_numerics( audio, "audio is NaN", name=None)
        return F0, excitation_signal, specenv, upsampled_rms

    def format_loss(self, losses):
        loss_str = super().format_loss(losses)
        return loss_str + self.block.format_loss(losses)

    def total_loss(self, outputs, inputs=None, step=0):
        tot_loss, *other_losses = super().total_loss(outputs, inputs, step)
        model_losses = self.block.total_loss(outputs, inputs, step=step)
        block_losses = []
        for ll, ww in model_losses:
            block_losses.append(ll)
            if ll is not None:
                tot_loss += ll * ww
        return (tot_loss, *other_losses, *block_losses)

    def get_config(self):
        config = super().get_config()
        del config["sub_sample_facts"]
        return config


class NormMelComponents(tf.Module):

    def __init__(self, preprocess_config, n_group=1, max_norm_fact=None, normalize_compressor_exp=None,
                 lin_amp_scale=1., lin_amp_off=1.e-5,
                 mel_amp_scale=1., use_max_limit=False,
                 normalize_use_pinv=False, normalize_rms_num_smooth_iters=0, normalize_smooth_win_scale=1,
                 normalize_smooth_with_squared_win = True,
                 dtype=tf.float32, **kwargs):

        super().__init__()
        self.preprocess_config = copy.deepcopy(preprocess_config)
        self.spect_win_size = self.preprocess_config.get('win_size', self.preprocess_config['fft_size'])
        self.spect_hop_size = self.preprocess_config['hop_size']

        if 4 * self.spect_hop_size != self.spect_win_size:
            raise RuntimeError(
                "NormMelComponents:error: this module currently supports only the case where win_size {win_size} = 4 * hop_size {hop_size}")

        self.n_group = n_group
        self.rms_linear_interpolation_layer = None
        self.rms_norm_fact = self.preprocess_config["fft_size"] * self.spect_win_size * 0.5
        self.use_pinv = normalize_use_pinv
        self.smooth_win_scale = normalize_smooth_win_scale
        self.mel_channels = preprocess_config["mel_channels"]
        if normalize_use_pinv:
            self.win_norm = np.sqrt(np.sum(get_stft_window(win_type="hann", win_len=self.spect_win_size, dtype="float32") ** 2))
            mel_basis = get_mel_filter(sr=preprocess_config['sample_rate'], n_fft=preprocess_config['fft_size'],
                                       n_mels=self.mel_channels,
                                       fmin=preprocess_config["fmin"], fmax=preprocess_config["fmax"], dtype="float32")
            self.mel_band_filter_inverted = np.linalg.pinv(mel_basis).T
        else:
            mel_f = librosa_mel_frequencies(n_mels=self.mel_channels + 2, fmin=preprocess_config["fmin"],
                                            fmax=preprocess_config["fmax"])
            self.inv_enorm = ((mel_f[2: self.mel_channels + 2] - mel_f[:self.mel_channels]) / 2.).astype(np.float32)

        self.normalize_rms_num_smooth_iters = 0
        self.scale_mel_lin_amp_scale = lin_amp_scale
        self.max_norm_fact = max_norm_fact
        self.scale_mel_lin_amp_off = lin_amp_off
        self.normalize_compressor_exp = normalize_compressor_exp
        self.scale_mel_mel_amp_scale = mel_amp_scale

        self.scale_mel_use_max_limit = use_max_limit
        win = get_stft_window("hann", win_len=self.spect_win_size, dtype=dtype)[np.newaxis, :]
        self.gwin = win / np.sum(win)
        self.smooth_win_size = int(self.spect_win_size * self.smooth_win_scale)
        self.smooth_syn_win = get_stft_window("hann", win_len=self.smooth_win_size, dtype=dtype)[np.newaxis, :]
        self.normalize_smooth_with_squared_win = normalize_smooth_with_squared_win
        if self.normalize_smooth_with_squared_win:
            self.smooth_syn_win = self.smooth_syn_win ** 2

        if normalize_rms_num_smooth_iters > 0:
            self.normalize_rms_num_smooth_iters = normalize_rms_num_smooth_iters
        else:
            self.rms_linear_interpolation_layer = LinInterpLayer(upsampling_factor=self.spect_hop_size)

    def build(self, input_shape):
        if self.rms_linear_interpolation_layer is not None:
            self.rms_linear_interpolation_layer.build(input_shape=input_shape)

    def normalize_inputs_by_rms(self, audio, mell, synth_length=None):
        """

        This function normalizes the mel spectrogram such that the signal has approximately constant variance
        over the signal analysis window. The audio signal mean is normalized to 0

        :params audio: None or audio signal with dimension (batch-size, audio_length, 1)
        :params mell: log amplitude mel spectrum from that rms will be estimated
        :params synth_length: audio signal length, will be used only if audio is None

        returns: estimate rms of audio from mell and return a 3-tuple of the

          audio input folded into n_group channels (or None if audio is None) normalized by rms estimate
          mell (supposed to be in log amplitude) normalized by rms_estimate and upsampled to audio.shape[1]//self.n_group
          the rms-estimate of length (audio.shape[1] or synth_length)//self.n_group
        """

        # estimate rms from log amplitude mell spectrum where mel band filters are assumed to be applied either to
        # magnitude spectra or to power spectra with subsequent sqrt

        if audio is not None:
            snd_lengths = audio.shape[1]
        elif synth_length is not None:
            snd_lengths = synth_length
        else:
            raise RuntimeError(
                "normalize_inputs_by_rms:error:either audio or synth_length parameter needs to be present")

        mel = tf.exp(mell)
        # Here we approximately calculate the energy of the signal under two different assumptions:
        # For use_pinv == False
        # we assume that the signal spectrum is sparse and concentrated at the center of the melbands
        # The fact that all the energy is assumed to be concentrated in a single point will lead to an over-estimation
        # of the energy depending on the band width. The wider the band the more the energy will be over-estimated
        # Similarly the impact of noise energy that tends to be more spread than sinusoidal energy will also be .
        # over-estimated. As a result the energy estimate will be to high which means that the normalization will
        # produce a signal with an energy smmaler than 1. It turns out however that the signals maximum amplitude over
        # ime is generally more constant compared to the more consistent energy estimation assuming a spread signal
        # spectrum.
        # For use_pinv == True
        # we assume that the signal spectrum is rather smoothly distributed over all
        # melbands. For the ambiguous inversion the pseudo inverse produces the input signal with minimum energy
        # that can explain the output mel spectrogram, so in general the energy estimate is too low. Therefore
        # the normalization factor will be too high. Visual signal inspection reveals that notably for
        # fricative phonemes the normalized signal has amplitude values that are higher than the voiced segments.
        # This seems unfortunate because it will lead
        # to the fact that unvoiced segments will tend to have a larger impact on the error.
        if self.use_pinv:
            mell_test = tf.tensordot(mel, self.mel_band_filter_inverted, axes=1) / self.win_norm
            rms_mel_ampl = tf.sqrt(tf.reduce_sum(tf.square(mell_test), axis=-1) / self.rms_norm_fact)
        else:
            rms_mel_ampl = tf.sqrt(tf.reduce_sum(tf.square(mel * self.inv_enorm), axis=-1) / self.rms_norm_fact)
        if self.max_norm_fact:
            rms_mel_ampl = tf.maximum(rms_mel_ampl, np.float32(1. / self.max_norm_fact))
        if self.normalize_compressor_exp is not None:
            rms_mel_ampl = tf.math.pow(rms_mel_ampl, self.normalize_compressor_exp)

        gain = None
        ##print(f"snd.shape {snd_lengths} mell.shape {mell.shape} self.rms_norm_fact { 1/self.rms_norm_fact} rms_mel_ampl {rms_mel_ampl.shape} {np.mean(rms_mel_ampl)}")
        if self.normalize_rms_num_smooth_iters > 0:

            # broadcasting could be used as well but matmul is about twice as fast
            norm_gain_frames = tf.linalg.matmul(tf.concat((tf.ones((1, 2), dtype=np.float32),
                                                           tf.ones((1,) + rms_mel_ampl.shape[1:], dtype=np.float32),
                                                           tf.ones((1, 2), dtype=np.float32)), axis=1)[:, :, np.newaxis],
                                                self.smooth_syn_win)
            norm_gain = tf.signal.overlap_and_add(norm_gain_frames,
                                                  self.spect_hop_size)[:, self.smooth_win_size//2+2*self.spect_hop_size-self.spect_win_size//2:]
            ##first_ind = self.spect_win_size //2-1
            ##first_ind2 = self.spect_win_size //2
            ##last_ind = first_ind2 + snd_lengths
            ##last_ind2 = first_ind2  + snd_lengths + self.spect_hop_size
            ##print(f"norm_gain.shape {norm_gain.shape} norm_gain[0,[{first_ind}, {first_ind2}, {last_ind}, {last_ind2}]]"
            ##          f"norm_gain:{norm_gain[0, first_ind]}, {norm_gain[0, first_ind2]}, {norm_gain[0, last_ind]}, {norm_gain[0, last_ind2]}")

            for _ in range(self.normalize_rms_num_smooth_iters):
                gain_frames = tf.linalg.matmul(tf.concat((rms_mel_ampl[:, :1], rms_mel_ampl[:, :1],
                                                          rms_mel_ampl,
                                                          rms_mel_ampl[:, -1:], rms_mel_ampl[:, -1:]), axis=1)[:,:, np.newaxis],
                                               self.smooth_syn_win)
                gain = tf.signal.overlap_and_add(gain_frames, self.spect_hop_size)[:,self.smooth_win_size//2+2*self.spect_hop_size-self.spect_win_size//2:]
                ##print(f"gain.shape {gain.shape} gain[0,[{first_ind}, {first_ind2}, {last_ind}, {last_ind2}]]"
                ##      f"gain {gain[0, first_ind]}, {gain[0, first_ind2]}, {gain[0, last_ind]}, {gain[0, last_ind2]}")
                ##print(f"gain {np.mean(gain)} norm_gain {np.mean(norm_gain)}")
                gain = gain / tf.maximum(tf.keras.backend.epsilon(), norm_gain)
                rms_mel_ampl = tf.nn.conv1d(tf.expand_dims(gain, axis=-1), self.gwin.T[:, :, np.newaxis],
                                            stride=[1, self.spect_hop_size, 1], padding="VALID", data_format="NWC")[:,
                               :mell.shape[1], 0]
            rms_mel_ampl = tf.expand_dims(rms_mel_ampl, axis=-1)
        else:
            rms_mel_ampl = tf.sqrt(tf.reduce_mean(tf.square(mel), keepdims=True, axis=1) )

        # rescale mell output after normalization
        mel = mel / (tf.maximum(tf.keras.backend.epsilon(), rms_mel_ampl)) * self.scale_mel_lin_amp_scale

        if self.scale_mel_use_max_limit:
            mell = self.scale_mel_mel_amp_scale * tf.math.log(tf.maximum(mel, self.scale_mel_lin_amp_off))
        else:
            mell = self.scale_mel_mel_amp_scale * tf.math.log(mel + self.scale_mel_lin_amp_off)

        grp_audio = None

        if self.normalize_rms_num_smooth_iters > 0:
            gain_off = int(self.spect_win_size // 2)
            upsampled_rms = tf.maximum(gain[..., gain_off:gain_off + snd_lengths], tf.keras.backend.epsilon())
            upsampled_rms = tf.reshape(upsampled_rms, (mell.shape[0], -1, self.n_group))
        else:
            upsampled_rms = self.rms_linear_interpolation_layer(rms_mel_ampl)

        if audio is not None:
            grp_audio = tf.reshape(audio, (audio.shape[0], -1, self.n_group))
            if grp_audio.shape[1] > upsampled_rms.shape[1]:
                upsampled_rms = tf.concat((upsampled_rms,
                                           tf.repeat(upsampled_rms[:, -1:, :],
                                                     grp_audio.shape[1] - upsampled_rms.shape[1],
                                                     axis=1)), axis=1)
            elif grp_audio.shape[1] < upsampled_rms.shape[1]:
                upsampled_rms = upsampled_rms[:, :grp_audio.shape[1]]

            # print(f"rms_mel_ampl.shape {rms_mel_ampl.shape}, upsampled_rms.shape {upsampled_rms.shape},spect {spect.shape}, upsampled_spect.shape {upsampled_spect.shape},  grp_audio {grp_audio.shape}")
            grp_audio = grp_audio / upsampled_rms

        elif synth_length is not None:
            if synth_length // self.n_group > upsampled_rms.shape[1]:
                upsampled_rms = tf.concat((upsampled_rms,
                                           tf.repeat(upsampled_rms[:, -1:, :],
                                                     synth_length // self.n_group - upsampled_rms.shape[1],
                                                     axis=1)), axis=1)
            elif synth_length // self.n_group < upsampled_rms.shape[1]:
                upsampled_rms = upsampled_rms[:, :(synth_length // self.n_group)]

        return grp_audio, mell, upsampled_rms
