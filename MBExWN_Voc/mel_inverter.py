
__copyright__= "Copyright (C) 2022 IRCAM"

import os, sys
import numpy as np
from pathlib import Path

from . import list_models, get_config_file
from .vocoder.model import config_utils as cutils
from .vocoder.model.models import create_model
from .vocoder.model.preprocess import log_to_db, compute_mel_spectrogram_internal

from typing import Dict, List, Tuple, Any, Union


class MELInverter(object):
    def __init__(self, model_id_or_path:Union[str, None]=None, verbose:bool=False):
        self.model = None
        self.model_id_or_path = model_id_or_path
        self.config_file = None
        self.preprocess_config = None
        self.mel_channels = None
        self.hop_size = None
        self.fft_size = None
        self.fmin = None
        self.fmax = None
        self._srate = None
        self.win_len = None

        self.lin_amp_scale = 1
        self.lin_amp_off = 1.e-5
        self.mel_amp_scale = 1
        self.use_max_limit = False

        if model_id_or_path:
            self.load_model(model_id_or_path=model_id_or_path, verbose=verbose)


    @property
    def srate(self):
        return self._srate

    def scale_mel(self, mel_config: Dict, verbose=False):
        lin_scale_win = 1

        if np.abs((mel_config['hoplen'] / mel_config['sr']) / (self.hop_size / self.srate) - 1) > 0.001:
            if verbose:
                print(f"compensate change in analysis hop size. "
                      f"mel analysis has {mel_config['hoplen'] / mel_config['sr']}"
                      f" while the model expects {self.hop_size / self.srate}.", file=sys.stderr)
            elif (not warn_given):
                print(f"at least one of the mel spectra does not have the correct hopsize. "
                      f"mel analysis has {mel_config['hoplen'] / mel_config['sr']} "
                      f"while the model expects {self.hop_size / self.srate}.", file=sys.stderr)
                warn_given = True
        if mel_config['sr'] != self.srate:
            if verbose:
                print(f"    WARNING::sample rate of mel analysis is  {mel_config['sr']} model expects {self.srate}.",
                      file=sys.stderr)

        if mel_config['fmin'] != self.fmin:
            raise RuntimeError(f"mell fmin {mel_config['fmin']} does not match model fmin {self.fmin}")
        if ((mel_config['fmax'] is None) and self.fmax != mel_config['sr'] / 2) or ((mel_config['fmax'] is not None) and mel_config['fmax'] != self.fmax):
            raise RuntimeError(f"mell fmax {mel_config['fmax']} does not match model fmax {self.fmax}")

        if "mell" in mel_config:
            log_mel_spectrogram = mel_config['mell'].T[np.newaxis]
            if "log_spec_offset" in mel_config and mel_config["log_spec_offset"] != 0:
                log_mel_spectrogram -= mel_config["log_spec_offset"]
            if "log_spec_scale" in mel_config and mel_config["log_spec_scale"] != 1:
                log_mel_spectrogram /= mel_config["log_spec_scale"]
            mel_spectrogram = np.exp(log_mel_spectrogram)
        elif "mel" in mel_config:
            mel_spectrogram = np.array(mel_config['mel'].T[np.newaxis])
            if verbose:
                log_mel_spectrogram = np.log(np.fmax(mel_spectrogram, np.finfo(mel_spectrogram.dtype).eps))
        else:
            raise RuntimeError(f"error::no supported mel spectrum (keys:mell or mell) in {mell_file}")

        dd_n_fft = mel_config.get("nfft", None)
        if dd_n_fft is None:
            dd_n_fft = mel_config.get("n_fft", None)
        if dd_n_fft is None:
            dd_n_fft = mel_config.get("fft_size", None)

        fft_scale_factor = self.fft_size // dd_n_fft
        if fft_scale_factor * lin_scale_win != 1:
            mel_spectrogram *= fft_scale_factor * lin_scale_win
            if verbose:
                log_mel_spectrogram += np.log(fft_scale_factor * lin_scale_win)

        if ("lin_spec_offset" in mel_config) and (mel_config["lin_spec_offset"] is not None) and (mel_config["lin_spec_offset"] != 0):
            mel_spectrogram -= mel_config["lin_spec_offset"]
        if "lin_spec_scale" in mel_config and mel_config["lin_spec_scale"] != 1:
            mel_spectrogram /= mel_config["lin_spec_scale"]

        if verbose:
            print(f"    max mel {log_to_db * np.max(log_mel_spectrogram):.3f} "
                  f"mean mel {np.mean(log_to_db * log_mel_spectrogram):.3f} "
                  f"min mel: {log_to_db * np.min(log_mel_spectrogram):.3f} mel.shape {mel_spectrogram.shape}", file=sys.stderr)

        if self.lin_amp_scale != 1:
            mel_spectrogram *= self.lin_amp_scale

        if self.use_max_limit:
            mell = np.log(np.fmax(mel_spectrogram, self.lin_amp_off)).astype(np.float32)
        else:
            mell = np.log(mel_spectrogram + self.lin_amp_off).astype(np.float32)

        if verbose:
            print(f"    stats conditioning mell:: mean: {log_to_db * np.mean(mell):.3f}dB, "
                  f"median: {log_to_db * np.median(mell):.3f}dB, max: {log_to_db * np.max(mell):.3f}dB, "
                  f"min: {log_to_db * np.min(mell):.3f}dB mell.shape {mell.shape}", file=sys.stderr)
            print(f"    mel params:: hoplen: {mel_config['hoplen']}, "
                  f"winlen: {mel_config['winlen']}, "
                  f"fft size: {dd_n_fft} srate: {mel_config['sr']}", file=sys.stderr)

        if np.abs((mel_config['hoplen'] / mel_config['sr']) / (self.hop_size / self.srate) - 1) > 0.001:
            if verbose:
                print(f"ATTENTION::interpolate mel spectrum to adapt hop "
                      f"size from {(mel_config['hoplen'] / mel_config['sr'])} to {self.hop_size / self.srate}", file=sys.stderr)
                log_mel_spectrogram = interp1d(
                    np.arange(mell.shape[1]) * mel_config['hoplen'] / mel_config['sr'],
                    log_mel_spectrogram,
                    axis=1,
                    bounds_error=False,
                    fill_value="extrapolate"
                )(
                    np.arange(
                        0,
                        (mell.shape[1] - 1 + 0.1) * mel_config['hoplen'] / mel_config['sr'],
                        self.hop_size / self.srate
                    )
                )
            mell = interp1d(
                np.arange(mell.shape[1]) * mel_config['hoplen'] / mel_config['sr'],
                mell,
                axis=1,
                bounds_error=False,
                fill_value="extrapolate"
            )(
                np.arange(
                    0,
                    (mell.shape[1] - 1 + 0.1) * mel_config['hoplen'] / mel_config['sr'],
                    self.hop_size / self.srate
                )
            ).astype(np.float32)

        return mell * self.mel_amp_scale


    def synth_from_mel(self, scaled_mell):
        syn_audio = self.model.infer(scaled_mell, sigma=None, synth_length=scaled_mell.shape[1] * self.hop_size).numpy()
        syn_audio = syn_audio.ravel()
        return syn_audio

    def generate_mel_from_snd(self, snd, srate):

        data_dict = {'nfft': self.fft_size,
                     'hoplen': self.hop_size,
                     'winlen': self.win_len,
                     'nmels': self.mel_channels,
                     'sr': self.srate,
                     'fmin': self.fmin,
                     'fmax': self.fmax,
                     'lin_spec_offset': self.lin_amp_off,
                     'lin_spec_scale': self.lin_amp_scale,
                     'log_spec_offset': 0.,
                     'log_spec_scale': self.mel_amp_scale,
                     "time_axis": 1}


        if srate != self.srate:
            snd, _ = resample(snd, srate, self.srate, axis=-1)

        if len(snd.shape) ==1:
            snd = np.array(snd)[np.newaxis]

        mel_ref, *rest = compute_mel_spectrogram_internal(snd, preprocess_config=self.preprocess_config,
                                                          band_limit=None, dtype=np.float32, do_post=False)

        data_dict['mell'] = mel_ref[0].T
        return data_dict

    def load_model(self, model_id_or_path, verbose=False):

        config_file = get_config_file(model_id_or_path=model_id_or_path)
        model_dir = os.path.dirname(config_file)
        hparams = cutils.read_config(config_file=config_file)
        training_config = hparams["training_config"]
        self.preprocess_config = hparams["preprocess_config"]

        # ## Instantiate model and optimizer
        self.model, mr_mode = create_model(
            hparams,
            training_config,
            self.preprocess_config,
            quiet=True,
        )

        # we need to run the model at least once si that all components are built otherwise the
        # state that is loaded from the checkpoint will disappear once the model is run
        # the first time when all layers are built.
        # Configure for arbitrary sound sizes
        self.model.build_model(variable_time_dim=True)

        model_weights_path = os.path.join(model_dir, "weights.tf")
        if verbose:
            print(f"restore from {model_weights_path}", file=sys.stderr)

        self.model.load_weights(model_weights_path)

        self.mel_channels= self.preprocess_config["mel_channels"]
        self.hop_size = self.preprocess_config["hop_size"]
        self.fft_size = self.preprocess_config["fft_size"]
        self.fmin = self.preprocess_config["fmin"]
        self.fmax = self.preprocess_config["fmax"]
        self._srate = self.preprocess_config['sample_rate']
        if 'win_size' in self.preprocess_config:
            self.win_len = self.preprocess_config['win_size']
        else:
            self.win_len = fft_size

        self.lin_amp_scale = 1
        if ("lin_amp_scale" in self.preprocess_config) and (self.preprocess_config["lin_amp_scale"] != 1):
            self.lin_amp_scale = self.preprocess_config["lin_amp_scale"]

        self.lin_amp_off = 1.e-5
        if "lin_amp_off" in self.preprocess_config and (self.preprocess_config["lin_amp_off"] is not None):
            self.lin_amp_off = self.preprocess_config["lin_amp_off"]

        self.mel_amp_scale = 1
        if ("mel_amp_scale" in self.preprocess_config) and (self.preprocess_config["mel_amp_scale"] != 1):
            self.mel_amp_scale = self.preprocess_config["mel_amp_scale"]

        self.use_max_limit = False
        if "use_max_limit" in self.preprocess_config and self.preprocess_config["use_max_limit"]:
            self.use_max_limit = self.preprocess_config["use_max_limit"]

        return

