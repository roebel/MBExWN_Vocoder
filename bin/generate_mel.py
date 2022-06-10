#!/usr/bin/env python3
# coding: utf-8
# AUTHOR:  A.Roebel
# COPYRIGHT: Copyright(c) 2022 IRCAM - Roebel

import os, sys
import numpy as np
# silence verbose TF feedback

from pysndfile import sndio

# check whether we running from the source directory, in which case the MBExWn_NVoc directory should be
# found next to the current directory. If this is the case we don't want to impot the installed version
test_path = os.path.join(os.path.dirname(__file__), '..', 'MBExWN_NVoc')
if os.path.exists(test_path):
    print(f"development version of MBExWN_NVoc directory detected. We will adapt the path to use it", file=sys.stderr )
    sys.path.insert(0, os.path.dirname(test_path))

from MBExWN_NVoc.fileio import iovar

from MBExWN_NVoc.vocoder.model import config_utils as cutils
from MBExWN_NVoc.vocoder.model.preprocess import compute_mel_spectrogram_internal
from MBExWN_NVoc import list_models, mbexwn_version, get_config_file
from MBExWN_NVoc.sig_proc.resample import resample


def main(input_audio_files, output_dir, model_id="VOICE"):


    config_file = get_config_file(model_id_or_path=model_id)

    if not os.path.exists(config_file) :
        raise FileNotFoundError(f"error::loading config file from {config_file}")

    hparams = cutils.read_config(config_file=config_file)
    preprocess_config = hparams['preprocess_config']

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dict = {'nfft': preprocess_config["fft_size"],
                 'hoplen': preprocess_config["hop_size"],
                 'winlen': preprocess_config["win_size"],
                 'nmels': preprocess_config["mel_channels"],
                 'sr': preprocess_config['sample_rate'],
                 'fmin': preprocess_config['fmin'],
                 'fmax': preprocess_config['fmax'],
                 'lin_spec_offset': preprocess_config['lin_amp_off'],
                 'lin_spec_scale': preprocess_config['lin_amp_scale'],
                 'log_spec_offset': 0.,
                 'log_spec_scale': preprocess_config['mel_amp_scale'],
                 "time_axis": 1}

    for audio_file in input_audio_files:
        print(f"process {audio_file}", file=sys.stderr)
        snd, sr, _ = sndio.read(audio_file, dtype=np.dtype("float32"))

        if sr != preprocess_config['sample_rate']:
            snd, _ = resample(snd, sr, preprocess_config['sample_rate'], axis=0)

        mel_ref, *rest = compute_mel_spectrogram_internal(snd, preprocess_config=preprocess_config,
                                               band_limit=None, dtype=np.float32, do_post=False)

        data_dict['mell'] = mel_ref[0].T
        iovar.save_var(
            os.path.join(
                output_dir,
                os.path.splitext(
                    os.path.basename(
                        audio_file
                    )
                )[0]+".mell"),
            data_dict,
        )


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser(description="create mel analysis from a given sound file using the configuration from a given config fle")
    parser.add_argument( "input_audio_files", nargs="+", help="input files to process")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="output directory where synthetic sounds will be stored")
    parser.add_argument("--model_id", default="VOICE", nargs="?", const="",
                        help="model identifier that is used to read the config file. As all models share the same mel "
                             "analysis configuration the default model is fine here. "
                             "In the future models with different mel spectrum configurations may be supported. "
                             "If you do not specify an argument after the --model_id flag the script will "
                             "list all available models. Note that you can also specify a valid path to "
                             "a model directory to specify models that are not part of the MBExWN_NVoc package.")

    args= parser.parse_args()
    args_dict = vars(args)
    main(**args_dict)
