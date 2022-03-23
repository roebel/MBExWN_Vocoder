#!/usr/bin/env python3
# coding: utf-8
# AUTHOR:  A.Roebel
# COPYRIGHT: Copyright(c) 2022 IRCAM - Roebel

import os, sys

import numpy as np
# silence verbose TF feedback
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

from pathlib import Path
import tensorflow as tf
import time
from pysndfile import sndio
try :
    import manage_gpus as gpl
    have_manage_gpus=True
except (ImportError, ModuleNotFoundError):
    have_manage_gpus=False

# check whether we running from the source directory, in which case the MBExWN_NVoc directory should be
# found next to the current directory. If this is the case we don't want to import the installed version
test_path = os.path.join(os.path.dirname(__file__), '..', 'MBExWN_NVoc')
if os.path.exists(test_path):
    print(f"development version of MBExWN_NVoc directory detected. We will adapt the path to use it", file=sys.stderr )
    sys.path.insert(0, os.path.dirname(test_path))

from MBExWN_NVoc.sig_proc import db
from MBExWN_NVoc import mel_inverter, list_models, mbexwn_version
from MBExWN_NVoc.fileio import iovar as iov

def main(model_id, input_mell_files, output_dir,
         single_seg_synth=True, use_gpu=False, sigma=None,
         format="flac", verbose=False, pre_scale_mel = None, seed=42, num_threads=2, quiet=False):

    if use_gpu:
        if have_manage_gpus:
            # configure GPU
            try:
                gpu_ids = gpl.board_ids()
                if gpu_ids is not None:
                    gpu_device_id = -1
                else:
                    print("resynth_mel::warning:: no gpu devices available on this system, please force using cpu", file=sys.stderr)
                    sys.exit(0)
            except gpl.NoGpuManager:
                gpu_device_id = 0

            # now we lock a GPU because we will need one
            if gpu_device_id is not None:
                try:
                    gpl.get_gpu_lock(gpu_device_id=gpu_device_id, soft=False)
                except gpl.NoGpuManager:
                    print("resynth_mel::warning::no gpu manager available - will use all available GPUs", file=sys.stderr)
                    pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=""

    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)

    # seed random number generators
    if seed >= 0:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    MelInv = mel_inverter.MELInverter(model_id_or_path=model_id)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for mell_file in input_mell_files:
        outfile = os.path.join(output_dir, "syn_" + os.path.splitext(os.path.basename(mell_file))[0]+"."+format)

        if not quiet:
            print(f"synthesize {mell_file} into {outfile}", file=sys.stderr)

        if verbose:
            print(f"load mell  from {mell_file}", file=sys.stderr)
        dd = iov.load_var(mell_file)

        log_mel_spectrogram = MelInv.scale_mel(dd, verbose=verbose)

        start_time = time.time()
        syn_audio = MelInv.synth_from_mel(log_mel_spectrogram)
        end_time = time.time()

        if verbose:
            mel_resyn = MelInv.generate_mel_from_snd(syn_audio, srate=MelInv.srate)['mell'].T[np.newaxis]
            mell_err = mel_inverter.log_to_db * np.mean(np.abs(log_mel_spectrogram-mel_resyn[:,:log_mel_spectrogram.shape[1]]))
            
            print(f"    synthesized audio with {syn_audio.size} samples "
                  f"in {end_time-start_time:.3f}s ({syn_audio.size/(end_time-start_time):.2f}Hz), "
                  f"mel_error: {mell_err:.3f}dB", file=sys.stderr)

        if np.max(np.abs(syn_audio)) > 1:
            norm = 0.99/np.max(np.abs(syn_audio))
            print(f'    to prevent clipping you would need to normalize {outfile} by {norm:.3f}', file=sys.stderr)

        if verbose:
            print(f"    save audio under {outfile}", file=sys.stderr)
        sndio.write(outfile, data=syn_audio, rate=MelInv.srate, format=format)


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser(description="pass a given audio file through and analysis/resynthesis cycle using a waveglow model")
    parser.add_argument("model_id", default=None, nargs="?", const =None,
                        help="model identifier. If not given the script will list all known model names, one "
                             "of which you should then select to be used. You don't need the full model name "
                             "the model to be used will be the first in the list of models that contains the given identifier. "
                             "Note that you can also specify a valid path to "
                             "a model directory to specify models that are not part of the MBExWN_NVoc package.")
    parser.add_argument("-i", "--input_mell_files", nargs="+",  help="list of mell spectra stored in pickle files")
    parser.add_argument("-o", "--output_dir",  help="output directory where synthetic sounds will be stored")
    parser.add_argument("--format", default="flac", help="file format for generated audio files (Def: %(default)s)")
    parser.add_argument("-nt", "--num_threads", default=2, type=int, help="number of cpu threads (Def: %(default)s)")
    parser.add_argument("-g", "--use_gpu", action="store_true", help="run on gpu")
    parser.add_argument("-v", "--verbose", action="store_true", help="display verbose progress info")
    parser.add_argument("-q", "--quiet", action="store_true", help="dont display progress")

    args= parser.parse_args()

    if not args.model_id:
        print("Please select one of the following models for mel inversion.\nYou don't need to select with a full ID. "
              "The first model containing the model_id you provide will be selected.\nFor example just specifying SPEECH will "
              "select the default SPEECH model.\nUsually, the default model is selected to be the most recent one.")
        for kk, ll in list_models().items():
            for md in ll:
                print(f" - {kk}/{md}")
    else:
        main(**vars(args))
