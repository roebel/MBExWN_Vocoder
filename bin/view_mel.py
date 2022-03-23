#! /usr/bin/env python3
# AUTHOR:  A.Roebel
# COPYRIGHT: Copyright(c) 2022 IRCAM - Roebel

import os
import sys
import numpy as np
from argparse import ArgumentParser
from pysndfile import sndio
import matplotlib.pyplot as plt

# check whether we running from the source directroy, in which case the MBExWn_NVoc directry should be
# found next to the current directory. If this is the case we don't want to impot the installed version
test_path = os.path.join(os.path.dirname(__file__), '..', 'MBExWN_NVoc')
if os.path.exists(test_path):
    print(f"development version of MBExWN_NVoc directory detected. We will adapt the path to use it", file=sys.stderr )
    sys.path.insert(0, os.path.dirname(test_path))

from MBExWN_NVoc import list_models, mbexwn_version, get_config_file
from MBExWN_NVoc.sig_proc.resample import resample
from MBExWN_NVoc.fileio import iovar as iov
from MBExWN_NVoc.vocoder.model.config_utils import read_config, modify_config
from MBExWN_NVoc.vocoder.model.preprocess import compute_mel_spectrogram_internal

def uniq_name(infile, infile_list) :
    other_list = [ff for ff in infile_list if ff != infile]
    # if infile does not contain a directory part the following loop will not run and is not requird
    test_name = infile
    if other_list:
        for ind in range(len(os.path.basename(infile)), len(infile)):
            test_name = infile[-ind:]
            is_uniq =  True
            for ss in other_list:
                if ss.endswith(test_name):
                    is_uniq = False
                    break
            if is_uniq:
                break
    else:
        test_name = os.path.basename(test_name)
    return test_name

if __name__ == "__main__":
    parser = ArgumentParser(description="load and display stored matplotlib fig file")
    parser.add_argument("--infiles", nargs="+", help="sndfiles to analyse or mell files to load")
    parser.add_argument("--model_id", default="VOICE", nargs="?", const="",
                        help="model identifier that is used to read the config file. As all models share the same mel "
                             "analysis configuration the default model is fine here. "
                             "In the future models with different mel spectrum configurations may be supported. "
                             "If you do not specify an argument after the --model_id flag the script will "
                             "list all available models. Note that you can also specify a valid path to "
                             "a model directory to specify models that are not part of the MBExWN_NVoc package.")
    parser.add_argument("-ws", "--win_size_for_stats_s", default=0.050, type=float,
                        help="window size for signal stat calculation (Def: %(default)s)")
    parser.add_argument("-hs", "--hop_size_for_stats_s", default=0.010, type=float,
                        help="hop size size for signal stat calculation (Def: %(default)s)")
    parser.add_argument("-r", "--max_atten", default=50, type=int, help="db display range below max amplitude (Def: %(default)s)")
    parser.add_argument("-a", "--cargs", default=None, nargs="+",
                    help="arbitrary config file entries in with ':' as field separator (Def: %(default)s)")
    parser.add_argument("-d", "--diff_mel", action="store_true",
                    help="show difference of mel spectra - always compared to first file (Def: %(default)s)")
    parser.add_argument("-ps", "--plot_snds", action="store_true",
                    help="display sounds signals in a separte figure (Def: %(default)s)")
    parser.add_argument("-n", "--noise_mask_atten_db", default=None, type=float,
                    help="masking noise added befor calculating the mel spectrum (Def: %(default)s)")
    args = parser.parse_args()

    if not args.model_id:
        print("Please select one of the following models for mel inversion.\nYou don't need to select with a full ID. "
              "The first model containing the model_id you provide will be selected.\nFor example just specifying SPEECH will "
              "select the default SPEECH model.\nUsually, the default model is selected to be the most recent one.")
        for kk, ll in list_models().items():
            for md in ll:
                print(f" - {kk}/{md}")

        sys.exit(1)

    master_ax = None
    mastery_axs = None

    cc = read_config(get_config_file(args.model_id))
    # modify arbitrary config file entries
    cc = modify_config(cc, args.cargs)

    pp=cc["preprocess_config"]
    sr = pp["sample_rate"]
    plot_data = {}
    stat_data = {}
    snd_data = {}
    max_mell_val_db = -1000
    min_mell_val_db = 1000
    for infile in args.infiles:
        try:
            ss,  insr, _= sndio.read(infile, force_2d=True, dtype=np.float32)
            if sr != insr:
                ss,_ = resample(ss, insr, sr, axis=0)
            snd_data[infile]=  ss
            ss_masked = ss
            if args.noise_mask_atten_db is not None:
                ss_masked = ss + 10**(-args.noise_mask_atten_db/20) * np.sqrt(np.mean(ss*ss)) * np.random.standard_normal(ss.shape)
            mell = compute_mel_spectrogram_internal(ss_masked.T, preprocess_config=pp, do_post=False)[0][0].T
        except Exception:
            dd = iov.load_var(infile)
            if 'mell' in dd:
                mell = dd['mell']
            else:
                mell = np.log(np.fmax(dd['mel'], 1e-5))

        plot_data[infile] = mell
        print(f"infile {infile}: max mell {np.max(mell)}")
        max_mell_val_db = np.fmax(max_mell_val_db, np.max(mell))
        min_mell_val_db = np.fmin(min_mell_val_db, np.min(mell))

    fige = plt.figure(constrained_layout=True)

    if args.plot_snds:
        figs = plt.figure(constrained_layout=True)
    if args.diff_mel:
        figd = plt.figure()
    min_mell_val_db = np.fmax(min_mell_val_db, max_mell_val_db - args.max_atten)
    for ind, infile in enumerate(args.infiles, start=1):
        print(infile, args.infiles)
        id_file_name = uniq_name(infile, args.infiles)

        axe=fige.add_subplot(len(args.infiles), 1, ind, sharex=master_ax, sharey=master_ax)
        if master_ax is None:
            master_ax=axe

        vals = plot_data[infile]
        ime=axe.imshow(vals, origin='lower', aspect='auto',
                           extent=[0, vals.shape[1] * pp["hop_size"]/sr, 0, pp["mel_channels"]],
                           cmap="jet", vmax=max_mell_val_db, vmin=min_mell_val_db)
        fige.colorbar(ime, ax=axe, fraction=0.05,pad=0.02)
        axe.set_title(id_file_name)
        axe.grid(True)
        if ind != len(args.infiles):
            plt.setp(axe.get_xticklabels(), visible=False)
        if args.plot_snds and (infile in snd_data):
            axs = figs.add_subplot(len(args.infiles), 1, ind, sharex=master_ax, sharey=mastery_axs)
            if mastery_axs is None:
                mastery_axs = axs
            axs.plot(np.arange(len(snd_data[infile])) /sr, snd_data[infile], label="snd")

            axs.grid(True)
            axs.set_ylim((-1, 1))
            axs.set_title(id_file_name)
            axs.legend()
        if args.diff_mel:
            if ind == 1:
                ori_mell =  vals
            else:
                axd=figd.add_subplot(len(args.infiles), 1, ind, sharex=master_ax, sharey=master_ax)
                dmax =np.max(np.abs(vals-ori_mell))
                imd=axd.imshow(vals-ori_mell, origin='lower', aspect='auto',
                               extent=[0, vals.shape[1] * pp["hop_size"]/sr, 0, pp["mel_channels"]],
                               cmap="seismic", vmin=-dmax, vmax=dmax)
                figd.colorbar(imd, ax=axd, fraction=0.05,pad=0.02)
                axd.set_title(id_file_name)
                axd.grid(True)

    plt.show()

