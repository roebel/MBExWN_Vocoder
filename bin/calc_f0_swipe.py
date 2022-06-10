#! /usr/bin/env python

from __future__ import division, print_function, absolute_import

import os, sys

# check whether we running from the source directory, in which case the MBExWn_NVoc directory should be
# found next to the current directory. If this is the case we don't want to impot the installed version
test_path = os.path.join(os.path.dirname(__file__), '..', 'MBExWN_NVoc')
devel_warn = False
if os.path.exists(test_path):
    devel_warn = True
    sys.path.insert(0, os.path.dirname(test_path))

import datetime
import argparse
import numpy as np
from fileio import iovar
import pysndfile.sndio as sndio
import sig_proc.swipe as swipe

def calc_f0_swipe(infile, outfile, f0min=50, f0max=450, time_step=0.005,
                  harmonicity_threshold=None,
                  use_spline_interp=False, ana_freq_limit=12500, music_mode=False,
                   remove_low_harm_f0=True, channel=0, verbose=False, txt_format=None):
    '''
    use swipe with given parameters to create the f0 analysis of infile and store the result in an sdif file

    raises RuntimeError in case of errors
    
    '''


    interp_mode = "zp"
    if use_spline_interp:
        interp_mode = "cubic"
    ana_mode = "speech"
    if use_spline_interp:
        ana_mode = "music"

    if verbose:
        print("run: swipe analysis for {0} to {1} f0min:{2:f} f0max:{3:f} fana:{4:f} interp:{5} config:{6} dt:{7:f}s ht:{8}".format(infile, outfile, f0min, f0max, ana_freq_limit, interp_mode, ana_mode,
                    time_step, harmonicity_threshold), file=sys.stderr)

    x, fs, enc = sndio.read(infile)
    if x.ndim > 1:
        x = x[:,channel]
    p, t, s = swipe.swipe(x, fs=fs, plim=np.array([f0min, f0max]), dt=time_step,
                          spline_interp = use_spline_interp, music_mode=music_mode,
                            freq_limit=ana_freq_limit)

    #    print np.transpose(np.array((ss.t, ss.p, ss.h)))

    f0vals = np.zeros((len(p), ))
    time_pos  = np.zeros((len(p),))
    harmonicity = np.zeros((len(p),))
    data_ind = 0
    for (ti, f0, hh) in zip(t, p, s) :
        if harmonicity_threshold :
            if remove_low_harm_f0:
                if  harmonicity_threshold < hh :
                    time_pos[data_ind]    = ti
                    f0vals[data_ind] = f0
                    harmonicity[data_ind] = hh
                    data_ind += 1
            else:
                time_pos[data_ind]    = ti
                if  harmonicity_threshold < hh :
                    f0vals[data_ind] = f0
                    harmonicity[data_ind] = hh
                data_ind += 1
        else:
            time_pos[data_ind]    = ti
            f0vals[data_ind] = f0
            harmonicity[data_ind] = hh
            data_ind += 1

    if txt_format:
        sep =""
        if txt_format == "csv":
            sep = ","
        elif txt_format == "tsv":
            sep = "\t"

        with open(outfile, "w") as fo:
            for t, f0, ha in zip(time_pos[:data_ind], f0vals[:data_ind], harmonicity[:data_ind]):
                print(f"{t:.7e}{sep} {f0:.5e}{sep} {ha:.4e}", file=fo)
    else:
        data_dict = {}
        data_dict['written_by']         = 'calc_f0_swipe.py'
        data_dict['harm_thresh']        = str(harmonicity_threshold)
        data_dict['remove_low_harm_f0'] = str(remove_low_harm_f0)
        data_dict['Soundfile']          = os.path.relpath(os.path.abspath(infile),
                                                    os.path.dirname(os.path.abspath(outfile)))
        data_dict['f0min']              = str(f0min)
        data_dict['f0max']              = str(f0max)
        data_dict['fana']               = str(ana_freq_limit)
        data_dict['mode']               = ana_mode
        data_dict['interp']             = interp_mode
        data_dict['date']               = datetime.datetime.now().strftime('%Y-%m-%d %H:%M.%S')

        data_dict["f0_times"] = time_pos[:data_ind]
        data_dict["f0_vals"] = f0vals[:data_ind]
        data_dict["f0_harmonicity"] = harmonicity[:data_ind]
        iovar.save_var(outfile,        data_dict   )
    return

if __name__ == "__main__" :

    prog = sys.argv[0]
    usage = "Usage: %prog infile outfile \n estimate f0 using swipe "
    parser = argparse.ArgumentParser(description=usage)

    parser.add_argument("infile", help="inputfile")
    parser.add_argument("outfile", help="sdif output file name")
    parser.add_argument("--f0min", default=50, type=float,
                        help="minimum of f0 range (def: %(default)s )" )
    parser.add_argument("--f0max",  default=550, type=float,
                        help="maximum of f0 range (def: %(default)s )" )
    parser.add_argument("--ana_fmax",  default=12500, type=float,
                        help="maximum frequency to consider for spectral analysis (def: %(default)s )" )
    parser.add_argument("-c", "--channel",  default=0, type=int,
                        help="channel to select for analysis of multi channel files (def: %(default)s )" )
    parser.add_argument("-s", "--time_step", dest="time_step", default=0.005, type=float,
                        help="time step of the analysis (def: %(default)s)" )
    parser.add_argument("--use_spline", action="store_true",
                        help="don't use zero padding but use rather costly spline interpolation of the original swipe algorithm for frequency domain interpolation  (def: %(default)s)" )
    parser.add_argument("--music_mode", action="store_true",
                        help="use swipe setup for analysis of music instead of speech  (def: %(default)s)" )
    parser.add_argument("-q","--quiet",  default=False, action="store_true",
                        help="suppress display of processing steps (Def: False)" )

    parser.add_argument("-z","--zero", dest="set_zero", default=False, action="store_true",
                        help="set frames with insufficient harmonicity to zero instead of removing them   (Def: False)" )
    parser.add_argument("-t","--harm_thresh", dest="harmonicity_threshold", default=None, type=float,
                        help="if set to value in [0, 1] all f0-frames with harmonicity below threshold "
                        "will be either removed or set to zero (see -z) (Def: None)" )
    parser.add_argument("--format", choices=["csv", "ssv", "tsv"], default=None,
                        help="store data in csv (comma), ssv (space), or tsv (tab) separated format (def: %(default)s)" )
    args = parser.parse_args()

    if not args.quiet:
        print(f"development version of MBExWN_NVoc directory detected. We have adapted the path to use it", file=sys.stderr)

    calc_f0_swipe(args.infile, args.outfile, f0min=args.f0min, f0max=args.f0max,
                  harmonicity_threshold=args.harmonicity_threshold,
                  remove_low_harm_f0=(not args.set_zero),
                  time_step= args.time_step,
                  ana_freq_limit=args.ana_fmax, use_spline_interp=args.use_spline,
                  music_mode=args.music_mode, verbose=(not args.quiet),
                  txt_format= args.format)


