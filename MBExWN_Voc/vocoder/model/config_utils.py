
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  utils for managing config files

import os, sys
import tensorflow as tf
import numpy as np
from copy import deepcopy
import re
import yaml
import ast
import io

_type_map = {
    "tf.float32": tf.float32,
    "tf.float16" : tf.float16,
    "np.float32": np.float32,
    "np.float16": np.float16,
    "None": None,
}
_inverse_type_map = {
    tf.float32 : "tf.float32",
    tf.float16 : "tf.float16",
    np.float32 : "np.float32",
    np.float16 : "np.float16",
}


def _fill_format(vv, config_base_dir=None):
    """
    replace environment variables and component configs
    """
    if isinstance(vv, str):
        if vv in _type_map :
            vv = _type_map[vv]
        else:
            if  "$" in vv:
                vv= os.path.expandvars(vv)
            if  "~" in vv:
                vv= os.path.expanduser(vv)
            vs=vv.strip()
            vsmapped = re.sub("<@CONFIG_DIR@/(.*)>$", f"{config_base_dir}/\\1", vs)
            #print(f"vs <{vs}> vsmapped <{vsmapped}>")
            if vs != vsmapped:
                file_name, *keys = vsmapped.split(":")
                vv = read_config(file_name, config_base_dir=config_base_dir)
                for kk in keys:
                    vv = vv[kk]

    elif isinstance(vv, dict):
        for kk, _vv in vv.items():
            vv[kk] = _fill_format(_vv, config_base_dir=config_base_dir)
    elif isinstance(vv, list):
        for ie in range(len(vv)):
            vv[ie] = _fill_format(vv[ie], config_base_dir=config_base_dir)
    return vv

def _fix_config(config):
    """
    fill self references in strings in config entries
    """
    for kk,vv in config.items():
        if isinstance(vv, dict):
            config[kk] = _fix_config(vv)
        elif isinstance(vv, (np.dtype, tf.DType)) and vv in _inverse_type_map :
            config[kk] = _inverse_type_map[vv]

    return config

int_pat=re.compile("^ *[0-9]+ *$")
# attention this pattern does match the empty string
float_pat = re.compile('^ *(-?\d*(?:\.\d*)?(?:[eE][-+]?\d+)?) *$')

list_dict_quoted_strings_pat = re.compile("^ *[{\[\"\'].*[}\]\"\'] *$")

def _auto_convert_str(par_string):
    ppss = par_string.lower()

    if ppss == "none" or ppss == "null":
        return None
    elif ppss == "true":
        return True
    elif ppss == "false":
        return False
    elif int_pat.match(par_string) or float_pat.match(par_string) or list_dict_quoted_strings_pat.match(par_string):
        try:
            return ast.literal_eval(par_string.strip())
        except Exception :
            print(f"error evaluating python expression <{par_string}>", file=sys.stderr)
            raise

    return par_string


_assign_regexp=re.compile("([^:]+)([:]?)")
_index_regexp=re.compile("^ *\[ *([-]?[0-9]+) *\] *$")

def set_sub_dict_multi(sub_config, plist, config_base_dir=None):

    #print(f"entering set_sub_dict_multi {sub_config} with {plist}")
    while plist:
        (ent, mrk), *plist = plist
        if (not ent) and (not mrk):
            continue

        #print(f"sub process: {ent} {mrk} -> {plist}")
        if "=" in ent:
            key, val_str = ent.split("=")
            if isinstance(sub_config, list) and re.match(_index_regexp, key):
                key = int(re.match(_index_regexp, key).group(1))
                if key < 0:
                    key=len(sub_config)
                while len(sub_config) <= key:
                    sub_config.append(None)
            elif key not in sub_config:
                raise RuntimeError(f"multi_modify_config::error:: you try to change the config key '{key}' "
                                   f"that does not exist in {sub_config}")
            try:
                if val_str.startswith("<"):
                    # here we have to read the file specified up to the first ":" and retrieve the content
                    # specified by the dictionary traversal
                    if not val_str.endswith(">"):
                        raise RuntimeError(f"multi_modify_config::error:: erroneous file specification {val_str}")
                    file, *dict_path = val_str[1:-1].split(":")
                    file = file.replace("@CONFIG_DIR@", config_base_dir)
                    other_config = read_config(file)
                    for dd in dict_path:
                        other_config= other_config[dd]

                    sub_config[key] = deepcopy(other_config)
                else:
                    sub_config[key] = _auto_convert_str(val_str)
            except Exception:
                print(f"set_sub_dict_multi::error evaluating {ent} {key} {val_str}")
                raise
            
        elif re.match(_index_regexp, ent):
            ind = int(re.match(_index_regexp, ent).group(1))
            plist = set_sub_dict_multi(sub_config[ind], plist, config_base_dir=config_base_dir)
        else:
            if not plist:
                raise RuntimeError(f"multi_modify_config::error:: you access a key {ent} in {sub_config}  without any further args. "
                                   f"Did you miss to place an equal sign?")

            if ent not in sub_config:
                raise RuntimeError(f"multi_modify_config::error:: key {ent} does not exist in {sub_config}")

            plist = set_sub_dict_multi(sub_config[ent], plist, config_base_dir=config_base_dir)

    return plist

def _find_sub_entries(carg):
    """
    find sub entries in a nested cargs config definition
    do not cut within dict/list/fieio assignments
    """
    dict_level = 0
    list_level = 0
    redirect_level = 0
    parsed_args = []
    last_start = 0
    for ind, cc in enumerate(carg):
        if cc == "{":
            dict_level += 1
        elif cc == "}":
            dict_level -= 1
            if dict_level < 0:
                raise RuntimeError("cargs string contains unbalanced dictionary entries")
        elif cc == "[":
            list_level += 1
        elif cc == "]":
            list_level -= 1
            if list_level < 0:
                raise RuntimeError("cargs string contains unbalanced list entries")
        elif cc == "<":
            redirect_level += 1
        elif cc == ">":
            redirect_level -= 1
            if redirect_level < 0:
                raise RuntimeError("cargs string contains file redirection entry")
        elif cc == ":" and dict_level == 0 and list_level == 0 and redirect_level == 0:
            parsed_args.append((carg[last_start:ind], cc))
            last_start = ind + 1

    if last_start < ind:
        parsed_args.append((carg[last_start:], ""))
    return parsed_args

def modify_config(hparams, config_args, config_base_dir=None):
    """
    set arbitrary entries in config dict according to list of strings that specify the entries in
    form of ':' separated strings
    """

    if config_args is None:
        return hparams

    if config_base_dir is None:
        config_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

    for carg in config_args:
        #print(f"processing {carg}")
        if ("=" in carg):
            #plist = re.findall(_assign_regexp, carg)
            plist = _find_sub_entries(carg)
            #print(f"plist: {plist}")
            plist = set_sub_dict_multi(hparams, plist, config_base_dir)
            if plist:
                raise RuntimeError(f"modify_config::error:: carg processing error in '{carg}' set_sub_dict_multi returns non empty {plist}")
        else:
            cal = carg.split(':')
            if len(cal) < 2:
                raise RuntimeError(f"modify_config::error:: parsing carg '{carg}' does not produce exploitable config modification entry.")

            sub_config = hparams
            for cc in cal[:-2]:
                sub_config = sub_config[cc]
            # set entry converting to the type of the existing dict entry
            if cal[-2] not in sub_config:
                raise RuntimeError(f"modify_config::error:: you try to change the config key '{cal[-2]}' that does not exist in {sub_config}")
            #print(sub_config, sub_config[cal[-2]], cal[-1], _auto_concert_str(cal[-1]))
            sub_config[cal[-2]] = _auto_convert_str(cal[-1])
            #print(sub_config, sub_config[cal[-2]], cal[-1], _auto_concert_str(cal[-1]))

    return hparams


def get_list_parameter(val, n_elements, n_repeater=None, n_repeater_list=None):
    """
    create a list of parameters with n_elements form a list or a scalar

    In case val is a scalar or has len 1 the value is duplicated n_elements times
    in case n_repeater is a scalar and val is of len n_elements//n_repeater each of its elements will be repeated n_repeater times
    in case n_repeater_list is not None, each element i of val  will be repeated n_repeater_list[i] times
    in case the resulting list is longer than n_elements the exceeding elements are discarded

    """

    try:
        val_list = val[:]
    except TypeError:
        val_list = [val]

    if (n_repeater is not None) and n_repeater_list:
        raise RuntimeError(f"get_list_parameter::error::only one of the arguments n_repeater {n_repeater} "
                           f"and n_repeater_list {n_repeater_list} is allowed to be present")

    if len(val_list) == 1:
        val_list = val_list * n_elements
    elif (n_repeater is not None) and (len(val_list) * n_repeater < n_elements + n_repeater):
        val_list = [vv for vv in val_list for _ in range(n_repeater)]
        # repeat last element if necessary
        if len(val_list) < n_elements:
             val_list = val_list + [val_list[-1] for _ in range(n_elements - len(val_list))]
        # cut list if  last element if necessary
        val_list = val_list[:n_elements]
    elif (n_repeater_list is not None) and (np.sum(n_repeater) == n_elements):
        _tmp_list = []
        for vv, rr in zip(val_list, n_repeater_list):
            _tmp_list += [vv] * rr
        val_list = _tmp_list
    elif len(val_list) != n_elements:
        raise RuntimeError(f"training_utils::error:: cannot construct list of {n_elements} "
                           f"from {val} with n_repeater {n_repeater} n_repeater_list {n_repeater_list}")
    return val_list

def _fill_defaults(config):
    tmp_config = deepcopy(config)
    for kk, vv in tmp_config.items():
        if kk == "__defaults__":
            for dk, dv in tmp_config[kk].items():
                if dk not in config:
                    config[dk] = dv
            config.pop("__defaults__")
        elif isinstance(vv, dict):
            _fill_defaults(config[kk])
        elif isinstance(vv, list):
            # check whether there is a list entry with a single key named __defaults__
            # if there is: remove it form the output config and use its members
            # to complete all the remaiing list entries in the output config
            list_entry_defaults = None
            defaults_index = None
            for ie, ve in enumerate(vv):
                if isinstance(ve, dict) and (len(ve) == 1) and ("__defaults__" in ve.keys()):
                    list_entry_defaults = deepcopy(ve["__defaults__"])
                    if defaults_index is not None:
                        raise RuntimeError(f"read_config::error::multiple __defaults__ entries in list {vv}")
                    else:
                        defaults_index = ie

            # now remove the defaults element from the final list
            # and complete all other list elements with all the keys available in the __defaults__ element
            if defaults_index is not None:
                del config[kk][defaults_index]

                for le in config[kk]:
                    if not isinstance(le, dict):
                        raise RuntimeError(f"read_config::error::cannot use default values from {list_entry_defaults} for list entries that are not dicts {le}")
                    for dk, dv in list_entry_defaults.items():
                        if dk not in le:
                            le[dk] = dv

            # now in case we have a list of dicts we search and replace further __defaults__ in these dicts
            for ie, ve in enumerate(config[kk]):
                if isinstance(ve, dict) :
                    _fill_defaults(ve)

    return

def read_config(config_file, config_base_dir=None):
    """
    read and fill config with self references
    """

    if config_base_dir is None:
        config_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

    if  isinstance(config_file, (list, tuple)):
        config_files= config_file
    else:
        config_files = [config_file]

    # concatenate all config files into a Stream
    config_io = io.StringIO()
    for file in config_files:
        with open(file, "r") as fi:
            config_io.write(fi.read())
    # read from the start
    config_io.seek(0)
    config = yaml.safe_load(config_io)
    for kk, vv in config.items():
        config[kk] = _fill_format(vv, config_base_dir=config_base_dir)

    _fill_defaults(config)
    return config

def dump_config(config_file, config):
    """
    write config into yaml file
    """
    if os.path.dirname(config_file) and not os.path.exists(os.path.dirname(config_file)):
        os.makedirs(os.path.dirname(config_file), exist_ok=True)

    config =  _fix_config(deepcopy(config))
    with open(config_file, "w") as fo:
        config = yaml.safe_dump(config, fo)
    return config

def _check_config_dict_implementation(config_dict, config_name,
                                      required_keys, optional_keys, obsolete_keys):
    possible_keys = required_keys + optional_keys + obsolete_keys
    unsupported = []
    obsolete_found = []
    for kk in config_dict:
        if kk not in possible_keys:
            unsupported.append(kk)
        if kk in obsolete_keys:
            obsolete_found.append(kk)
        while kk in required_keys:
            required_keys.remove(kk)

    if unsupported:
        raise RuntimeError(f"{config_name}::error: the following top level entries in your {config_name} are not supported {unsupported}")
    if required_keys:
        raise RuntimeError(f"{config_name}::error: the following required entries in your {config_name} are not provided {required_keys}")
    if obsolete_found:
        print(f"obsolete parameters {obsolete_found} detected in {config_name}, please update yor config")

def check_config_dict(hparams):
    optional_keys = ["waveglow_config", "waveflow_config", "waveglow_mr_config", "waveflow_config", "comp_melgen_config",
                     "melgen_config", "acmelgen_config", "wavegen2D_config", "waveAE1D_config", "wavegan_config",
                     "preprocess_config", "training_config", "checkpoint_config", "pickle_config",
                     "path_config", "waveSGen_config", "pangen_config", "panm_config", "sde_config", "mbwpan_config",
                     "mbexwn_config", "aliases", "use_tf25_compatible_implementation"]
    required_keys = ["preprocess_config", "training_config", "checkpoint_config", "pickle_config" ]
    obsolete_keys = []

    _check_config_dict_implementation(hparams, "config_dict", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)

def check_preprocess_config(preprocess_config):
    required_keys = ["sample_rate", "segment_length", "hop_size", "mel_channels", "fft_size", "fmin", "fmax"]
    optional_keys = [ "use_centered_STFT", "win_size", "random_filter_length",
                      "random_filter_amp", "random_mult", "random_mult_max_amp",
                      "band_pass", "norm_mel", "mel_amp_scale", "lin_amp_scale",
                      "lin_amp_off", "use_max_limit"]
    obsolete_keys = ["include_sub_sampled"]

    _check_config_dict_implementation(preprocess_config, "preprocess_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)


def check_training_config(training_config):
    required_keys = ["epochs", "epoch_size", "train_batch_size", "ftype", "optimizer"]
    optional_keys = [ "learning_rate", "reduce_on_plateau", "read_files_max_length_s",
                      "pca_num_steps", "add_speaker_id", "batch_cache_config", "file_reader_procs",
                      "batch_creat_procs", "batch_min_cache_perc", "batch_cache_perc",
                      "total_loss_debug_thresh", "batch_debug_dir",
                      "init_inv1x1_with_pca", "output_soft_thresh_fac", "dither_level",
                      "spect_loss_config",  "stage", "all_optimizers_start",
                      "pretrain_activations_target", "pretrain_activations_max_iters",
                      "pretrain_activations_to_rmse", "pretrain_activations_lr", "TD_loss_weight", "TD_loss_win_len"]
    obsolete_keys = ["buffer_size", "mixed_precision", "mell_loss_weight", "cpdl_loss_weight","seed"]

    _check_config_dict_implementation(training_config, "training_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)


def check_spect_loss_config(spect_loss_config):
    required_keys = [ "win_size", "hop_size"]
    optional_keys = [  "spect_loss_weight", "spect_loss_schedule",
                       "loss_type", "fft_over", "mell_loss_weight", "MCCTP_loss_weight", "PP_loss_weight", "BC_loss_weight",
                       "MCCT_loss_weight", "MCCTP_loss_weight", "MCCTS_loss_weight", "NLL_loss_weight",
                       "MODSPEC_loss_weight",
                       "NPOW_loss_weight", "NLL_min_std",
                       "PP_band_width_Hz", "PP_segment_size_s", "PP_loss_method", "remove_mean_hz",
                       "BC_segment_size_s", "BC_loss_method", "BC_max_off_Hz", "MODSPEC_loss_method",
                       "masking_noise_std", "rel_masking_noise_atten_db",
                       "low_band_extra_weight", "low_band_extra_weight_limit_Hz", "low_band_extra_weight_transition_Hz",
                      "MCC_segment_size_s", "MCC_pad_size_s", "lin_amp_off", "rel_lin_amp_off", "magnitude_compression",
                       "use_lin_amp_off_for_mc", "spect_error_gain"  ]
    obsolete_keys = ["cpdl_loss_weight", "MCCT_segment_size_s","MCCT_pad_size_s", "RIC_loss_weight",
                     "RIC_num_filters", "RIC_band_width", "RIC_seed", "RIC_segment_size_s"]

    _check_config_dict_implementation(spect_loss_config, "spect_loss_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)


def check_checkpoint_config(checkpoint_config):
    required_keys = ["max_to_keep", "log_dir", "checkpoint_dir", "save_model_every", "show_progress_every",
                     "save_audio_every"]
    optional_keys = ["keep_audio_every", "store_model_every"]
    obsolete_keys = []

    _check_config_dict_implementation(checkpoint_config, "checkpoint_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)


def check_wavenet_config(wavenet_config):
    required_keys = ["n_layers", "n_channels", "kernel_size"]
    optional_keys = ["enable_weight_norm",  "enable_equalized_lr",
                     "dilation_rate_step", "max_log2_dilation_rate", "activation", "return_activations",
                     "weight_schedule", "disabled", "loss_thresh", "rel_masking_noise_atten_db"]
    obsolete_keys = ["use_weight_norm", "n_in_channels"]

    _check_config_dict_implementation(wavenet_config, "wavenet_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)

def check_wavegen1D_config(wavegen1D_config):
    required_keys = ["wavenet_config", "n_flows", "n_group", "sigma"]
    optional_keys = ["n_up_every", "spect_upsampling"]
    obsolete_keys = []

    _check_config_dict_implementation(wavegen1D_config, "wavegen1D_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)

def check_waveGAN_config(wavegan_config):
    required_keys = ["generator_config", "discriminator_config"]
    optional_keys = ["adv_loss_weight", "fc_loss_weight", "adv_loss_schedule", "fc_loss_schedule",
                     "disc_cond_config", "disable_discriminator"]
    obsolete_keys = []

    _check_config_dict_implementation(wavegan_config, "wavegan_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)


def check_waveSGen_config(waveSGen_config):
    required_keys = ["wavenet_config", "n_stages", "n_wn_per_stage", "n_group", "sigma"]
    optional_keys = ["spect_upsampling",  "normalize_rms_from_mell", "max_norm_fact",
                     "normalize_rms_num_smooth_iters"]
    obsolete_keys = []

    _check_config_dict_implementation(waveSGen_config, "waveSGen_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)

def check_discriminator_config(wavenet_discriminator_config):
    required_keys = ["loss_method", "n_group", "activation"]
    optional_keys = ["wavenet_config", "convnet_config", "spectnet_config", "disc_list",
                     "rel_masking_noise_atten_db", "masking_noise_std", "sig_low_pass",
                     "remove_mean_hz", "mask_dc_rms", "mel_max_amp_offset", "normalize_rms_from_mell", "max_norm_fact",
                     "normalize_rms_num_smooth_iters", "fc_act_norm", "n_group_overlap", "pre_emphasis_filter_coefficients"]
    obsolete_keys = []

    _check_config_dict_implementation(wavenet_discriminator_config, "wavenet_discriminator_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)


def check_waveglow_config(waveglow_config):
    required_keys = ["wavenet_config", "n_flows", "n_group", "sigma", "n_early_every", "n_early_size"]
    optional_keys = ["use_svd_for_permutation", "pre_white"]
    obsolete_keys = []

    _check_config_dict_implementation(waveglow_config, "waveglow_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)

    check_wavenet_config(waveglow_config["wavenet_config"])
    return

def check_waveglow_mr_config(waveglow_mr_config):
    required_keys = ["wavenet_config", "n_flows", "n_group", "sigma", "n_early_every"]
    optional_keys = ["learned_permutation", "use_svd_for_permutation",
                     "remove_mean_hz", "mell_loss_weight", "normalize_rms_from_mell", "max_norm_fact",
                     "normalize_rms_num_smooth_iters", "wrong_spect_conditioning", "old_conditioning_layer",
                     "inv1x1_predict_kernels", "upsamling", "upsampling", "pre_white"]
    obsolete_keys = ["include_band_limited"]

    _check_config_dict_implementation(waveglow_mr_config, "waveglow_mr_config", required_keys=required_keys,
                                      optional_keys=optional_keys, obsolete_keys=obsolete_keys)

    check_wavenet_config(waveglow_mr_config["wavenet_config"])
    return

def get_model_config(hparams):
    """
    get the mayor model configuration for the config dict.
    """
    if 'waveglow_mr_config' in hparams:
        return hparams['waveglow_mr_config']
    elif "waveflow_config" in hparams:
        return hparams['waveflow_config']
    elif "waveAE1D_config" in hparams:
        return hparams['waveAE1D_config']
    elif "melgen_config" in hparams:
        return hparams['melgen_config']
    elif "pangen_config" in hparams:
        return hparams['pangen_config']
    elif "panm_config" in hparams:
        return hparams['panm_config']
    elif "mbwpan_config" in hparams:
        return hparams['mbwpan_config']
    elif "mbexwn_config" in hparams:
        return hparams['mbexwn_config']
    elif "sde_config" in hparams:
        return hparams['sde_config']
    elif "acmelgen_config" in hparams:
        return hparams['acmelgen_config']
    elif "comp_melgen_config" in hparams:
        return hparams['comp_melgen_config']
    elif "waveSGen_config" in hparams:
        return hparams['waveSGen_config']
    elif "wavegen2D_config" in hparams:
        return hparams['wavegen2D_config']
    elif "wavegan_config" in hparams:
        return hparams["wavegan_config"]
    elif "waveglow_config" in hparams:
        return hparams['waveglow_config']
    else:
        raise RuntimeError(f"get_model_config::error::no known model config found in hparams with keys: {list(hparams.keys())}")


