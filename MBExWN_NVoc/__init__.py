# coding: utf-8


# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
# 

import os, sys
from typing import Dict, List, Tuple, Any, Union

mbexwn_version=(1, 2, 2)

# this should be a directory containing domains (SING, SPEECH, VOICE) each containing
# lists of models for that domain.
# ATTENTION: These lists will be used to selected a matching model using the first come first serves basis.
# The first entry will be used as default model if only the domain specifier is given.
# Therefore, the lists should be ordered such that the first entry is the best one.
_mel_inv_models = {
    "SING" : [
        "MBExWN_SIIConv_V71g_SING_IMP0_IMPORTmod_MCFG0_WNCHA320_DCHA32_1024_DPTACT0_ADLW0.1_GMCFG5_24kHz",
    ],
    "SPEECH": [
        "MBExWN_SIIConv_V71g_SPEECH_IMP0_IMPORTmod_MCFG0_WNCHA320_DCHA32_1024_DPTACT0_ADLW0.1_GMCFG5_24kHz",
    ],
    "VOICE": [
        "MBExWN_SIIConv_V71g_VOICE2_WNCHA340_IMP0_WNCHA340_IMPORTmod_MCFG0_WNCHA340_DCHA32_1024_DPTACT0_ADLW0.1_GMCFG0_24kHz",
    ]
}

def list_models(voice_type:Union[str, None]=None):
    """
    get list of all mel inverter models that are available for a voice class.
    Supported voice classes are singing and speech

    :return: list of model names (basename of model file without extension)
    """
    import copy
    if voice_type is None:
        return copy.deepcopy(_mel_inv_models)

    return copy.deepcopy(_mel_inv_models)


def get_config_file( model_id_or_path, verbose = False):
    from pathlib import Path

    if os.path.exists(model_id_or_path):
        model_dir = model_id_or_path
    else:
        for  kk, val in list_models().items():
            for kk, ll in list_models().items():
                for    md in ll:
                    if model_id_or_path in f"{kk}/{md}":
                        model_dir = Path(__file__).absolute().parent / "models" / md
                        break

    config_file = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"error::loading config file from {config_file}"
        )
    return config_file
