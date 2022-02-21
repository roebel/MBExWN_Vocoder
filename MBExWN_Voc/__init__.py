import os, sys
from typing import Dict, List, Tuple, Any, Union

mbexwn_version=(1, 0, 2)

# this should be a directory containing domains (SING, SPEECH, VOICE) each containing
# lists of models for that domain.
# ATTENTION: These lists will be used to selected a matching model using the first come first serves basis.
# Therfore, the lists should be ordered such that the first entry is the best one.
# The first entry will be used as default model if only the domain specifier is given.
_mel_inv_models = {
    "SING" : [
        "MBExWN_ICASSP_V57_SING_IMPORTmod_WNCHA320_DISC32_MAXDISC1024_DSC2_SPL1_PTACT0.1_PCPQMF1PM1_PSSTFT1_MLW150_4_WT_FCLW0_ADLW0.1_E0_V2_24kHz",
    ],
    "SPEECH": [
        "MBExWN_ICASSP_V57_MINV_IMPORTf0_WNCHA320_DISC32_MDISC1024_DSC2_SPL1_PTACT0.1_PCPQMF1PM1_PSSTFT1_MLW150_4_WT_FCLW0_ADLW0.1_E0_V2_24kHz",
        "MBExWN_GAN_V45_WHITE_WNCH180_WNG1_ADLW01SC2_MINV_NMEL1_KS3_CKS3_PADVFalse_PSEO3_WNC180GRP1_DISC32_MAXDISC1024_DSC2_SPL1_WTwhpuls_FCLW0_ADLW0.1_24kHz",
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