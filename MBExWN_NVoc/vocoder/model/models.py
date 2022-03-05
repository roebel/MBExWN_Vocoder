# coding: utf-8
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  model creator


from .wavegen_1d import PaNWaveNet

def create_model(hparams, training_config, preprocess_config,
                 name="myWaveGlow", quiet=False, use_tf25_compatible_implementation=None, **kwargs):

    mr_mode = False
    if use_tf25_compatible_implementation is None:
        use_tf25_compatible_implementation = hparams.get("use_tf25_compatible_implementation", None)
    tf25_compatibility_switch = {}
    if use_tf25_compatible_implementation is not None:
        tf25_compatibility_switch = {"use_tf25_compatible_implementation": use_tf25_compatible_implementation}

    if "mbexwn_config" in hparams:
        myWaveGlow = PaNWaveNet(model_config=hparams['mbexwn_config'],
                                training_config=training_config,
                                preprocess_config=preprocess_config,
                                quiet=quiet,
                                **tf25_compatibility_switch,
                                name=name)
        mr_mode = False
    else:
        raise NotImplementedError(f"create_model::error::unkown config requested {list(hparams.keys())}. Only mbexwn_config is currently supported.")
    return myWaveGlow, mr_mode
