# coding: utf-8

# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#
import tensorflow as tf

import numpy as np
import os, sys
from copy import deepcopy



class ParamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial, name, type="constant", quiet=False, **kwargs):
        self.type = type

        if type == "constant":
            self.scheduler = PiecewiseConstantSchedule(initial=initial,  name=name, **kwargs)
        elif type == "linear":
            self.scheduler = PiecewiseLinearSchedule(initial=initial, name=name, **kwargs)
        elif type == "exponential":
            self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial,
                                                                            name=name, **kwargs)
        else:
            raise RuntimeError(f"PiecewiseParamSchedule::error:: type {type} not supported. Please choose either 'constant' or 'linear'")

        if not quiet:
            if self.type == "exponential":
                print(f"initialized exponential scheduler for parameter {self.scheduler.name} with initial {self.scheduler.initial_learning_rate} and args {kwargs}",
                      file=sys.stderr, flush=True)
            else:
                print("initialized", str(self.scheduler), file=sys.stderr, flush=True)

    def __call__(self, step : tf.int64)  -> tf.float32:
        ret = self.scheduler(step=step)
        # tf.print("scheduler return", ret)
        return ret

    def get_config(self):
        config = self.scheduler.get_config()
        if "initial_learning_rate" in config:
            config.update(initial=config["initial_learning_rate"])
            del config["initial_learning_rate"]
        config.update(type= self.type)
        return config



