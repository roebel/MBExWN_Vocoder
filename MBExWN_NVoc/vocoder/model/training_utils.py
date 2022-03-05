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

# this is the scheduler base class that does not support any scheduling.
class ParamSchedule(object):
    def __init__(self, initial, name, type="constant", quiet=False, **kwargs):
        self.type = type
        self.initial = initial
        self.name = name

    def __call__(self, step : tf.int64)  -> tf.float32:
        return self.initial

    def get_config(self):
        config["initial"] = self.initial
        config["type"] = self.type
        config["name"] = self.name
        return config



