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
        config = {}
        config["initial"] = self.initial
        config["type"] = self.type
        config["name"] = self.name
        return config


class PiecewiseConstantSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial, name, segments= None):
        """
        a piece wise constant parameter scheduler with constant extraxpolation
        defined by an initial value and a list of pairs of number of steps and target values.

        initial  = 2.
        segments = [[10, 3.], [9, 5.]]

        creates value 2 for all steps below 10 and 3 for all values 10 < tep < 19. And 5 > 19

        """
        self.initial = tf.cast(initial, tf.float32)
        self.segments = tf.convert_to_tensor(segments, tf.float32) if segments else tf.convert_to_tensor([], tf.float32)
        self.name = name

    def __str__(self):
        return f"piecewise constant scheduler for {self.name}: initial={self.initial}" + ("" if not self.segments.shape[0] else f" segments = {self.segments}")


    def __call__(self, step : tf.int64, **kwargs) -> tf.float32:
        # here we need a python if such that the recursion compiled into a graph will have appropriate depth
        if not self.segments.shape[0]:
            return self.initial

        def loop(segments, curr_target, step):
            # here we need a python if such that the recursion compiled into a graph will have appropriate depth
            if segments.shape[0]:
                steps  = tf.cast(segments[0][0], tf.float64)
                next_target = segments[0][1]
                segments = segments[1:]
                #tf.print(step, steps, curr_target, segments)
                ret = tf.cond(
                    step < steps,
                    lambda: curr_target,
                    lambda: tf.cond(tf.greater(tf.size(segments), 0),
                                    lambda: loop(segments, next_target, step - steps),
                                    lambda: next_target
                                    ))
                #tf.print(step, "ret", ret)
            else:
                return curr_target
            return ret

        return loop(self.segments, self.initial, tf.cast(step, tf.float64))

    def get_config(self):
        config = {"intial": self.initial,
                  "segments" : self.segments}
        return config

