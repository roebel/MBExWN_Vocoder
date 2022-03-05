# coding: utf-8
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  custom layers

import os, sys
import numpy as np
from typing import Union, List, Tuple
import tensorflow as tf
from tensorflow.keras import layers
from .tf2_components.layers.tf2c_base_layer import TF2C_BaseLayer, TF2C_BasePretrainableLayer
# attention these imports are used in other modules - don't delete them even if they appear unused here
from .tf2_components.layers.conv_layers import  TF2C_Conv1DWeightNorm

class TFPad1d(TF2C_BaseLayer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type="REFLECT", name="PadLayer", **kwargs):
        """Initialize TFPad1d module.
        Args:
            padding_size (int, Tuple[int])
            padding_type (str) ("CONSTANT", "REFLECT", "SYMMETRIC", or EDGE. Default is "REFLECT")
        """
        super().__init__(name=name, **kwargs)
        supported_types = ["CONSTANT", "REFLECT", "SYMMETRIC", "EDGE"]
        try:
            self.padding_size = padding_size[0], padding_size[1]
        except IndexError:
            self.padding_size = padding_size, padding_size
        self.padding_type = padding_type.upper()
        if self.padding_type not in supported_types:
            raise RuntimeError(f"TFPad1d::error:: padding_type {self.padding_type} is not supported. Use one of {supported_types}")

    def build_or_compute_output_shape(self, input_shape, do_build=False):
        if do_build:
            self._built_input_shape = input_shape
        else:
            return (
                input_shape[0],
                None if (input_shape[1] is None) else (input_shape[1]+ self.padding_size[0]+ self.padding_size[1]),
                input_shape[2]
            )

    def call(self, inputs):
        """Calculate forward propagation.
        Args:
            inputs (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        if self.padding_type == "EDGE":
            return tf.concat(
                (
                    tf.tile(
                        inputs[:,:1],
                        (1,self.padding_size[0], 1)
                    ),
                    inputs,
                    tf.tile(
                        inputs[:,-1:],
                        (1,self.padding_size[1], 1)
                    )
                ),
                axis=1
            )
        return tf.pad(inputs,
                      paddings=[[0, 0], [self.padding_size[0], self.padding_size[1]], [0, 0]],
                      mode=self.padding_type)

    def get_config(self):
        config = super().get_config()
        config.update(padding_size=self.padding_size)
        config.update(padding_type=self.padding_type)
        return config

