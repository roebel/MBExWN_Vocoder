# coding: utf-8
#
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  TF2C base layers

import os, sys
from typing import Union, List, Tuple
import numpy as np
import abc

import tensorflow as tf

class TF2C_BaseLayer(tf.keras.layers.Layer, metaclass=abc.ABCMeta):
    """
    keras layer interface extension simplifying a coherent implementation of outut_shape, compute_output_shape
    and build for custom layers.
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._built_input_shape = None

    @property
    def output_shape(self):
        return self.build_or_compute_output_shape(self._built_input_shape)

    def compute_output_shape(self, input_shape):
        return self.build_or_compute_output_shape(input_shape)

    def build(self, input_shape):
        self.prepare_build_or_compute_output_shape(input_shape)
        self.build_or_compute_output_shape(input_shape, do_build=True)
        super().build(input_shape=input_shape)

    @abc.abstractmethod
    def build_or_compute_output_shape(self, input_shape, do_build=False) -> Union[None, Tuple]:
        """ An abstract method for building the model or for computing the output shape

        This method should never becalled directly, it will be called from the build() and compute_output_shape methods
        or the output_shape property.
        """

    def prepare_build_or_compute_output_shape(self, input_shape) -> None:
        """ test whether model is already built and set internal built_input_shape member"""
        if self._built_input_shape is not None:
            print(f"{self.name}:: error you try to rebuild a layer that was already built: inshape {input_shape} "
                  f"stored  {self._built_input_shape}", flush=True)
            raise RuntimeError("already built")
        self._built_input_shape = input_shape


class TF2C_BasePretrainableLayer(TF2C_BaseLayer, metaclass=abc.ABCMeta):
    """
    keras layer interface extension for supporting pretraining
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pretrain_activations = False
        self._built_input_shape = None
        self._ms_activations = None

    def _filter_pretrainable_layers(self, ll, check_pretrain_activation=False):
        if hasattr(ll, "pretrain_activations"):
           return (not check_pretrain_activation) or ll.pretrain_activations
        return False

    @property
    def pretrain_activations(self):
        return self._pretrain_activations

    @pretrain_activations.setter
    def pretrain_activations(self, onoff):
        self._pretrain_activations = onoff
        for ll in self._flatten(recursive=False, predicate=self._filter_pretrainable_layers):
            ll.pretrain_activations = onoff
            self._ms_activations = None

    @property
    def ms_activations(self):
        if self._ms_activations is not None:
            return self._ms_activations + (self.name,)
        else:
            return None

    @property
    def trainable_weights(self):
        if self.pretrain_activations:
            trainable_weights = self.pretrainable_weights
            for ll in self._flatten(recursive=False,
                                    predicate=lambda ll: self._filter_pretrainable_layers(ll,
                                                                                          check_pretrain_activation=True)):
                trainable_weights.extend(ll.trainable_weights)
            return trainable_weights
        else:
            return super().trainable_weights

    @abc.abstractproperty
    def pretrainable_weights(self) -> List[tf.Tensor]:
        """
        should return the weights that should be adapted for pretraining.
        This function should only return the weights that will not be found in sublayers.

        You can return an empty list if there are no pretrainable weights in the layer
        """
