# coding: utf-8
#
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  TF2C base model

import os, sys
from typing import Union, List, Tuple
import numpy as np
import abc

import tensorflow as tf

class TF2C_BaseModel(tf.keras.Model, metaclass=abc.ABCMeta):
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
        self.built = True

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


class TF2C_BasePretrainableModel(TF2C_BaseModel, metaclass=abc.ABCMeta):
    """
    keras layer interface extension for supporting pretraining
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pretrain_activations = False
        self._built_input_shape = None

    @property
    def pretrain_activations(self):
        return self._pretrain_activations

    def _filter_pretrainable_layers(self, ll, check_pretrain_activation=False):
        if hasattr(ll, "pretrain_activations"):
           return (not check_pretrain_activation) or ll.pretrain_activations
        return False

    @pretrain_activations.setter
    def pretrain_activations(self, onoff):
        self._pretrain_activations = onoff
        for ll in self._flatten(recursive=False, predicate=self._filter_pretrainable_layers):
            ll.pretrain_activations = onoff
            self._ms_activations = None

    @property
    def ms_activations(self):
        ms_activations = []
        for ll in self._flatten(recursive=True,
                                predicate=lambda ll: hasattr(ll, "ms_activations")):
            if ll.ms_activations is not None:
                ms_activations.append(ll.ms_activations)

        return ms_activations

    @property
    def trainable_weights(self):
        if self.pretrain_activations:
            # return [vv for vv in super().trainable_weights if hasattr(vv, "is_pre_trainable")]

            trainable_weights = []
            for ll in self._flatten(recursive=False,
                                    predicate=lambda ll: self._filter_pretrainable_layers(ll, check_pretrain_activation=True)):

                    trainable_weights.extend(ll.trainable_weights)
            return trainable_weights
        else:
            return super().trainable_weights


