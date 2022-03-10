# coding: utf-8
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  Convolutional Layers Supporting WeightNorm and pretraining.

import os, sys
import numpy as np
from typing import Union, List, Tuple, Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations

from .tf2c_base_layer import TF2C_BaseLayer, TF2C_BasePretrainableLayer


# ## Custom Implementation of Conv1D with weight normalization
class TF2C_Conv1DWeightNorm(TF2C_BasePretrainableLayer):

    def __init__(self, filters, kernel_size, strides:int =1,
                 use_weight_norm=True,
                 use_equalized_lr=False, no_cb_for_up_fac=0, **kwargs):
        """
        Conv1D extension supporting
          use_weight_norm: if set to true the kernel is split into a gain and a normalized kernel product.
               By default the kernel has nom2 == 1. If use_equalized_lr is set together with use_weight_norm
               the the kernel will be normalized to have stddev 1. This normalization is ensured throughout training
               and to achieve an equivalent expressivity an additional gain factor g is introduced.

          use_equalized_lr: if set to True the kernel will be normalized during intialization to have stddev == 1
               To preserve equiavlent intialization the activation is produced by means of maulitplyaing with a gain that
               is set equal to the stddev of the initial gains produced by the underlying Conv1D layer.
               If only use_equalized_lr is set then the gain facor is not trained during standard training preserving the
               input utput mapping of the selected kernel initializer.

          The gain factor can be pre trainined
          no_cb_for_up_fac: force all kernels to be identical over channel dimension for depth to time folding of factor with given factor,
            this prevents checkerboard artifacts after initialisation during upsampling as described in
             Aitken, arXiv.1707, Checkerboard artifact free sub-pixel convolution
             This is useful notably for subpixel based upsamling
        """

        supported_layer_kwargs = ["trainable", "dtype", "dynamic", "batch_size"]
        layer_kwargs = {}

        for kw in kwargs:
            if kw in supported_layer_kwargs:
                layer_kwargs[kw] = kwargs[kw]

        self.no_cb_for_up_fac = no_cb_for_up_fac
        self.use_weight_norm = use_weight_norm
        self.use_equalized_lr = use_equalized_lr

        # we handle activation in Conv1DWeightNorm
        activation= kwargs.pop("activation", None)
        self.conv1d_layer = layers.Conv1D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            **kwargs)
        self.activation = activations.get(activation)
        super().__init__(
            name=self.conv1d_layer.name+"_base",
            **layer_kwargs)

    def build_or_compute_output_shape(self, input_shape, do_build=False) -> Union[None, Tuple]:

        if do_build:
            self.conv1d_layer.build(input_shape)
            if self.no_cb_for_up_fac:
                self.conv1d_layer.kernel.assign(tf.reshape(tf.tile(tf.reduce_mean(tf.reshape(self.conv1d_layer.kernel,
                                                                self.conv1d_layer.kernel.shape[:2]+[self.no_cb_for_up_fac]
                                                                +[self.conv1d_layer.kernel.shape[2]//self.no_cb_for_up_fac]), axis=-2, keepdims=True),
                                      (1,1, self.no_cb_for_up_fac, 1)), self.conv1d_layer.kernel.shape))
            if self.use_weight_norm:
                self.kernel_norm_axes = [0, 1]
                # v now holds the variable that has been added as trainable weight
                # the name self.conv1d_layer.kernel will how eve no loger refer to this variable.
                # note that the variable name of self. v remains kernel:0
                self.v = self.conv1d_layer.kernel

                if not self.use_equalized_lr:
                    new_g = tf.linalg.norm(tf.reshape(self.conv1d_layer.kernel, (-1, self.conv1d_layer.filters)), axis=0)
                else:
                    # here we can calculate the stddev over all kernels as they are all initialized the same anyway
                    new_g_val = tf.sqrt(tf.reduce_mean(tf.square(self.conv1d_layer.kernel)))
                    new_g = tf.ones(self.conv1d_layer.filters) * new_g_val

                    # attention do not calculate selv.v from kernel above, this would render v as a tensor and not as a trainable variable
                    self.v.assign(self.conv1d_layer.kernel/ new_g)

                self.conv1d_layer.kernel = None
                self.g = self.add_weight(
                    name="g",
                    shape=self.conv1d_layer.filters,
                    initializer=tf.keras.initializers.get("zero"),
                    dtype=self.dtype,
                    trainable=True)
                self.g.assign(new_g)
                self.g.is_pre_trainable = True

            elif self.use_equalized_lr:

                ini_std = tf.sqrt(tf.reduce_mean(tf.square(self.conv1d_layer.kernel)))
                self.g = self.add_weight(
                    name="g",
                    shape=self.conv1d_layer.filters,
                    initializer=tf.keras.initializers.Constant(np.float(ini_std)),
                    dtype=self.dtype,
                    trainable=True)
                self.g.is_pre_trainable = True
                self.conv1d_layer.kernel.assign(self.conv1d_layer.kernel/ini_std)
        else:
            return self.conv1d_layer.compute_output_shape(input_shape=input_shape)

    @property
    def pretrainable_weights(self):
        if self.use_weight_norm or self.use_equalized_lr:
            if self.conv1d_layer.use_bias:
                return [self.g, self.conv1d_layer.bias]
            else:
                return [self.g]
        else :
            print(f"TF2C_Conv1DWeightNorm({self.name})::error::pretraining can only be performed if either"
                  f"when weight_norm={self.use_weight_norm} or equalized_lr={self.use_equalized_lr} is True!",
                  file=sys.stderr)
            return []


    def call(self, inputs, **kwargs):

        if self.use_equalized_lr and (not self.use_weight_norm):
            act = self.g * self.conv1d_layer.call(inputs)
            if self.pretrain_activations:
                ma = tf.reduce_mean(act, axis=range(1,len(act.shape)), keepdims=True)
                self._ms_activations = (
                    tf.reshape(ma, (ma.shape[0],)),
                    # tf.sqrt(tf.reduce_mean(tf.square(act - ma), axis=range(1, len(act.shape))))
                    tf.reduce_mean(tf.abs(act - ma), axis=range(1, len(act.shape)))
                )

            if self.activation is not None:
                return self.activation(act)
            return act

        if self.use_weight_norm:
            if self.use_equalized_lr:
                self.conv1d_layer.kernel = self.g * self.v / tf.sqrt (tf.reduce_mean(tf.square(self.v), axis=self.kernel_norm_axes, keepdims=True))
            else:
                self.conv1d_layer.kernel = self.g * tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes)
        act = self.conv1d_layer.call(inputs)
        if self.pretrain_activations:
            ma = tf.reduce_mean(act, axis=range(1, len(act.shape)), keepdims=True)
            self._ms_activations = (
                tf.reshape(ma, (ma.shape[0],)),
                # tf.sqrt(tf.reduce_mean(tf.square(act - ma), axis=range(1, len(act.shape))))
                tf.reduce_mean(tf.abs(act - ma), axis=range(1, len(act.shape)))
            )

        if self.activation is not None:
            return self.activation(act)
        return act


    def get_config(self):
        config = self.conv1d_layer.get_config()
        config.update(use_weight_norm=self.use_weight_norm)
        config.update(use_equalized_lr=self.use_equalized_lr)
        config.update(no_cb_for_up_fac=self.no_cb_for_up_fac)
        config.update(activation=activations.serialize(self.activation))
        return config

# ## Custom Implementation of Conv1D with up/downsampling
class TF2C_Conv1DUpDownSample(TF2C_Conv1DWeightNorm):

    def __init__(self, filters, kernel_size=3, up_sample=None, factor=2,
                 use_checkerboard_free_init=False, **kwargs):
        """
        filters always specifies the number of features in the output array.

        Internally a larger or smaller number of features may be used
        if up_sample is None the depth/channel dimension and time dimensions are kept unchanged
        (standard  conv1d operation)
        if up_sample is True the depth/channel dimension of the internal conv1d will be filters * factor
        which will then be unfolded by means of depth to time mapping creating the shape transformation
           B x T x Cin  -> B x T*factor x filters
        if up_sample is False the depth/channel dimension of the internal conv1d will be filters // factor
        and then increased by means of folding time into depth dimension creating the shape transformation
           B x T x Cin  -> B x T//factor x filters

        use_checkerboard_free_init: force all kernels to be identical over channel dimension, this prevents
        checkerboard artifacts after initialisation during upsampling as described in
        Aitken, arXiv.1707, Checkerboard artifact free sub-pixel convolution

        """

        self.up_sample = up_sample
        self.factor = factor
        self.out_filters = filters
        self.down_sample = (up_sample is not None) and (not up_sample)
        self.use_checkerboard_free_init = use_checkerboard_free_init
        if self.use_checkerboard_free_init and (not self.up_sample):
            raise RuntimeError("TF2C_Conv1DUpDownSample::error::use_checkerboard_free_init can only be used when upsampling is requested")

        super().__init__(
            filters = (filters * factor) if up_sample else ((filters // factor) if self.down_sample else filters),
            kernel_size=kernel_size,
            #activation="linear",
            no_cb_for_up_fac= self.factor if (use_checkerboard_free_init and self.up_sample) else 0,
            **kwargs)

        if up_sample is not None:
            if int(factor) != factor:
                raise RuntimeError(f"TF2C_Conv1DUpDownSample::error:: factor {factor} has to be an integer")

            #print(f"TF2C_Conv1DUpDownSample-{self.name}:: inchannels {filters} -> out filters {filters*2 if up_sample else filters//2} up_sample {up_sample}")
            if (not up_sample) and (factor > kernel_size):
                print(f"TF2C_Conv1DUpDownSample::warning::kernel_size {kernel_size} > DownSampling factor {factor}")

            if self.down_sample and factor * (filters // factor) != filters:
                raise RuntimeError(f"TF2C_Conv1DUpDownSample::error:: filters {filters} is not a multiple of factor {factor}")

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape) -> Tuple:
        curr_shape = super().compute_output_shape(input_shape)
        if self.up_sample:
            if curr_shape[1] is not None:
                return curr_shape[0], curr_shape[1] * self.factor, self.out_filters
            else:
                return curr_shape[0], curr_shape[1], self.out_filters
        elif self.down_sample:
            if curr_shape[1] is not None:
                return curr_shape[0], curr_shape[1] // self.factor, self.out_filters
            else:
                return curr_shape[0], curr_shape[1], self.out_filters

        return curr_shape


    @property
    def output_shape(self):
        if self._built_input_shape is not None:
            return TF2C_Conv1DUpDownSample.compute_output_shape(self._built_input_shape)

    def call(self, inputs, **kwargs):
        res = super().call(inputs)
        if self.up_sample:
            #print(f"TF2C_Conv1DUpDownSample-{self.name}::call:: upsample weights_shape", self.kernel.shape,
            #      "inputs shape", inputs.shape, "res.shape", res.shape)
            return tf.reshape(res, (res.shape[0], res.shape[1] * self.factor, -1))
        elif self.down_sample:
            #print(f"TF2C_Conv1DUpDownSample-{self.name}::call:: downsample weights_shape", self.kernel.shape,
            #      "inputs shape", inputs.shape, "res.shape", res.shape)
            return tf.reshape(res, (res.shape[0], -1, res.shape[2] * self.factor))
        else:
            return res

    def get_config(self):
        config = super().get_config()
        config.update(up_sample=self.up_sample)
        config.update(factor=self.factor)
        return config



