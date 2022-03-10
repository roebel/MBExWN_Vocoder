# coding: utf-8
# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  TF2/Keras layers implementing various operations

import os, sys
import numpy as np
from typing import Union, List, Tuple

import tensorflow as tf
from tensorflow.keras import layers
from .tf2c_base_layer import TF2C_BaseLayer


    
def _init_linear_interpolator_weights(shape, dtype):
    ww = np.zeros((1, 2, 1, shape[-1]), np.float64)

    ww[0, 0, 0, :] = (shape[-1]-np.arange(shape[-1]))/shape[-1]
    ww[0, 1, 0, :] = np.arange(shape[-1])/shape[-1]

    # duplicate the same kernel for all input channels
    w2 = np.repeat(ww, shape[2], axis=2)
    return tf.convert_to_tensor(w2, dtype=dtype)


class TF2C_LinInterpLayer(TF2C_BaseLayer):
    """
    Tensorflow 2.0 implementation of a linear interpolation layer
    """

    def __init__(self, upsampling_factor, num_pad_end=0, drop_last=False,
                 **kwargs):
        """
        A layer performing linear interpolation along axis 1. Given upsamling factor U
        an input of shape BxTxC will be transformed into
        an output of shape:

        B x (T-1)*U + D x C

        Where D = int(not drop_last)
        If num_pad_end = P  > 0 the axis 1 will padded at the end with num_pad_end copies of the last value before the liner interpolation is performed.
        So the output shape in this case will be

         B x (T+P-1)*U + D x C


        :param upsampling_factor: linear interpolation with this upsamling factor,
        :type upsampling_factor: int

        :param num_pad_end: Perform duplications of the last value by this many times (Def:0)
        :type num_pad_end: int

        :param drop_last: Drop the last value of the output on the interpolated axis (Def: False). Using
              drop last  = True creates an output shape that is exactly the same as the shape produced with
              Conv1DUpDownSample (for the same upsamling factor)

        :type drop_last: bool



        """

        if "trainable" not in kwargs:
            super().__init__(trainable=False, **kwargs)
        else:
            super().__init__( **kwargs)

        self.upsampling_factor = upsampling_factor
        self.num_pad_end = num_pad_end
        self.drop_last = drop_last
        if drop_last:
            self.last_size = 0
        else:
            self.last_size = 1

        self.single_channel_mode = None
        self._built_input_shape = None
        if self.trainable:
            raise RuntimeError("LinInterpLayer::error::this layer being a fixed interpolation layer it cannot be trained!")

    def build_or_compute_output_shape(self, input_shape, do_build=False) -> Union[None, Tuple]:
        if do_build:
            self.single_channel_mode =  (input_shape[-1] == 1)
            self.kernel = self.add_weight(name="kernel",
                                          shape=[1, 2, input_shape[-1], self.upsampling_factor],
                                          initializer=_init_linear_interpolator_weights,
                                          dtype=self.dtype,
                                          trainable=False)
        else:
            if input_shape[1] is not None:
                return (input_shape[0], (input_shape[1] + self.num_pad_end - 1) * self.upsampling_factor + self.last_size) + input_shape[2:]
            return input_shape


    def call(self, inputs):
        # debug = False
        #print("LinInter input", inputs.shape)
        x = inputs
        if self.num_pad_end > 0:
            x = tf.concat((x, tf.tile(x[:, -1:], (1,self.num_pad_end,1))), axis=1)
        res = tf.nn.depthwise_conv2d(tf.expand_dims(x, axis=1),
                                     filter=self.kernel, strides=[1,1,1,1], padding="SAME",
                                     data_format ="NHWC", dilations=None)
        #print('LinearInterplp after depthwise_conv2d',res.shape)
        if  (not self.single_channel_mode):
            res = tf.reshape(res, (x.shape[0],
                                   x.shape[1],
                                   x.shape[2], self.upsampling_factor))
            res =  tf.transpose(res, (0,1,3,2))
        res = tf.reshape(res, (x.shape[0],
                               x.shape[1] * self.upsampling_factor,
                               x.shape[2]))

        #print('LinearInterplp after reshape',res.shape)
        res = res[:,:(x.shape[1] - 1) * self.upsampling_factor + self.last_size, :]
        #print('LinearInterplp after cut',res.shape)
        return res

    def get_config(self):
        config = super().get_config()
        config.update(upsampling_factor = self.upsampling_factor)
        config.update(num_pad_end = self.num_pad_end)
        config.update(drop_last = self.drop_last)






