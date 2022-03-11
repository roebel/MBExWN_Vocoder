# coding: utf-8


# AUTHORS
#    A.Roebel
# COPYRIGHT
#    Copyright(c) 2021 - 2022 IRCAM, Roebel
#
#  TF2C base layer supporting pretraining

from typing import List, Union, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .tf2_components.layers.support_layers import TF2C_LinInterpLayer as LinInterpLayer
from .tf2_components.layers.conv_layers import TF2C_Conv1DWeightNorm, TF2C_Conv1DUpDownSample

from .tf2_components.layers.tf2c_base_layer import TF2C_BasePretrainableLayer

class ActivationLayer(layers.Layer):
    def __init__(self, activation_function=None, alpha=0.2, name="ActivationLayer", **kwargs):

        super().__init__(name=name, **kwargs)
        self.activation_function = activation_function if not activation_function else activation_function.lower()
        if not activation_function or (activation_function == "linear"):
            self.act_fun  = None
        elif activation_function == "tanh":
            self.act_fun = tf.keras.activations.tanh
        elif activation_function == "sigmoid":
            self.act_fun = tf.keras.activations.sigmoid
        elif activation_function == "soft_sign":
            self.act_fun = tf.keras.activations.softsign
        elif activation_function == "elu":
            self.act_fun = tf.keras.activations.elu
        elif activation_function == "selu":
            self.act_fun = tf.keras.activations.selu
        elif activation_function == "soft_sigmoid":
            self.act_fun = self._soft_sigmoid
        elif activation_function == "soft_sqrt":
            self.act_fun = self._soft_sqrt
        elif activation_function == "exp":
            self.act_fun = tf.keras.activations.exponential
        elif activation_function == "relu":
            self.act_fun = tf.keras.layers.ReLU()
        elif activation_function == "leaky_relu":
            self.act_fun = layers.LeakyReLU(alpha=alpha)
        elif activation_function == "prelu":
            self.act_fun = layers.PReLU(alpha_initializer=tf.keras.initializers.constant(alpha), **kwargs)
        else:
            raise RuntimeError(
                f"ActivationLayer::error::unkown activation selected {activation_function}, "
                f"only tanh, sigmoid, soft_sign, soft_sqrt, relu, leaky_relu, and prelu are supported")
        self._built_input_shape = None

    def build(self, input_shape):
        self._built_input_shape = input_shape
        #print(f"Conv1DUpDownSample-{self.name}::build:: input_shape", input_shape)
        if (self.activation_function is not None) and self.activation_function.endswith("relu"):
            self.act_fun.build(input_shape)

        super().build(input_shape)

    @property
    def output_shape(self):
        return self._built_input_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def _exp(self, x):

        """
        apply exponential
        x -> exp(x)
        """

        return tf.exp(x)


    def _soft_sqrt(self, x):

        """
        soft symmetric sqrt

        x -> x/(1+ sqrt(abs(x))
        """

        return x /( 1 + tf.sqrt(tf.abs(x)))

    def _soft_sigmoid(self, x):

        """
        soft approximate sigmoid

        x -> 0.5 + 0.5 * x /( 1 + tf.abs(x))
        """

        return 0.5 + 0.5 * x /( 1 + tf.abs(x))

    def call(self, inputs, **kwargs):
        if self.act_fun:
            return self.act_fun(inputs)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(activation_function=self.activation_function)
        return config



# ## WaveNet Implementation
class WaveNetAE(TF2C_BasePretrainableLayer):
    """
    Wavenet layer
    WaveNet convolution causality is controlled via padding parameter; Use either CAUSAL or SAME.
    """

    def __init__(self,  n_channels=256,
                 n_layers=12, kernel_size=3, n_out_channels = None, n_ch_groups=1,
                 dilation_rate_step = 1, max_log2_dilation_rate = None,
                 use_weight_norm=False, use_equalized_lr=False,
                 activation="gtu", return_activations = (),
                 padding="SAME", disable_conditioning=False,
                 cond_kernel_size: int = 1,
                 pre_cond_layer_channels: Union[Tuple[int], List[int], None] = None,
                 cond_conv_upsampling: Union[int,None] = None,
                 cond_lin_upsampling: int = 1,
                 use_tf25_compatible_implementation :bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        assert (kernel_size % 2 == 1)
        assert (n_channels % 2 == 0)

        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_ch_groups = n_ch_groups
        self.kernel_size = kernel_size
        self.dilation_rate_step = dilation_rate_step
        self.max_log2_dilation_rate = max_log2_dilation_rate
        self.conv_layers = []
        self.normalisation_layers = []
        self.res_skip_layers = []
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        self.use_equalized_lr = use_equalized_lr
        self.return_activations = return_activations
        self.return_activations_mask = []
        self.return_start_activation = 0 in self.return_activations
        self.padding = padding
        self.cond_kernel_size = cond_kernel_size
        self.cond_conv_upsampling = cond_conv_upsampling
        self.cond_lin_upsampling = cond_lin_upsampling
        if activation not in ["gtu", "glu", "gfu", "gsu"]:
            raise RuntimeError(f"WaveNetAE::error::unsupported wavenet activation {activation} selected. "
                               f"For gated units please select one of gtu, gfu, gsu, or glu.")

        if n_out_channels is None:
            raise RuntimeError("WaveNetAE::error::n_out_channels parameter is required")
        self.n_out_channels = n_out_channels
        self._built_input_shape = None

        if self.n_channels % self.n_ch_groups:
            raise RuntimeError(f"WaveNetAE::error::n_channels parameter {self.n_channels} has to be a multiple of chanel groups parameter {self.n_ch_groups}")

        self.n_grp_channels = self.n_channels // self.n_ch_groups
        self.use_tf25_compatible_implementation = use_tf25_compatible_implementation,
        if use_tf25_compatible_implementation:
            Conv1D_Layer = TF2C_Conv1DWeightNorm
            Conv1D_UpDownSample = TF2C_Conv1DUpDownSample
        else:
            raise NotImplementedError(
                "WaveNetAE::error::implmentations not selecting use_tf25_compatible_implementation are not supported")

        self.start = Conv1D_Layer(filters=self.n_channels,
                                      kernel_size=1,
                                      dtype=self.dtype,
                                      use_weight_norm=self.use_weight_norm,
                                      use_equalized_lr=use_equalized_lr,
                                      name="start")

        self.end = Conv1D_Layer( filters=self.n_out_channels,
                                     kernel_size=1,
                                     use_weight_norm=self.use_weight_norm,
                                     use_equalized_lr=use_equalized_lr,
                                     dtype=self.dtype,
                                     name="end")
        self.pre_cond_layer_channels = pre_cond_layer_channels if pre_cond_layer_channels is not None else []
        self.disable_conditioning = disable_conditioning
        self.pre_cond_layers = []
        for ii, chans in enumerate(self.pre_cond_layer_channels):
            self.pre_cond_layers.append(
                Conv1D_Layer(filters=chans,
                             kernel_size=cond_kernel_size,
                             dtype=self.dtype,
                             use_weight_norm=self.use_weight_norm,
                             use_equalized_lr=use_equalized_lr,
                             padding=self.padding,
                             name=f"precond_{ii}")
            )

        if disable_conditioning:
            self.cond_layer = None
        elif cond_conv_upsampling is None:
            self.cond_layer = Conv1D_Layer(filters=2 * self.n_channels * self.n_layers,
                                               kernel_size=cond_kernel_size,
                                               dtype=self.dtype,
                                               use_weight_norm=self.use_weight_norm,
                                               use_equalized_lr=use_equalized_lr,
                                               padding=self.padding,
                                               name="cond_")
        else:
            self.cond_layer = Conv1D_UpDownSample(filters=2 * self.n_channels,
                                                 kernel_size=cond_kernel_size,
                                                 dtype=self.dtype,
                                                 use_weight_norm=self.use_weight_norm,
                                                 use_equalized_lr=use_equalized_lr,
                                                 factor=cond_conv_upsampling,
                                                 up_sample=True,
                                                 use_checkerboard_free_init= True,
                                                 padding=self.padding,
                                                 name="cond_")
            self.cond_lin_upsampling_layer = LinInterpLayer(upsampling_factor=cond_lin_upsampling,
                                                            num_pad_end=1, drop_last=True,
                                                            name="wn_cond_LinUpLayer")
        self.conv_layers = []
        for index in range(self.n_layers):
            if max_log2_dilation_rate is not None:
                dilation_rate = 2 ** (int(index // self.dilation_rate_step) % self.max_log2_dilation_rate)
            else:
                dilation_rate = 2 ** int(index // self.dilation_rate_step)

            for i_grp in range(self.n_ch_groups):
                in_layer = Conv1D_Layer(filters=2 * self.n_grp_channels,
                                            kernel_size=self.kernel_size,
                                            dilation_rate=dilation_rate,
                                            padding=self.padding,
                                            dtype=self.dtype,
                                            use_weight_norm=self.use_weight_norm,
                                            use_equalized_lr=use_equalized_lr,
                                            name=f"conv1D_{index}" + (f"g{i_grp}" if i_grp else ""))


                self.conv_layers.append(in_layer)
                if index < self.n_layers - 1:
                    res_skip_channels = 2 * self.n_grp_channels
                else:
                    res_skip_channels = self.n_grp_channels

                res_skip_layer = Conv1D_Layer(filters=res_skip_channels,
                                                  kernel_size=1,
                                                  dtype=self.dtype,
                                                  use_weight_norm=self.use_weight_norm,
                                                  use_equalized_lr=use_equalized_lr,
                                                  name=f"res_skip_{index}" + (f"g{i_grp}" if i_grp else ""))

                self.res_skip_layers.append(res_skip_layer)

            # the zero's layer is the start layer, the dilated conv layers start counting with 1 then
            if self.return_activations and ((index + 1) in self.return_activations):
                self.return_activations_mask.append(True)
            else:
                self.return_activations_mask.append(False)


    @property
    def pretrainable_weights(self) -> List[tf.Tensor]:
        return []


    def call(self, inputs, **_):
        """
         upsample conditioning tensor and
         calculate WaveNet output
        """
        audio_0, spect = inputs

        started = [self.start(audio_0)]

        if self.cond_layer is not None:
            cond_layers_io = spect
            for ll in self.pre_cond_layers:
                cond_layers_io = ll(cond_layers_io)

            if self.cond_conv_upsampling is not None:
                cond_layers_io = self.cond_layer(cond_layers_io)
                cond_layers = tf.split(self.cond_lin_upsampling_layer(cond_layers_io), self.n_ch_groups, axis=-1)
            else:
                cond_layers = tf.split(self.cond_layer(cond_layers_io), self.n_layers * self.n_ch_groups, axis=-1)
        else:
            cond_layers = tf.zeros((self.n_layers * self.n_ch_groups,), dtype=tf.float32)

        if self.return_activations:
            if self.return_start_activation:
                activations = started
            else:
                activations = []

        output = []

        if self.n_ch_groups >= 1:
            started = tf.split(started[0], num_or_size_splits=self.n_ch_groups, axis=-1)
        for index, ret_mask in zip(range(self.n_layers), self.return_activations_mask):
            for i_grp in range(self.n_ch_groups):
                in_layered = self.conv_layers[index*self.n_ch_groups+i_grp](started[i_grp])
                if self.cond_conv_upsampling is not None:
                    half_act, half_sigmoid = tf.split(in_layered + cond_layers[i_grp], 2, axis=-1)
                else:
                    half_act, half_sigmoid = tf.split(in_layered  + cond_layers[index*self.n_ch_groups+i_grp], 2, axis=-1)
                if self.activation == "gtu":
                    half_act = tf.nn.tanh(half_act)
                elif self.activation == "gfu":
                    half_act = half_act/(1+ tf.abs(half_act))
                elif self.activation == "gsu":
                    half_act = half_act/(1+ tf.sqrt(tf.abs(half_act)))

                half_sigmoid = tf.nn.sigmoid(half_sigmoid)

                activated = half_act * half_sigmoid
                if ret_mask:
                    activations.append(activated)
                res_skip_activation = self.res_skip_layers[index*self.n_ch_groups+i_grp](activated)

                if index < (self.n_layers - 1):
                    res_skip_activation_0, skip_activation = tf.split(res_skip_activation, 2, axis=-1)
                    started[i_grp] += res_skip_activation_0
                else:
                    skip_activation = res_skip_activation

                if index == 0:
                    output.append(skip_activation)
                else:
                    output[i_grp] += skip_activation

        if self.n_ch_groups > 1:
            output = self.end(tf.concat(output, axis=-1))
        else:
            output = self.end(output[0])

        #tf.summary.histogram('AE_postoutput', output, buckets=20)
        if self.return_activations:
            return (output, *activations)

        return output

    def build_or_compute_output_shape(self, input_shape, do_build=False) -> Union[None, Tuple]:
        audio_0_shape, spect_shape = input_shape
        if do_build:
            self.start.build(audio_0_shape)
        started_shape = self.start.compute_output_shape(audio_0_shape)

        if (not self.disable_conditioning):
            cond_io_shape= spect_shape
            for ll in self.pre_cond_layers:
                if do_build:
                    ll.build(cond_io_shape)
                cond_io_shape = ll.compute_output_shape(cond_io_shape)
            if do_build:
                self.cond_layer.build(cond_io_shape)
            cond_io_shape =  self.cond_layer.compute_output_shape(cond_io_shape)
            if self.cond_conv_upsampling:
                if do_build:
                    self.cond_lin_upsampling_layer.build(cond_io_shape)
                cond_io_shape = self.cond_lin_upsampling_layer.compute_output_shape(cond_io_shape)

        if self.n_ch_groups > 1:
            started_shape = started_shape[:-1] + (started_shape[-1] // self.n_ch_groups,)
        for index in range(self.n_layers):
            for i_grp in range(self.n_ch_groups):
                if do_build:
                    self.conv_layers[index*self.n_ch_groups +i_grp].build(started_shape)
                in_layered_shape = self.conv_layers[index*self.n_ch_groups +i_grp].compute_output_shape(started_shape)
                activated_shape = in_layered_shape[:-1] + (in_layered_shape[-1]//2,)

                if do_build:
                    self.res_skip_layers[index*self.n_ch_groups +i_grp].build(activated_shape)
                res_skip_activation_shape = self.res_skip_layers[index*self.n_ch_groups +i_grp].compute_output_shape(activated_shape)

        if self.n_ch_groups > 1:
            res_skip_activation_shape = res_skip_activation_shape[:-1] + (res_skip_activation_shape[-1]*self.n_ch_groups,)

        if do_build:
            self.end.build(res_skip_activation_shape)
        else:
            return self.end.compute_output_shape(res_skip_activation_shape)

    def summary(self, print_fn=print, indent=""):
        audio_shape, mel_shape = self._built_input_shape

        if (not self.disable_conditioning):
            print_fn(indent + "conditioning preparation")
            cond_shape = mel_shape
            for ll in self.pre_cond_layers:
                cond_shape = ll.compute_output_shape(cond_shape)
                num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
                print_fn(indent+f"  {ll.name:28s} -> {str(cond_shape):20s} ## {num_weights:d}")

            ll = self.cond_layer
            cond_shape = ll.compute_output_shape(cond_shape)
            num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
            print_fn(indent+f"  {ll.name:28s} -> {str(cond_shape):20s} ## {num_weights:d}")
            if self.cond_conv_upsampling:
                ll = self.cond_lin_upsampling_layer
                cond_shape = ll.compute_output_shape(cond_shape)
                num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
                print_fn(indent+f"  {ll.name:28s} -> {str(cond_shape):20s} ## {num_weights:d}")

        print_fn(indent + "WaveNet")
        ll = self.start
        started_shape = ll.compute_output_shape(audio_shape)
        num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
        print_fn(indent + f"  {ll.name:28s} -> {str(started_shape):20s} ## {num_weights:d}")

        if self.n_ch_groups > 1:
            started_shape = started_shape[:-1] + (started_shape[-1] // self.n_ch_groups,)
        for index in range(self.n_layers):
            for i_grp in range(self.n_ch_groups):
                ll = self.conv_layers[index*self.n_ch_groups +i_grp]
                in_layered_shape = ll.compute_output_shape(started_shape)
                activated_shape = in_layered_shape[:-1] + (in_layered_shape[-1]//2,)
                num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
                print_fn(indent + f"  {ll.name:28s} -> {str(activated_shape):20s} ## {num_weights:d}")
                ll = self.res_skip_layers[index*self.n_ch_groups +i_grp]
                res_skip_activation_shape = ll.compute_output_shape(activated_shape)
                num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
                print_fn(indent + f"  {ll.name:28s} -> {str(res_skip_activation_shape):20s} ## {num_weights:d}")

        if self.n_ch_groups > 1:
            res_skip_activation_shape = res_skip_activation_shape[:-1] + (res_skip_activation_shape[-1]*self.n_ch_groups,)


        ll =  self.end
        out_shape = ll.compute_output_shape(res_skip_activation_shape)
        num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
        print_fn(indent + f"  {ll.name:28s} -> {str(out_shape):20s} ## {num_weights:d}")


    def get_config(self):
        config = super().get_config()
        config.update(n_channels=self.n_channels)
        config.update(n_out_channels=self.n_out_channels)
        config.update(n_layers=self.n_layers)
        config.update(n_ch_groups=self.n_ch_groups)
        config.update(kernel_size=self.kernel_size)
        config.update(dilation_rate_step=self.dilation_rate_step)
        config.update(max_log2_dilation_rate=self.max_log2_dilation_rate)
        config.update(use_weight_norm=self.use_weight_norm)
        config.update(use_equalized_lr=self.use_equalized_lr)
        config.update(activation=self.activation)
        config.update(disable_conditioning=self.disable_conditioning)
        return config


# ## WaveNet
class WaveNetAEBlock(TF2C_BasePretrainableLayer):
    """
    Wavenet block containing a WaveNet layer followed by upsampling
    """

    def __init__(self, n_out_channels, n_channels=256,
                 n_layers=12, kernel_size=3, dilation_rate_step=1, max_log2_dilation_rate=None,
                 up_sample=None, up_down_factor=1,
                 use_weight_norm = True, activation="gtu",
                 use_equalized_lr = False, padding="SAME",
                 disable_conditioning = False, n_ch_groups=1,
                 cond_kernel_size: int = 1,
                 cond_conv_upsampling: Union[int, None] = None,
                 cond_lin_upsampling: int = 1,
                 pre_cond_layer_channels: Union[Tuple[int], List[int], None] = None,
                 use_tf25_compatible_implementation: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.up_sample = up_sample
        self.up_down_factor = up_down_factor
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_ch_groups = n_ch_groups
        self.kernel_size = kernel_size
        self.dilation_rate_step = dilation_rate_step
        self.max_log2_dilation_rate = max_log2_dilation_rate
        self.n_out_channels = n_out_channels
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        self.disable_conditioning = disable_conditioning
        self.cond_conv_upsampling = cond_conv_upsampling
        self.cond_lin_upsampling = cond_lin_upsampling
        self.pre_cond_layer_channels = pre_cond_layer_channels
        self.cond_kernel_size = cond_kernel_size
        self.use_tf25_compatible_implementation = use_tf25_compatible_implementation,
        if use_tf25_compatible_implementation:
            Conv1D_UpDownSample = TF2C_Conv1DUpDownSample
        else:
            raise NotImplementedError(
                "generate_subnet_from_specs::error::implmentations not selecting use_tf25_compatible_implementation are not supported")

        self.wavenet = WaveNetAE(n_channels=n_channels,
                                 n_layers=n_layers,
                                 kernel_size=kernel_size,
                                 dilation_rate_step=dilation_rate_step,
                                 max_log2_dilation_rate=max_log2_dilation_rate,
                                 n_out_channels=n_out_channels,
                                 use_weight_norm=self.use_weight_norm,
                                 activation=self.activation,
                                 use_equalized_lr=use_equalized_lr,
                                 dtype=self.dtype,
                                 n_ch_groups=n_ch_groups,
                                 padding=padding,
                                 disable_conditioning=disable_conditioning,
                                 cond_kernel_size = cond_kernel_size,
                                 cond_conv_upsampling = cond_conv_upsampling,
                                 cond_lin_upsampling= cond_lin_upsampling,
                                 pre_cond_layer_channels=pre_cond_layer_channels,
                                 use_tf25_compatible_implementation=use_tf25_compatible_implementation,
                                 name=self.name+"_WNBlock_WN")

        self.up_down_sample = None
        if self.up_sample is not None :
            self.up_down_sample = Conv1D_UpDownSample(n_out_channels,
                                                     kernel_size=3,
                                                     padding=padding,
                                                     up_sample=self.up_sample,
                                                     factor=self.up_down_factor,
                                                     name=self.name+f"_WNBlock_UP_{self.up_down_factor}")


    @property
    def pretrainable_weights(self) -> List[tf.Tensor]:
        return []

    def build_or_compute_output_shape(self, input_shape, do_build=False) -> Union[None, Tuple]:
        if do_build:
            self.wavenet.build(input_shape)
        curr_shape = self.wavenet.compute_output_shape(input_shape)
        if self.up_down_sample:
            if do_build:
                self.up_down_sample.build(curr_shape)
            curr_shape = self.up_down_sample.compute_output_shape(curr_shape)

        if not do_build :
            return curr_shape

    def summary(self, print_fn=print, indent=""):
        audio_shape, mel_shape = self._built_input_shape
        print_fn(indent+"==========================")
        print_fn(indent+f"{self.name:28s} input -> aud: {audio_shape}, mel: {mel_shape}")

        self.wavenet.summary(print_fn=print_fn, indent=indent+"   ")
        tot_num_weights = 0
        ll = self.wavenet
        curr_shape = (audio_shape, mel_shape)
        curr_shape = ll.compute_output_shape(curr_shape)
        num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
        print_fn(indent+f"  {ll.name:28s} -> {str(curr_shape):20s} ## {num_weights:d}")

        if self.up_down_sample:
            ll = self.up_down_sample
            curr_shape = ll.compute_output_shape(curr_shape)
            num_weights = int(np.sum([np.prod(w.shape) for w in ll.trainable_weights]))
            print_fn(indent+f"  {ll.name:28s} -> {str(curr_shape):20s} ## {num_weights:d}")

        tot_num_weights += num_weights
        print_fn(indent+f"  sub net params -> {tot_num_weights}")

    def call(self, inputs, **_):
        """

        """
        audio, spect = inputs
        wavenet_output = self.wavenet((audio, spect))
        if self.up_down_sample:
            wavenet_output = self.up_down_sample(wavenet_output)

        return wavenet_output

    def get_config(self):
        config = super().get_config()
        config.update(up_sample=self.up_sample)
        config.update(n_channels=self.n_channels)
        config.update(n_layers=self.n_layers)
        config.update(n_ch_groups=self.n_ch_groups)
        config.update(wavenet_n_out_channels=self.wavenet_n_out_channels)
        config.update(kernel_size=self.kernel_size)
        config.update(dilation_rate_step=self.dilation_rate_step)
        config.update(use_weight_norm=self.use_weight_norm)
        config.update(activation=self.activation)
        config.update(disable_conditioning=self.disable_conditioning)
        return config


