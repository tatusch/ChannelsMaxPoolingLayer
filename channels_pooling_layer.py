from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
import numpy as np

class ChannelsMaxPooling(Layer):
    '''Channels MaxPooling Layer for 2D Inputs.

        # Arguments
            pool_size: int
             -> Size of Filter
            stride: int
             -> Steps for Sliding Window
        # Input shape
            4D tensor with shape:
            (samples, channels, rows, cols) if dim_ordering='th'
            or 4D tensor with shape:
            (samples, rows, cols, channels) if dim_ordering='tf'.
        # Output shape
            4D tensor with shape:
            (samples, ((channels - pool_size) / stride) + 1, rows, cols) if dim_ordering='th'
            or 4D tensor with shape:
            (samples, rows, cols, ((channels - pool_size) / stride) + 1) if dim_ordering='tf'
    '''

    def __init__(self, pool_size=64, stride=1, **kwargs):
        self.pool_size = pool_size
        self.stride = stride
        self.data_format = K.image_dim_ordering()
        super(ChannelMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'th':
            self.nb_channels = input_shape[1]
        elif self.data_format == 'tf':
            self.nb_channels = input_shape[3]

    def call(self, input):
        if self.data_format == 'th':
            return K.pool2d(input, (1, self.pool_size), strides=(1, self.stride), pool_mode='max', data_format="channels_last")
        elif self.data_format == 'tf':
            return K.pool2d(input, (1, self.pool_size), strides=(1, self.stride), pool_mode='max', data_format="channels_first")

    def compute_output_shape(self, input_shape):
        if self.data_format == 'tf':
            output_shape = (input_shape[0], input_shape[1], input_shape[2], ((input_shape[3] - self.pool_size) / self.stride) + 1)
        elif self.data_format == 'th':
            output_shape = (input_shape[0], ((input_shape[1] - self.pool_size) / self.stride) + 1, input_shape[2], input_shape[3])
        return output_shape

