# ChannelsMaxPoolingLayer Implementation for Keras
This layer is implemented for all cases where it is not wanted to reduce the size of the feature maps but the number of channels in a neural network.
It pools each pixel one-dimensional over a given number of feature maps. For that it uses the MaxPooling functionality from Keras. The layer is constructed for 2D inputs.
You can simply include this layer in your model by downloading the file and importing the ChannelsMaxPooling class.

