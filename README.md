# ChannelsMaxPooling Layer Implementation for Keras
-> Pooling over Channels

This layer is implemented for all cases where it is not wanted to reduce the size of the feature maps but the number of channels in a neural network.
It pools each pixel one-dimensional over a given number of feature maps. For that it uses the MaxPooling functionality from Keras. The layer is currently constructed for 2D inputs.

For example this line adds the ChannelsMaxPooling Layer with a filter size of 3 pixels and a sliding window step size of 1 pixel to a given model:
```
model.add(ChannelsMaxPooling(pool_size=3, stride=1))
```

You can simply include this layer in your model by downloading the file and importing the ChannelsMaxPooling class.
```
from channels_pooling_layer import ChannelsMaxPooling
```
