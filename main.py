# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 01:54:08 2018

@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import scipy
import math
import os
import time
import random
from keras import models
from keras import layers
from keras.preprocessing import image
from inception_v3 import InceptionV3

conv_base = InceptionV3(include_top=True, 
                        weights='imagenet',   
                        input_shape=(299, 299, 3))

conv_base.summary()

img_path = './wmylxmj.jpg'

img = image.load_img(img_path, target_size=(299, 299))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255

print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

layer_outputs = [layer.output for layer in conv_base.layers]
activation_model = models.Model(inputs=conv_base.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

plt.matshow(activations[0][0, :, :, 0], cmap='viridis')
plt.show()
plt.matshow(activations[1][0, :, :, 7], cmap='viridis')
plt.show()
plt.matshow(activations[2][0, :, :, 0], cmap='viridis')
plt.show()
plt.matshow(activations[4][0, :, :, 2], cmap='viridis')
plt.show()
plt.matshow(activations[5][0, :, :, 6], cmap='viridis')
plt.show()
plt.matshow(activations[7][0, :, :, 30], cmap='viridis')
plt.show()
plt.matshow(activations[15][0, :, :, 45], cmap='viridis')
plt.show()
plt.matshow(activations[17][0, :, :, 59], cmap='viridis')
plt.show()
plt.matshow(activations[25][0, :, :, 44], cmap='viridis')
plt.show()
plt.matshow(activations[28][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[32][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[37][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[43][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[57][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[64][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[78][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[90][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[120][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[178][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[250][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[278][0, :, :, 58], cmap='viridis')
plt.show()
plt.matshow(activations[309][0, :, :, 76], cmap='viridis')
plt.show()
plt.matshow(activations[309][0, :, :, 2], cmap='viridis')
plt.show()
plt.matshow(activations[309][0, :, :, 34], cmap='viridis')
plt.show()
plt.matshow(activations[309][0, :, :, 28], cmap='viridis')
plt.show()

layer_names = []
for layer in conv_base.layers:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    try:
        plt.savefig("./images/"+str(layer_name)+".jpg", aspect='auto', cmap='viridis')
        pass
    except:
        pass
    plt.show()