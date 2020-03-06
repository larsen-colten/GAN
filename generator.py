import numpy as np
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam


class Generator():
    # Example from https://pathmind.com/wiki/generative-adversarial-network-gan
    def pathmind(self, img_shape):
        noise_shape = (100,)

        model = Sequential()
        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(img_shape), activation='tanh'))
        model.add(Reshape(img_shape))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])

        return model
