import numpy as np
from keras.layers import (Activation, BatchNormalization, Conv2DTranspose,
                          Dense, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential


class Generator:
    def small(self, img_shape, latent_dim):
        noise_shape = (latent_dim,)

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
        model.add(Dense(np.prod(img_shape), activation="tanh"))
        model.add(Reshape(img_shape))

        return model

    def fcc_gan(self, img_shape, latent_dim):
        noise_shape = (latent_dim,)

        model = Sequential()
        model.add(Dense(64, input_shape=noise_shape))
        model.add(Activation("relu"))
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dense(1152))
        model.add(BatchNormalization())
        model.add(Reshape((3, 3, 128)))
        model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(32, (2, 2), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(1, (2, 2), strides=(2, 2)))
        model.add(Activation("tanh"))

        return model
