import numpy as np
from keras.layers import BatchNormalization, Dense, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential


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

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        return model
