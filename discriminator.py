from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import keras.layers as layers


class Discriminator:
    def fcc_gan(self, img_shape):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=img_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, (3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3, 3), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        
        model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

        return model

    def dc_gan(self, img_shape):
        model = Sequential()

        model.add(Conv2D(64, (5, 5), padding="same", input_shape=img_shape))
        model.add(Activation("tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation("tanh"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("tanh"))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        
        opt = SGD(lr=0.00005, nesterov=True)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        return model
    
    def tf_tut(self, img_shape):
        model = Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=img_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
    
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
    
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        model.add(Activation("sigmoid"))
        
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
        return model
