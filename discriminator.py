from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Flatten, MaxPool2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential


class Discriminator:
    def fcc_gan(self, img_shape):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), strides=(2, 2), input_shape=img_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), input_shape=img_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (3, 3), strides=(2, 2), input_shape=img_shape))
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

        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        return model

    def small(self, img_shape):
        model = Sequential()

        model.add(
            Conv2D(
                input_shape=img_shape,
                filters=10,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(
            Conv2D(filters=10, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        return model

    # Implementation of VGG 16 Classifier model
    def vgg_16(self, img_shape):
        model = Sequential()
        model.add(
            Conv2D(
                input_shape=img_shape,
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation="relu"))
        model.add(Dense(4096, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        return model
