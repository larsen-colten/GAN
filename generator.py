from keras.layers import Activation, BatchNormalization, Conv2DTranspose, Dense, Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU


class Generator:
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

    def dc_gan(self, img_shape, latent_dim):
        noise_shape = (latent_dim,)

        model = Sequential()
        model.add(Dense(1024, input_shape=noise_shape))
        model.add(Activation("tanh"))
        model.add(Dense(128 * 7 * 7))
        model.add(BatchNormalization())
        model.add(Activation("tanh"))
        model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(64, (5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (5, 5), padding="same"))
        model.add(Activation("tanh"))
        return model
    
    def tf_tut (self, img_shappe, latent_dim):
        noise_shape = (latent_dim,)
        
        model = Sequential()
        model.add(Dense(12544, use_bias=False, input_shape=noise_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)
        
        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        
        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same", use_bias=False))
        assert model.output_shape == (None, 28, 28, 1)
        
        return model