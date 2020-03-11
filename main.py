from generator import Generator
from discriminator import Discriminator
from keras.datasets import mnist
import numpy as np
from keras.layers import Input
from keras.models import Sequential, Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

# import cv2

IMG_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 20
EPOCHS = 10
VALIDATION_SPLIT = 0.10


class GAN:
    def __init__(self):
        self.img_rows = IMG_SIZE
        self.img_cols = IMG_SIZE
        self.channels = CHANNELS
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # Set generator and discriminator
        self.generator = Generator().pathmind(self.img_shape)
        self.discriminator = Discriminator().small(self.img_shape)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer="adam")

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def train(self, epochs, batch_size):
        for epoch in range(epochs):
            # Generate images
            noise = np.random.normal(0, 1, (100, 100))
            imgs_gen = self.generator.predict(noise)

            # Train discriminator on half generated images and half
            idx = np.random.randint(0, self.x_train.shape[0], 100)
            imgs_train = self.x_train[idx]
            imgs_train = (imgs_train.astype(np.float32) - 127.5) / 127.5
            imgs_train = np.expand_dims(imgs_train, axis=3)
            np.expand_dims(imgs_train, axis=3)

            valid = np.ones((100, 1))
            fake = np.zeros((100, 1))

            d_loss_real = self.discriminator.train_on_batch(imgs_train, valid)
            d_loss_fake = self.discriminator.train_on_batch(imgs_gen, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print(d_loss)


# Initialize GAN, generator and discriminator
gan = GAN()

# Generator
print(
    "|--------------------------------------------|\n"
    + "                 Generator\n"
    + "|--------------------------------------------|\n"
)
# gan.generator.summary()

# Discriminator
print(
    "|--------------------------------------------|\n"
    + "               Discriminator\n"
    + "|--------------------------------------------|\n"
)
gan.discriminator.summary()

# Load Data
print(
    "|--------------------------------------------|\n"
    + "                  Load Data\n"
    + "|--------------------------------------------|\n"
)
gan.load_data()
print("Data Loaded")

print(
    "|--------------------------------------------|\n"
    + "                  Training \n"
    + "|--------------------------------------------|\n"
)
gan.train(EPOCHS, BATCH_SIZE)
