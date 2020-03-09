from generator import Generator
from discriminator import Discriminator
from keras.datasets import mnist
import numpy as np
from keras.layers import Input
from keras.models import Sequential, Model
from keras.optimizers import Adam


IMG_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 20
EPOCHS = 10
VALIDATION_SPLIT = .10


class GAN():
    def __init__(self):
        self.img_rows = IMG_SIZE
        self.img_cols = IMG_SIZE
        self.channels = CHANNELS
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

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
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')


# Initialize GAN, generator and discriminator
gan = GAN()

# Generator
print('|--------------------------------------------|\n' +
      '                 Generator\n' +
      '|--------------------------------------------|\n')
# gan.generator.summary()

# Discriminator
print('|--------------------------------------------|\n' +
      '               Discriminator\n' +
      '|--------------------------------------------|\n')
gan.discriminator.summary()

# Load Data
print('|--------------------------------------------|\n' +
      '                  Load Data\n' +
      '|--------------------------------------------|\n')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Data Loaded")

print('|--------------------------------------------|\n' +
      '                  Training\n' +
      '|--------------------------------------------|\n')
# Rescale -1 to 1
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)

half_batch = int(BATCH_SIZE / 2)

# Adversarial ground truths
valid = np.ones((BATCH_SIZE, 1))
fake = np.zeros((BATCH_SIZE, 1))

for epoch in range(EPOCHS):
    ########################
    #  Train Discriminator
    ########################
    # Select random half batch of images
    indexes = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
    images = x_train[indexes]

    noise = np.random.normal(0, 1, (BATCH_SIZE, gan.latent_dim))

    # Generate half batch of images
    gen_images = gan.generator.predict(noise)

    d_loss_real = gan.discriminator.train_on_batch(images, valid)
    d_loss_fake = gan.discriminator.train_on_batch(gen_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    ########################
    #  Train Generator
    ########################
    # Generate Noise
    noise = np.random.normal(0, 1, (BATCH_SIZE, gan.latent_dim))

    g_loss = gan.combined.train_on_batch(noise, valid)

    # Plot the progress
    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
