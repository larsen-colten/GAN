from generator import Generator
from discriminator import Discriminator
from keras.datasets import mnist


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

        # Set generator and discriminator
        self.generator = Generator().pathmind(self.img_shape)
        self.discriminator = Discriminator().small(self.img_shape)


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
