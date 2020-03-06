from generator import Generator
from discriminator import Discriminator

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

        self.generator = Generator().pathmind(self.img_shape)
        self.discriminator = Discriminator().pathmind(self.img_shape)


# Initialize GAN
gan = GAN()

# Generator
print('|--------------------------------------------|\n' +
      '                 Generator\n' +
      '|--------------------------------------------|\n')
gan.generator.summary()

# Discriminator
print('|--------------------------------------------|\n' +
      '                 Discriminator\n' +
      '|--------------------------------------------|\n')
gan.discriminator.summary()
