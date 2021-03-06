import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from matplotlib import pyplot as plt
import cv2
import glob
import tensorflow as tf

from discriminator import Discriminator
from generator import Generator

IMG_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 12
EPOCHS = 100
LATENT_DIM = 100


class GAN:
    def __init__(self):
        self.img_rows = IMG_SIZE
        self.img_cols = IMG_SIZE
        self.channels = CHANNELS
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.dataset = None
        self.latent_dim = LATENT_DIM

        # Set generator and discriminator
        self.generator = Generator().tf_tut(self.img_shape, LATENT_DIM)
        self.discriminator = Discriminator().tf_tut(self.img_shape)
        self.gan = self.define_gan(self.generator, self.discriminator)

    def define_gan(self, gen, dis):
        dis.trainable = False
        model = Sequential()
        model.add(gen)
        model.add(dis)
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model

    def load_data(self):
        (x_train, y_train), (_, _) = mnist.load_data()
        X = np.expand_dims(x_train, axis=-1)
        x_indexes = y_train == 8
        X = x_train[x_indexes]
        X = (x_train - 127.5) / 127.5
        print(X.shape)
        self.dataset = X
        return "MINST dataset"

    def load_data_retriever(self):
        image_list = []
        for filename in glob.glob("retriever_data/*.jpg"):
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            image_list.append(img)

        plt.imshow(image_list[0])
        plt.show()
        image_list = np.array(image_list)
        
        self.dataset = image_list
        return 'Standford Dogs dataset'
            

    def train(self, epoch, batch_size):
        batches_per_epoch = int(self.dataset.shape[0] / batch_size)
        number_of_steps = batches_per_epoch * epoch
        half_batch = int(batch_size / 2)
        d1_hist, d2_hist, g_hist, a1_hist, a2_hist = (
            list(),
            list(),
            list(),
            list(),
            list(),
        )

        for i in range(number_of_steps):
            # Select real samples
            ix = np.random.randint(0, self.dataset.shape[0], half_batch)
            real_images = self.dataset[ix]
            real_images = np.expand_dims(real_images, axis=3)
            real_images_labels = np.ones((half_batch, 1))

            # Train discriminator on real images
            d_loss1, d_acc1 = self.discriminator.train_on_batch(
                real_images, real_images_labels
            )

            # Generate gen_inputs
            gen_input = np.random.randn(self.latent_dim * half_batch)
            gen_input = gen_input.reshape(half_batch, self.latent_dim)

            # Generate fake images
            fake_images = self.generator.predict(gen_input)
            fake_images_labels = np.zeros((half_batch, 1))

            # Train disriminator on fake images
            d_loss2, d_acc2 = self.discriminator.train_on_batch(
                fake_images, fake_images_labels
            )
            # Train disriminator on real images
            d_loss2, d_acc2 = self.discriminator.train_on_batch(
                real_images, real_images_labels
            )

            # Create latent points for Generator
            x_gan = np.random.randn(self.latent_dim * batch_size)
            x_gan = x_gan.reshape(batch_size, self.latent_dim)

            y_gan = np.ones((batch_size, 1))

            g_loss = self.gan.train_on_batch(x_gan, y_gan)

            if i % batches_per_epoch == 0:
                print(
                    "Epoch: %d || d1_loss = %.3f || d2_loss = %.3f || g_loss = %.3f || acc1 = %d || acc2 = %d"
                    % (
                        i / batches_per_epoch,
                        d_loss1,
                        d_loss2,
                        g_loss,
                        int(100 * d_acc1),
                        int(100 * d_acc2),
                    )
                )

            # record history
            d1_hist.append(d_loss1)
            d2_hist.append(d_loss2)
            g_hist.append(g_loss)
            a1_hist.append(d_acc1)
            a2_hist.append(d_acc2)

            if i % batches_per_epoch == 0:
                self.sample_images(i / batches_per_epoch)

        # Plot
        self.plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

    def sample_images(self, number):
        rows, column = 5, 5
        noise = np.random.normal(0, 1, (rows * column, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(rows, column)
        cnt = 0
        for i in range(rows):
            for j in range(column):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig("retriever_dc/%d.png" % number)
        plt.show()
        plt.close()

    def plot_history(self, d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.plot(d1_hist, label="d-real")
        plt.plot(d2_hist, label="d-fake")
        plt.plot(g_hist, label="gen")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.legend()
        # plot discriminator accuracy
        plt.subplot(2, 1, 2)
        plt.plot(a1_hist, label="acc-real")
        plt.plot(a2_hist, label="acc-fake")
        plt.legend()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.savefig("results/plot_retriever_dc.png")
        plt.show()
        plt.close()


##############################################################
# Running the Model
##############################################################
# Initialize GAN, generator and discriminator
gan = GAN()

# Generator
print(
    "\n|--------------------------------------------|\n"
    + "                 Generator\n"
    + "|--------------------------------------------|\n"
)
gan.generator.summary()

# Discriminator
print(
    "\n|--------------------------------------------|\n"
    + "               Discriminator\n"
    + "|--------------------------------------------|\n"
)
gan.discriminator.summary()

# Load Data
print(
    "\n|--------------------------------------------|\n"
    + "                  Load Data\n"
    + "|--------------------------------------------|\n"
)
print(gan.load_data_retriever())
print("Data Loaded")

print(
    "\n|--------------------------------------------|\n"
    + "                  Training \n"
    + "|--------------------------------------------|\n"
)
gan.train(EPOCHS, BATCH_SIZE)
