# GAN (Tensorflow)
This is a project for CS 450 with the overall goal of creating a GAN
that takes in any image data sets and generates simalar look images 
from noise.


**Authors**\
Colten Larsen
Gustavo Hideo
Hansel Palencia
Atsushi Jindo 

## Results
### FCC GAN
This model was pulled from a paper named FCC GAN. It is ocnsistant but traines as a slower rate that other options avalible. Below is the results from the 1st, 5th, and 25th epoch.  
![1st epoch](results/fcc_mnist_25/0.png)  
![5st epoch](results/fcc_mnist_25/4.png)  
![25st epoch](results/fcc_mnist_25/25.png)  
The resulting graph of the loses and accuracy of the discriminator show that the GAN collapsed. With further tunning this could become more stable.  
![Results FCC_GAN](results/fcc_mnist_25/plot_mnist_fcc.png)
### DC GAN
This model is a deep connected GAN. This is one of the more common types of models that you can find. Basing our model on several others, this performed much better than the FCC GAN. Below is the results from the 1st, 5th, and 25th epoch.  
![1st epoch](results/dc_mnist_25/0.png)  
![5st epoch](results/dc_mnist_25/4.png)  
![25st epoch](results/dc_mnist_25/24.png)  
The resulting graph is much more stable with the generator and the discriminator competing. 
![Results DC_GAN](results/dc_mnist_25/plot_dc.png)

## Structure
**discriminator.py**\
Contains all of the different implementations of a classifier
This includes small, an implementation from pathmind and vgg_16

**generator.py**\
Contains all of the different implementations of a generator
This includes an implementation from pathmind

**main.py**\
This is the controller of the project.

## Running the Code
python main.py
