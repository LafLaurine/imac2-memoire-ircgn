import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from keras.layers import Conv2D, BatchNormalization, Input, GlobalAveragePooling2D, UpSampling2D, Dense, Deconvolution2D, Reshape
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras import models    

import pandas as pd
import os
from keras.optimizers import Adam

# generator is our decoder
# function for building a CNN block for upsampling the image
def add_generator_block(x, filter_size):
    x = UpSampling2D()(x)
    x = Conv2D(512, kernel_size=filter_size, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D()(x)
    x = Conv2D(256, kernel_size=filter_size, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, kernel_size=filter_size, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, kernel_size=filter_size, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, kernel_size=filter_size, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.8)(x)
    return x

def build_generator(start_filters, filter_size, latent_dim, img_size):
  # input is a noise vector 
  inp = Input(shape=(latent_dim,))

  # projection of the noise vector into a tensor
  x = Dense(start_filters * (img_size[0] // (2 ** filter_size))  *  (img_size[1] // (2 ** filter_size)), activation="relu", input_dim=latent_dim)(inp)
  x = Reshape(((img_size[0] // (2 ** filter_size)),
                           (img_size[1] // (2 ** filter_size)),
                           start_filters))(x)
  x = BatchNormalization(momentum=0.8)(x)

  # design the generator
  x = add_generator_block(x, filter_size)   

  # turn the output into a 3D tensor, an image with 3 channels 
  x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
  x = Activation("tanh")(x)
  return Model(inputs=inp, outputs=x)

def build_discriminator(start_filters, latent_dim, img_size, filter_size):
    inp = Input(shape=(latent_dim,))
    img_shape = (img_size[0], img_size[1], 3)
    x = Dense(start_filters * (img_size[0] // (2 ** filter_size))  *  (img_size[1] // (2 ** filter_size)), activation="relu", input_dim=latent_dim)(inp)
    x = Conv2D(32, kernel_size=filter_size, strides=4, input_shape=img_shape, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, kernel_size=filter_size, strides=4, input_shape=img_shape, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(128, kernel_size=filter_size, strides=2, input_shape=img_shape, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(256, kernel_size=filter_size, strides=2, input_shape=img_shape, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(512, kernel_size=filter_size, strides=2, input_shape=img_shape, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=img_shape, outputs=x)

# load our dataset attributes
df_dataset = pd.read_csv('../all_data.csv')
TOTAL_SAMPLES = df_dataset.shape[0]

# we will downscale the images
SPATIAL_DIM = 64 
# size of noise vector
LATENT_DIM_GAN = 100 
# filter size in conv layer
FILTER_SIZE = 5
# number of filters in conv layer
NET_CAPACITY = 64
# batch size
BATCH_SIZE_GAN = 32
# interval for displaying generated images
PROGRESS_INTERVAL = 80 
# directory for storing generated images
ROOT_DIR = 'visualization'


if not os.path.isdir(ROOT_DIR):
    os.mkdir(ROOT_DIR)

    
def construct_models(verbose=False, saved = False):
    if not saved:
        discriminator = build_discriminator(NET_CAPACITY, LATENT_DIM_GAN, (512,512), FILTER_SIZE)
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['mae'])
        # build generator
        generator = build_generator(NET_CAPACITY, FILTER_SIZE, LATENT_DIM_GAN, (512,512))
    else:
        generator = models.load_model('generatorTrained.h5')

    ### DCGAN 
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False 
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['mae'])

    if verbose: 
        generator.summary()
        discriminator.summary()
        gan.summary()

    generator.save("generatorTrained.h5")
    return generator, discriminator, gan
  
generator_faces, discriminator_faces, gan_faces = construct_models(verbose=True, saved=False)


# number of discriminator updates per alternating training iteration
DISC_UPDATES = 1  
# number of generator updates per alternating training iteration
GEN_UPDATES = 1 

# helper function for selecting 'size' real images
# and downscaling them to lower dimension SPATIAL_DIM
def get_real_images(df, size, total):
    cur_files = df.sample(frac=1).iloc[0:size]
    X = np.empty(shape=(size, SPATIAL_DIM, SPATIAL_DIM, 3))
    for i in range(0, size):
        file = cur_files.iloc[i]
        img_uri = '../'+file.File_name
        img = cv2.imread(img_uri)
        img = cv2.resize(img, (SPATIAL_DIM, SPATIAL_DIM))
        img = np.flip(img, axis=2)
        img = img.astype(np.float32) / 127.5 - 1.0
        X[i] = img
    return X


# function for training a GAN
def run_training(generator, discriminator, gan, df=df_dataset, start_it=0, num_epochs=5, 
                 get_real_images=get_real_images):

  # list for storing loss
  avg_loss_discriminator = []
  avg_loss_generator = []
  total_it = start_it

  # main training loop
  for epoch in range(num_epochs):

      # alternating training loop
      loss_discriminator = []
      loss_generator = []
      for it in range(200): 

          for i in range(DISC_UPDATES): 
              # select a random set of real images
              imgs_real = get_real_images(df, BATCH_SIZE_GAN, TOTAL_SAMPLES)
              # generate a set of random noise vectors
              noise = np.random.randn(BATCH_SIZE_GAN, LATENT_DIM_GAN)
              # generate a set of fake images using the generator
              imgs_fake = generator.predict(noise)
              # train the discriminator on real images with label 1
              d_loss_real = discriminator.train_on_batch(imgs_real, np.ones([BATCH_SIZE_GAN]))[1]
              # train the discriminator on fake images with label 0
              d_loss_fake = discriminator.train_on_batch(imgs_fake, np.zeros([BATCH_SIZE_GAN]))[1]


          # display some fake images for visual control of convergence
          if total_it % PROGRESS_INTERVAL == 0:
              #plt.figure(figsize=(5,2))
              num_vis = min(BATCH_SIZE_GAN, 1)
              imgs_real = get_real_images(df, num_vis, TOTAL_SAMPLES)
              noise = np.random.randn(num_vis, LATENT_DIM_GAN)
              imgs_fake = generator.predict(noise)
              cv2.imwrite(str(total_it)+".jpg", imgs_fake)
              for obj_plot in [imgs_fake]:
                  plt.figure(figsize=(num_vis * 3, 3))
                  if obj_plot is imgs_fake:
                      plt.savefig(os.path.join(ROOT_DIR, str(total_it) + '.jpg'), format='jpg', bbox_inches='tight')

          #### Generator training loop ####
          loss = 0
          y = np.ones([BATCH_SIZE_GAN, 1]) 
          for j in range(GEN_UPDATES):
              # generate a set of random noise vectors
              noise = np.random.randn(BATCH_SIZE_GAN, LATENT_DIM_GAN)
              # train the generator on fake images with label 1
              loss += gan.train_on_batch(noise, y)[1]

          # store loss
          loss_discriminator.append((d_loss_real + d_loss_fake) / 2.)
          loss_generator.append(loss / GEN_UPDATES)
          total_it += 1

      # visualize loss
      clear_output(True)
      print('Epoch', epoch)
      avg_loss_generator.append(np.mean(loss_generator))
      print('Avg loss generator', avg_loss_generator)

      avg_loss_discriminator.append(np.mean(avg_loss_discriminator))
      print('Avg loss discriminator', avg_loss_discriminator)

  return generator, discriminator, gan

generator_faces, discriminator_faces, gan_faces = run_training(generator_faces, discriminator_faces, gan_faces, num_epochs=5, df=df_dataset)