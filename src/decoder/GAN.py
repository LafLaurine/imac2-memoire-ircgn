import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, GlobalAveragePooling2D, Dense, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.optimizers import Adam

# function for building the discriminator layers
def build_discriminator(start_filters, spatial_dim, filter_size):
    
    # function for building a CNN block for downsampling the image
    def add_discriminator_block(x, filters, filter_size):
      x = Conv2D(filters, filter_size, padding='same')(x)
      x = BatchNormalization()(x)
      x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
      x = BatchNormalization()(x)
      x = LeakyReLU(0.3)(x)
      return x
    
    # input is an image with shape spatial_dim x spatial_dim and 3 channels
    inp = Input(shape=(spatial_dim, spatial_dim, 3))

    # design the discrimitor to downsample the image 3x
    x = add_discriminator_block(inp, start_filters, filter_size)
    x = add_discriminator_block(x, start_filters * 2, filter_size)
    x = add_discriminator_block(x, start_filters * 4, filter_size)
    
    # average and return a binary output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)    
    return Model(inputs=inp, outputs=x)

# function for building a CNN block for upsampling the image
def add_generator_block(x, filters, filter_size):
	x = Conv2DTranspose(filters, filter_size, strides=2, padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(0.3)(x)
	return x

def build_generator(start_filters, filter_size, latent_dim):
  # input is a noise vector 
  inp = Input(shape=(latent_dim,))

  # projection of the noise vector into a tensor with 
  # same shape as last conv layer in discriminator
  x = Dense(128 * 64 * 64, activation="relu", input_dim=latent_dim)(inp)
  x = BatchNormalization()(x)
  x = Reshape((64, 64, 128))(x)
  x = add_generator_block(x, start_filters * 4, filter_size)
  x = add_generator_block(x, start_filters * 2, filter_size)    
  x = add_generator_block(x, start_filters, filter_size)

  # turn the output into a 3D tensor, an image with 3 channels 
  x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
  
  return Model(inputs=inp, outputs=x)

# load our dataset attributes
df_dataset = pd.read_csv('../all_data.csv')
TOTAL_SAMPLES = df_dataset.shape[0]

SPATIAL_DIM = 512
# size of noise vector
LATENT_DIM_GAN = 100 
# filter size in conv layer
FILTER_SIZE = 5
# number of filters in conv layer
NET_CAPACITY = 16
# batch size
BATCH_SIZE_GAN = 16
# interval for displaying generated images
PROGRESS_INTERVAL = 40
# directory for storing generated images
ROOT_DIR = 'visualization'
if not os.path.isdir(ROOT_DIR):
    os.mkdir(ROOT_DIR)

saved = False
    
def construct_models(verbose=False):
    if not saved:
        ### discriminator
        discriminator = build_discriminator(NET_CAPACITY, SPATIAL_DIM, FILTER_SIZE)
        # compile discriminator
        discriminator.trainable = False
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['mae'])
        discriminator.summary()
        ### generator
        # build generator
        generator = build_generator(NET_CAPACITY, FILTER_SIZE, LATENT_DIM_GAN)
    else:
        discriminator = load_model('discriminatorTrained.h5')
        discriminator.trainable = False
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['mae'])
        generator = load_model('generatorTrained.h5')

    ### DCGAN 
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    discriminator.trainable = False 
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['mae'])

    if verbose: 
        discriminator.summary()
        generator.summary()
        gan.summary()

    discriminator.save("discriminatorTrained.h5")
    generator.save("generatorTrained.h5")
    save = True
    return generator, discriminator, gan
  
generator_faces, discriminator_faces, gan_faces = construct_models(verbose=True)


# number of discriminator updates per alternating training iteration
DISC_UPDATES = 1  
# number of generator updates per alternating training iteration
GEN_UPDATES = 1 

def get_real_images(df, size, total):
    cur_files = df.sample(frac=1).iloc[0:size]
    X = np.empty(shape=(size, SPATIAL_DIM, SPATIAL_DIM, 3))
    for i in range(0, size):
        file = cur_files.iloc[i]
        img_uri = '../'+file.File_name
        img = cv2.imread(img_uri)
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

          #### Discriminator training loop ####
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
              num_vis = min(BATCH_SIZE_GAN, 5)
              imgs_real = get_real_images(df, num_vis, TOTAL_SAMPLES)
              noise = np.random.randn(num_vis, LATENT_DIM_GAN)
              imgs_fake = generator.predict(noise)
              img = []
              for obj_plot in [imgs_fake]:
                  plt.figure(figsize=(num_vis * 3, 3))
                  for b in range(num_vis):
                      disc_score = float(discriminator.predict(np.expand_dims(obj_plot[b], axis=0))[0])
                      plt.subplot(1, num_vis, b + 1)
                      plt.title(str(round(disc_score, 3)))
                      plt.imshow(obj_plot[b] * 0.5 + 0.5)
                      img = obj_plot[b]
                  if obj_plot is imgs_fake:
                      plt.savefig(os.path.join(ROOT_DIR, str(total_it) + '.jpg'), format='jpg', bbox_inches='tight')
                      cv2.imwrite(os.path.join(ROOT_DIR, str(total_it) + 'bis.jpg'), img)
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
      avg_loss_discriminator.append(np.mean(loss_discriminator))
      avg_loss_generator.append(np.mean(loss_generator))
      print('Avg loss discriminator', avg_loss_discriminator)
      print('Avg loss generator', avg_loss_generator)

  return generator, discriminator, gan

generator_faces, discriminator_faces, gan_faces = run_training(generator_faces, 
                                                               discriminator_faces, 
                                                               gan_faces, 
                                                               num_epochs=5, 
                                                               df=df_dataset)