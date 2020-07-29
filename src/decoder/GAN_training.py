import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
 
# number of discriminator updates per alternating training iteration
DISC_UPDATES = 1  
# number of generator updates per alternating training iteration
GEN_UPDATES = 1 

# function for training a GAN
def run_training(generator, discriminator, gan, df=df_celeb, start_it=0, num_epochs=1000, 
                 get_real_images=get_real_images):

  # helper function for selecting 'size' real images
  # and downscaling them to lower dimension SPATIAL_DIM
  def get_real_images(df, size, total):
      cur_files = df.sample(frac=1).iloc[0:size]
      X = np.empty(shape=(size, SPATIAL_DIM, SPATIAL_DIM, 3))
      for i in range(0, size):
          file = cur_files.iloc[i]
          img_uri = 'extracted_faces/Pierre' + file.image_id
          img = cv2.imread(img_uri)
          img = cv2.resize(img, (SPATIAL_DIM, SPATIAL_DIM))
          img = np.flip(img, axis=2)
          img = img.astype(np.float32) / 127.5 - 1.0
          X[i] = img
      return X
  
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
              plt.figure(figsize=(5,2))
              num_vis = min(BATCH_SIZE_GAN, 5)
              imgs_real = get_real_images(df, num_vis, TOTAL_SAMPLES)
              noise = np.random.randn(num_vis, LATENT_DIM_GAN)
              imgs_fake = generator.predict(noise)
              for obj_plot in [imgs_fake, imgs_real]:
                  plt.figure(figsize=(num_vis * 3, 3))
                  for b in range(num_vis):
                      disc_score = float(discriminator.predict(np.expand_dims(obj_plot[b], axis=0))[0])
                      plt.subplot(1, num_vis, b + 1)
                      plt.title(str(round(disc_score, 3)))
                      plt.imshow(obj_plot[b] * 0.5 + 0.5) 
                  if obj_plot is imgs_fake:
                      plt.savefig(os.path.join(ROOT_DIR, str(total_it).zfill(10) + '.jpg'), format='jpg', bbox_inches='tight')
                  plt.show()  

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
      plt.plot(range(len(avg_loss_discriminator)), avg_loss_discriminator)
      plt.plot(range(len(avg_loss_generator)), avg_loss_generator)
      plt.legend(['discriminator loss', 'generator loss'])
      plt.show()

  return generator, discriminator, gan

generator_celeb, discriminator_celeb, gan_celeb = run_training(generator_celeb, 
                                                               discriminator_celeb, 
                                                               gan_celeb, 
                                                               num_epochs=500, 
                                                               df=df_celeb)