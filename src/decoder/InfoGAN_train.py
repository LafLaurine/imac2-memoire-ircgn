from numpy import zeros
from numpy import ones
import matplotlib.pyplot as plt
from numpy import expand_dims
from numpy import hstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, Flatten, Conv2D, UpSampling2D, Conv2DTranspose, LeakyReLU, BatchNormalization,Activation
from matplotlib import pyplot
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import cv2

img_rows = 512
img_cols = 512
channels = 3
num_classes = 50
img_shape = (img_rows, img_cols, channels)
latent_dim = 112

def build_disk_and_q_net(img_shape=img_shape, num_classes=num_classes):

  img = Input(shape=img_shape)

  # Shared layers between discriminator and recognition network
  model = Sequential()
  model.add(Conv2D(3, kernel_size=3, input_shape=img_shape, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(16, kernel_size=3, input_shape=img_shape, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Flatten())

  img_embedding = model(img)

  # Discriminator
  validity = Dense(1, activation='sigmoid')(img_embedding)

  # Recognition
  q_net = Dense(128, activation='relu')(img_embedding)
  label = Dense(num_classes, activation='softmax')(q_net)
  model.summary()

  # Return discriminator and recognition network
  return Model(img, validity), Model(img, label)
 

def build_generator(latent_dim=latent_dim, channels=channels):

  model = Sequential()
  model.add(Dense(128 * 64 * 64, activation="relu", input_dim=latent_dim))
  model.add(Reshape((64, 64, 128)))
  model.add(BatchNormalization(momentum=0.8))
  model.add(UpSampling2D())
  model.add(Conv2D(64, kernel_size=3, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(UpSampling2D())
  model.add(Conv2D(32, kernel_size=3, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(UpSampling2D())
  model.add(Conv2D(16, kernel_size=3, padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(Conv2D(channels, kernel_size=3, padding='same'))
  model.add(Activation("tanh"))

  gen_input = Input(shape=(latent_dim,))
  img = model(gen_input)

  model.summary()

  return Model(gen_input, img)

def mutual_info_loss(c, c_given_x):
  """The mutual information metric we aim to minimize"""
  eps = 1e-8
  conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
  entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

  return conditional_entropy + entropy
 
df_dataset = pd.read_csv('../all_data.csv')
TOTAL_SAMPLES = df_dataset.shape[0]

def get_real_images(df, size, total):
    cur_files = df.sample(frac=1).iloc[0:size]
    X = np.empty(shape=(size, 512, 512, 3))
    for i in range(0, size):
        file = cur_files.iloc[i]
        img_uri = '../'+file.File_name
        img = cv2.imread(img_uri)
        X[i] = img
    return X


def sample_generator_input(batch_size, num_classes=num_classes):
  # Generator inputs
  sampled_noise = np.random.normal(0, 1, (batch_size, 62))
  sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
  sampled_labels = to_categorical(sampled_labels, num_classes=num_classes)

  return sampled_noise, sampled_labels
  
def train(epochs, batch_size=4, sample_interval=50):
	# Load the dataset
	X_train = get_real_images(df_dataset, 16, TOTAL_SAMPLES)

	# Adversarial ground truths
	valid = np.ones((batch_size, 1))
	fake = np.zeros((batch_size, 1))

	for epoch in tqdm(range(epochs)):
		# Select a random half batch of images
		idx = np.random.randint(0, X_train.shape[0], batch_size)
		imgs = X_train[idx]
		# Sample noise and categorical labels
		sampled_noise, sampled_labels = sample_generator_input(batch_size)
		gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
		gen_imgs = generator.predict(gen_input)

		# Train on real and generated data
		d_loss_real = discriminator.train_on_batch(imgs, valid)
		d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

		# Avg. loss
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

		# ---------------------
		#  Train Generator and Q-network
		# ---------------------

		g_loss = combined.train_on_batch(gen_input, [valid, sampled_labels])

		# If at save interval => save generated image samples
		if epoch % sample_interval == 0:
			sample_images(epoch)
 
def sample_images(epoch, num_classes=num_classes):
	r, c = 10, 10
	fig, axs = plt.subplots(r, c)
	for i in range(c):
		sampled_noise, _ = sample_generator_input(c)
		label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=num_classes)
		gen_input = np.concatenate((sampled_noise, label), axis=1)
		gen_imgs = generator.predict(gen_input)
		gen_imgs = 0.5 * gen_imgs + 0.5
		for j in range(r):
			axs[j,i].imshow(gen_imgs[j,:,:,0], cmap='gray')
			axs[j,i].axis('off')
			fig.savefig("%d.png" % epoch)
	plt.close()

optimizer = Adam(0.0002, 0.5)
losses = ['binary_crossentropy',mutual_info_loss]

# Build and the discriminator and recognition network
discriminator, auxilliary = build_disk_and_q_net()

# For the combined model we will only train the generator
discriminator.trainable = False
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

# Build and compile the recognition network Q
auxilliary.compile(loss=[mutual_info_loss], optimizer=optimizer, metrics=['accuracy'])

# Build the generator
generator = build_generator()

# The generator takes noise and the target label as input
# and generates the corresponding digit of that label
gen_input = Input(shape=(latent_dim,))
img = generator(gen_input)


# The discriminator takes generated image as input and determines validity
valid = discriminator(img)
# The recognition network produces the label
target_label = auxilliary(img)

# The combined model  (stacked generator and discriminator)
combined = Model(gen_input, [valid, target_label])
combined.compile(loss=losses, optimizer=optimizer)

train(epochs=1000, batch_size=4, sample_interval=50)


def save_model():

  def save(model, model_name):
      model_path = "%s.json" % model_name
      weights_path = "%s_weights.hdf5" % model_name
      options = {"file_arch": model_path,
                  "file_weight": weights_path}
      json_string = model.to_json()
      open(options['file_arch'], 'w').write(json_string)
      model.save_weights(options['file_weight'])

  save(generator, "generator")
  save(discriminator, "discriminator")

save_model()