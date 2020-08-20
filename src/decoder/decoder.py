from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# latent dim represents our vector
latentDim=14
filters=(64,128,256,512)
depth = 3
chanDim = -1
volumeSize = (32,32,32) # last layer of encoder
latentInputs = Input(shape=(latentDim,))
x = Dense(np.prod((volumeSize)))(latentInputs)
x = Reshape((32, 32, 32))(x)

for f in filters[::-1]:
    # apply a CONV_TRANSPOSE => RELU => BN operation
    print(f)
    x = Conv2DTranspose(f, (3, 3), strides=2,
        padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=chanDim)(x)

x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
outputs = Activation("sigmoid")(x)

decoder = Model(latentInputs, outputs, name="decoder")
opt = Adam(lr=1e-3)
decoder.compile(loss="mse", optimizer=opt)
decoder.summary()