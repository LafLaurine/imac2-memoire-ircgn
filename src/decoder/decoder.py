from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import cv2

df_dataset = pd.read_csv('../all_data.csv')
BATCH_SIZE = 64

def get_real_images(df, size):
    cur_files = df.sample(frac=1).iloc[0:size]
    X = np.empty(shape=(size, 512, 512, 3))
    for i in range(0, size):
        file = cur_files.iloc[i]
        img_uri = '../'+file.File_name
        img = cv2.imread(img_uri)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X[i] = im_rgb
    return X

def get_data(df, size):
    cur_files = df.sample(frac=1).iloc[0:size]
    X = []
    for i in range(0, size):
        file = cur_files.iloc[i]
        lips = file.Lips_distance
        euler = file.Euler_angles
        expr = file.Expression
        center = file.Center
        bb = file.Bounding_box
        X.append(lips)
        X.append(euler)
        X.append(expr)
        X.append(center)
        X.append(bb)
    return X

# latent dim represents our vector
latentDim=14
filters=(32,64,128,256)
depth = 3
chanDim = -1
volumeSize = (32,32,3) # last layer of encoder
latentInputs = Input(shape=(latentDim,))
x = Dense(np.prod((volumeSize)))(latentInputs)
x = Reshape((32, 32, 3))(x)

for f in filters[::-1]:
    # apply a CONV_TRANSPOSE => RELU => BN operation
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

label = get_data(df_dataset, BATCH_SIZE)
X_train, X_test = train_test_split(label, test_size=0.1, random_state=42)
print(X_test)
decoder.fit(x=X_train, y=X_train, epochs=10,
                    validation_data=[X_test, X_test])