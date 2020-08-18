from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, UpSampling2D, LSTM
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import regularizers
from keras.utils import plot_model
from keras.utils import to_categorical
from random import randint
import numpy as np
from numpy import array
from numpy import argmax
from numpy import array_equal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import cv2


def draw(x_test,decoded_imgs):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

df_dataset = pd.read_csv('../all_data.csv')
TOTAL_SAMPLES = df_dataset.shape[0]
BATCH_SIZE = 4
CODE_SIZE = 100

def get_real_images(df, size, total):
    cur_files = df.sample(frac=1).iloc[0:size]
    X = np.empty(shape=(size, 512, 512, 3))
    for i in range(0, size):
        file = cur_files.iloc[i]
        img_uri = '../'+file.File_name
        img = cv2.imread(img_uri)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X[i] = im_rgb
    return X

class ConvolutionalAutoEncoder:
    def main():
        input_img = Input(shape=(512, 512, 3))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        decoder = Model(input_img, decoded)
        decoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        X = get_real_images(df_dataset, BATCH_SIZE, TOTAL_SAMPLES)
        X = X.astype('float32') / 255.0 - 0.5
        x_train, x_test = train_test_split(X, test_size=0.1, random_state=42)
        x_train = np.reshape(x_train, (len(x_train), 512, 512, 3))
        x_test = np.reshape(x_test, (len(x_test), 512, 512, 3))
        decoder.fit(x_train, x_train,
                    epochs=3,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test))
        
        decoded_imgs = decoder.predict(x_test)
        draw(x_test,decoded_imgs)


ConvolutionalAutoEncoder.main()