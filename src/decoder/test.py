import numpy as np
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model
import tensorflow.keras.backend as K
import tensorflow as tf

df_dataset = pd.read_csv('../all_data.csv')
BATCH_SIZE = 80
CODE_SIZE = 3

cur_files = df_dataset.sample(frac=1).iloc[0:BATCH_SIZE]

def get_real_images(df, size):
    X = np.empty(shape=(size, 512, 512, 3))
    for i in range(0, size):
        file = cur_files.iloc[i]
        img_uri = '../'+file.File_name
        img = cv2.imread(img_uri)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X[i] = im_rgb
    return X

def get_real_data(df, size):
    X = np.empty(shape=(size, CODE_SIZE , 1))
    for i in range(0, size):
        file = cur_files.iloc[i]
        lips = file.Lips_distance
        angles = file.Euler_angles
        expression = file.Expression
        print(angles)
        center = file.Center
        box = file.Bounding_box
        #for angles, center and expression "cannot convert string to float"
        X[i][0] = lips
        X[i][1] = np.sin(1)
        X[i][2] = box
    return X 

X = get_real_images(df_dataset, BATCH_SIZE)
X = X.astype('float32') / 255.0 - 0.5
print(X.max(), X.min())

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    plt.show()


X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))

    return encoder, decoder
################### IMAGE AUTOENCODER
IMG_SHAPE = (X.shape[1:])
encoder_B, decoder_B = build_autoencoder(IMG_SHAPE, CODE_SIZE)

inp = Input((IMG_SHAPE))
code = encoder_B(inp)
reconstruction = decoder_B(code)

autoencoder_B = Model(inp,reconstruction)
autoencoder_B.compile(optimizer='adamax', loss='mse')

print(autoencoder_B.summary())

###################  DATA AUTOENCODER

A = get_real_data(df_dataset, BATCH_SIZE)
print('A           aaaaaaaaaaaaaaaaaaaa :: \n')
print(A)
print('\n')
A_train, A_test = train_test_split(A, test_size=0.1, random_state=42)

DATA_SHAPE = (A.shape[1:])
encoder_A, decoder_A = build_autoencoder(DATA_SHAPE, CODE_SIZE)

inp = Input((DATA_SHAPE))
code = encoder_A(inp)
reconstruction = decoder_A(code)

autoencoder_A= Model(inp,reconstruction)
autoencoder_A.compile(optimizer='adamax', loss='mse')

print(autoencoder_A.summary())


def visualize_B(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 512, 512, 3) which is the same as the model input
    code = encoder_B.predict(img[None])[0]
    reco = decoder_B.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

def visualize_A(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 512, 512, 3) which is the same as the model input
    code = encoder_A.predict(img[None])[0]
    reco = decoder_A.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()   


def visualize_A_with_B(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 512, 512, 3) which is the same as the model input
    code = encoder_A.predict(img[None])[0]
    reco = decoder_B.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()     

def apply_gaussian_noise(X, sigma=0.1):
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise

'''plt.subplot(1,4,1)
show_image(X_train[0])
plt.subplot(1,4,2)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.01)[0])
plt.subplot(1,4,3)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.1)[0])
plt.subplot(1,4,4)
show_image(apply_gaussian_noise(X_train[:1],sigma=0.5)[0])'''

for i in range(1):
    print("Epoch %i/25, Generating corrupted samples..."%(i+1))
    X_train_noise = apply_gaussian_noise(X_train)
    X_test_noise = apply_gaussian_noise(X_test)

    # We continue to train our model with new noise-augmented data
    autoencoder_B.fit(x=X_train_noise, y=X_train, epochs=10,
                    validation_data=[X_test_noise, X_test])



X_test_noise = apply_gaussian_noise(X_test)

for i in range(3):
    img = X_test[i]
    visualize_B(img,encoder_B,decoder_B)

##################################################""

for i in range(200):
    print("Epoch %i/200, Generating corrupted samples..."%(i+1))
    A_train_noise = apply_gaussian_noise(A_train)
    A_test_noise = apply_gaussian_noise(A_test)

    # We continue to train our model with new noise-augmented data
    autoencoder_A.fit(x=A_train_noise, y=A_train, epochs=10,
                    validation_data=[A_test_noise, A_test])



A_test_noise = apply_gaussian_noise(A_test)

for i in range(3):
    img = A_test[i]
    visualize_A(img,encoder_A,decoder_A)

for i in range(3):
    img = A_test[i]
    visualize_A_with_B(img,encoder_A,decoder_B)    