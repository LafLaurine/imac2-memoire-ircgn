import numpy as np
import pandas as pd
import os
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.models import Sequential, Model, model_from_json
import tensorflow.keras.backend as K
import tensorflow as tf

df_dataset = pd.read_csv('../all_data.csv')
BATCH_SIZE = 128
CODE_SIZE = 12

def get_data(df, size):
    X = np.empty(shape=(size, 512, 512, 3))
    data = np.empty(shape=(size, CODE_SIZE , 1))
    cur_files = df_dataset.sample(frac=1).iloc[0:BATCH_SIZE]
    for i in range(0, size):
        file = cur_files.iloc[i]
        img_uri = '../'+file.File_name
        img = cv2.imread(img_uri)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X[i] = im_rgb
        lips = file.Lips_distance
        theta = file.Theta
        phi = file.Phi
        psi = file.Psi
        expression1 = file.Expression1
        expression2 = file.Expression2
        expression3 = file.Expression3
        expression4 = file.Expression4
        expression5 = file.Expression5
        expression6 = file.Expression6
        expression7 = file.Expression7
        center = file.Center
        box = file.Bounding_box
        data[i][0] = lips
        data[i][1] = theta
        data[i][2] = phi
        data[i][3] = psi
        data[i][4] = expression1
        data[i][5] = expression2
        data[i][6] = expression3
        data[i][7] = expression4
        data[i][8] = expression5
        data[i][9] = expression6
        data[i][10] = expression7
        data[i][11] = box
    return X, data

X, A = get_data(df_dataset, BATCH_SIZE)
X = X.astype('float32') / 255.0 - 0.5

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    plt.show()


X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

filters=(32,64,128,256)
depth = 3
chanDim = -1

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(32, (3, 3), strides = (1,1), padding="same", activation='relu'))
    encoder.add(Conv2D(64, (3, 3), strides = (2, 2), padding="same", activation='relu'))
    encoder.add(Conv2D(128, (3, 3), strides = (2, 2), padding="same", activation='relu'))
    encoder.add(Conv2D(256, (3, 3), strides = (2, 2), padding="same", activation='relu'))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(64*64*256))
    decoder.add(Reshape((64,64,256)))
    decoder.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))

    encoder.summary()
    decoder.summary()
    return encoder, decoder
################### IMAGE AUTOENCODER
IMG_SHAPE = (X.shape[1:])
encoder_B, decoder_B = build_autoencoder(IMG_SHAPE, CODE_SIZE)

inp = Input((IMG_SHAPE))
code = encoder_B(inp)
reconstruction = decoder_B(code)

if os.path.isfile('autoencoder_B.json') and os.path.isfile('autoencoder_B_weights.hdf5'):
    # load json and create model
    json_file = open('autoencoder_B.json', 'r')
    loaded_autoencoder_B_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_autoencoder_B_json)
    # load weights into new model
    loaded_model.load_weights("autoencoder_B_weights.hdf5")
    print("Loaded model from disk")
    autoencoder_B = loaded_model
else:
    autoencoder_B = Model(inp,reconstruction)

autoencoder_B.compile(optimizer='adamax', loss='mse')

print(autoencoder_B.summary())

###################  DATA AUTOENCODER

A_train, A_test = train_test_split(A, test_size=0.1, random_state=42)

DATA_SHAPE = (A.shape[1:] + (1,))
print(DATA_SHAPE)
encoder_A, decoder_A = build_autoencoder(DATA_SHAPE, CODE_SIZE)

inp = Input((DATA_SHAPE))
code = encoder_A(inp)
reconstruction = decoder_A(code)

if os.path.isfile('autoencoder_A.json') and os.path.isfile('autoencoder_A_weights.hdf5'):
    # load json and create model
    json_file = open('autoencoder_A.json', 'r')
    loaded_autoencoder_A_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_autoencoder_A_json)
    # load weights into new model
    loaded_model.load_weights("autoencoder_A_weights.hdf5")
    print("Loaded model from disk")
    autoencoder_A = loaded_model
else:
    autoencoder_A = Model(inp,reconstruction)

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

for i in range(3):
    print("Epoch %i/25, Generating corrupted samples..."%(i+1))
    autoencoder_B.fit(x=X_train, y=X_train, epochs=10,
                    validation_data=[X_test, X_test])

for i in range(3):
    img = X_test[i]
    visualize_B(img,encoder_B,decoder_B)

##################################################""

for i in range(200):
    print("Epoch %i/200, Generating corrupted samples..."%(i+1))

    autoencoder_A.fit(x=A_train, y=A_train, epochs=10,
                    validation_data=[A_test, A_test])

for i in range(3):
    img = A_test[i]
    visualize_A(img,encoder_A,decoder_A)

for i in range(3):
    img = A_test[i]
    visualize_A_with_B(img,encoder_A,decoder_B)

def save_model():

  def save(model, model_name):
      model_path = "%s.json" % model_name
      weights_path = "%s_weights.hdf5" % model_name
      options = {"file_arch": model_path,
                  "file_weight": weights_path}
      json_string = model.to_json()
      open(options['file_arch'], 'w').write(json_string)
      model.save_weights(options['file_weight'])

  save(autoencoder_A, "autoencoder_A")
  save(autoencoder_B, "autoencoder_B")

save_model()