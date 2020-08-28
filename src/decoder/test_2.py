import numpy as np
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv2D, Conv2DTranspose, MaxPooling2D,UpSampling2D
from keras.models import Sequential, Model
import tensorflow.keras.backend as K
import tensorflow as tf
from keras import regularizers

df_dataset = pd.read_csv('../all_data.csv')
BATCH_SIZE = 96
CODE_SIZE = 12

cur_files = df_dataset.sample(frac=1).iloc[0:BATCH_SIZE]

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
    return reco

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    plt.show()


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
print(X.max(), X.min())



X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

def build_autoencoder_code(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size, activity_regularizer=regularizers.l1(10e-5)))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))

    return encoder, decoder

def build_autoencoder_img(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    #########
    
    ##########
    
    encoder.add(Flatten())
    encoder.add(Dense(code_size, activity_regularizer=regularizers.l1(10e-5)))
    ######################
    input_flattened = np.arange(0, 2, 0.5)
    input_image = np.reshape(input_flattened, (2, 2, 1))
    input_image_shape = np.shape(input_image)
    input_image_shape = (input_image_shape[0], input_image_shape[1],1)
    print(input_image_shape)

    # Create the model
    model = Sequential()
    model.add(UpSampling2D((32, 32), input_shape=input_image_shape, interpolation='bilinear'))
    model.summary()
    ######################
    
    # The decoder
    decoder = Sequential()
    #decoder.add(InputLayer((code_size,code_size,1)))
    ##########
    #decoder.add(Reshape(1,1,12,1))
    decoder.add(UpSampling2D((32,32), input_shape=(code_size,code_size,1),interpolation='bilinear'))
    #########
    decoder.summary()
    #decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    #decoder.summary()
    decoder.add(Reshape(img_shape))
    
    return encoder, decoder    


################### IMAGE AUTOENCODER
IMG_SHAPE = (X.shape[1:])
encoder_B, decoder_B = build_autoencoder_img(IMG_SHAPE, CODE_SIZE)

inp = Input((IMG_SHAPE))
code = encoder_B(inp)
reconstruction = decoder_B(code)

autoencoder_B = Model(inp,reconstruction)
autoencoder_B.compile(optimizer='adamax', loss='mse')

print(autoencoder_B.summary())

###################  DATA AUTOENCODER
A_train, A_test = train_test_split(A, test_size=0.1, random_state=42)

DATA_SHAPE = (A.shape[1:])
encoder_A, decoder_A = build_autoencoder_code(DATA_SHAPE, CODE_SIZE)

inp = Input((DATA_SHAPE))
code = encoder_A(inp)
reconstruction = decoder_A(code)

autoencoder_A= Model(inp,reconstruction)
autoencoder_A.compile(optimizer='adamax', loss='mse')

print(autoencoder_A.summary())

#############################################
inp = Input((DATA_SHAPE))
code = encoder_A(inp)
reconstruction = decoder_B(code)

autoencoder_C= Model(inp,reconstruction)
autoencoder_C.compile(optimizer='adamax', loss='mse')

print(autoencoder_C.summary())

def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

    #decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded   


for i in range(5):
    print("Epoch %i/50, Generating corrupted samples..."%(i+1))
    autoencoder_B.fit(x=X_train, y=X_train, epochs=10,
                    validation_data=[X_test, X_test])


for i in range(1000):
    print("Epoch %i/200, Generating corrupted samples..."%(i+1))
    autoencoder_A.fit(x=A_train, y=A_train, epochs=10,
                    validation_data=[A_test, A_test])

for k in range(2):
    data = A_test[k]
    reco = 255*(visualize_A_with_B(data,encoder_A,decoder_B) +0.5)
    cv2.imwrite('reconstructed'+str(i) + ','+ str(k)+'.jpg', cv2.cvtColor(reco, cv2.COLOR_RGB2BGR))  

##################################################""


'''

for i in range(3):
    img = X_test[i]
    visualize_B(img,encoder_B,decoder_B)

for i in range(3):
    data = A_test[i]
    visualize_A(data,encoder_A,decoder_A)

for i in range(7):
    data = A_test[i]
    reco = 255*(visualize_A_with_B(data,encoder_A,decoder_B) +0.5)
    print(reco)
    cv2.imwrite('reconstructed'+str(i)+'.jpg', cv2.cvtColor(reco, cv2.COLOR_RGB2BGR))  '''