import tensorflow.keras.layers
import tensorflow.keras.models
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2

decoder_input = tensorflow.keras.layers.Input(shape=(786432), name="decoder_input")

decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=300, name="decoder_dense_1")(decoder_input)
decoder_activ_layer1 = tensorflow.keras.layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_dense_layer1)
# 262 144 comes from 512x512 which is the size of the image
decoder_dense_layer2 = tensorflow.keras.layers.Dense(units=262144, name="decoder_dense_2")(decoder_activ_layer1)
decoder_activation = tensorflow.keras.layers.Activation('softmax')(decoder_dense_layer2)
decoder_output = tensorflow.keras.layers.LeakyReLU(name="decoder_output")(decoder_activation)

decoder = tensorflow.keras.models.Model(decoder_input, decoder_output, name="decoder_model")
decoder.summary()

# RMSE
def rmse(y_true, y_predict):
    return tensorflow.keras.backend.mean(tensorflow.keras.backend.square(y_true-y_predict))


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

X = get_real_images(df_dataset, BATCH_SIZE, TOTAL_SAMPLES)
X = X.astype('float32') / 255.0 - 0.5
x_train_orig, x_test_orig = train_test_split(X, test_size=0.1, random_state=42)

X_train = np.reshape(x_train_orig, newshape=(x_train_orig.shape[0], np.prod(x_train_orig.shape[1:])))
X_test = np.reshape(x_test_orig, newshape=(x_test_orig.shape[0], np.prod(x_test_orig.shape[1:])))

decoder.compile(loss="mse", optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005))
decoder.fit(x=X_train, y=X_train, epochs=10, batch_size=32, shuffle=True,validation_data=[X_test, X_test])
