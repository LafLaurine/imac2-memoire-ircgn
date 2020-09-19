import numpy as np
import pandas as pd
import os
import cv2 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, Model, model_from_json
import tensorflow.keras.backend as K

df_dataset = pd.read_csv('../all_data.csv')
BATCH_SIZE = 64
CODE_SIZE = 14
EPOCHS = 20
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_data(df, size):
    X = np.empty(shape=(size, 512, 512, 3))
    data = np.empty(shape=(size, CODE_SIZE , 1))
    cur_files = df_dataset.sample(frac=1).iloc[0:BATCH_SIZE]
    for i in range(0, size):
        file = cur_files.iloc[i]
        print(file)
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
        center1 = file.Center1
        center2 = file.Center2
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
        data[i][11] = center1
        data[i][12] = center2
        data[i][13] = box
    return X, data

X, A = get_data(df_dataset, BATCH_SIZE)
X = X.astype('float32') / 255.0 - 0.5

def get_image_training_set():
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
    return X_train, X_test

def get_data_training_set():
    A_train, A_test = train_test_split(A, test_size=0.1, random_state=42)
    A_train = A_train[:, :, 0]
    A_test = A_test[:, :, 0]
    return A_train, A_test

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    plt.show()

def custom_loss(y_true, y_pred):
   y2_pred = y_pred[0]
   y2_true = y_true[0]
   loss = K.mean(K.square(y2_true - y2_pred), axis=-1)
   return loss

X_train, X_test = get_image_training_set()
A_train, A_test = get_data_training_set()

def build_decoder(code_size):
    # The decoder
    decoder = Sequential()
    decoder.add(Input(shape=(code_size,)))
    decoder.add(Dense(64*64*3))
    decoder.add(Reshape((64,64,3)))
    decoder.add(LeakyReLU(alpha=0.3))
    decoder.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(BatchNormalization(epsilon=0.005))
    decoder.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
    decoder.add(BatchNormalization(epsilon=0.005))
    decoder.add(Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding='same'))
    return decoder

if os.path.isfile('decoder.json') and os.path.isfile('decoder_weights.hdf5'):
    json_file = open('decoder.json', 'r')
    loaded_decoder = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_decoder)
    loaded_model = build_decoder(CODE_SIZE)
    # load weights into new model
    loaded_model.load_weights("decoder_weights.hdf5")
    print("Loaded model from disk")
    decoder = loaded_model

decoder = build_decoder(CODE_SIZE)
decoder.compile(optimizer='adam', loss=custom_loss,loss_weights=[1.0])
decoder.summary()

# Choosen vector
vector = A[4].reshape((1,14))

def visualize(decoder, i):
    reco = decoder.predict(vector)
    reco = np.array(reco)
    reco.reshape((512,512,3))
    reco = np.squeeze(reco,axis=0)
    cv2.imwrite('reconstructed'+str(i)+'.jpg', cv2.cvtColor(255*(reco+0.5), cv2.COLOR_RGB2BGR))
    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

for i in range(EPOCHS):
    print("Epoch %i, Generating images..."%(i))
    decoder.fit(x=A_train, y=X_train, epochs=EPOCHS,validation_data=[A_test, X_test])
    visualize(decoder, i)

def save_model():

  def save(model, model_name):
      model_path = "%s.json" % model_name
      weights_path = "%s_weights.hdf5" % model_name
      options = {"file_arch": model_path,
                  "file_weight": weights_path}
      json_string = model.to_json()
      open(options['file_arch'], 'w').write(json_string)
      model.save_weights(options['file_weight'])

  save(decoder, "decoder")

save_model()