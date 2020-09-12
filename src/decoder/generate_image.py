import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 
from decoder import get_data
from decoder import build_decoder
from decoder import custom_loss
from tensorflow.keras.models import model_from_json

EPOCHS = 50
CODE_SIZE = 14

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))
    plt.show()

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

decoder.compile(optimizer='adam', loss=custom_loss,loss_weights=[1.0])
decoder.summary()

df_dataset = pd.read_csv('../all_data.csv')
BATCH_SIZE = 64
X, A = get_data(df_dataset, BATCH_SIZE)
X = X.astype('float32') / 255.0 - 0.5

# Choosen image to generate with vector
image = A[4].reshape((1,14))

def visualize(decoder, i):
    reco = decoder.predict(image)
    reco = np.array(reco)
    reco.reshape((512,512,3))
    reco = np.squeeze(reco,axis=0)
    cv2.imwrite('reconstructed'+str(i)+'.jpg', reco)
    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

for i in range(EPOCHS):
    print("Epoch %i, Generating images..."%(i))
    visualize(decoder, i)
