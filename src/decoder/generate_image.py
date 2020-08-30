import numpy as np
import pandas as pd
import os
import cv2 
from decoder import get_data_training_set
from decoder import get_image_training_set

EPOCHS = 50

if os.path.isfile('decoder.json') and os.path.isfile('decoder_weights.hdf5'):
    json_file = open('decoder.json', 'r')
    loaded_decoder = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_decoder)
    # load weights into new model
    loaded_model.load_weights("decoder_weights.hdf5")
    print("Loaded model from disk")
    decoder = loaded_model


X_train, X_test = get_image_training_set()
A_train, A_test = get_data_training_set()

image = A[4].reshape((1,12))

def visualize(decoder, i):
    reco = decoder.predict(image)
    print('=======================\n')
    print(reco)
    print('=======================\n')
    reco = np.array(reco)
    print(reco)
    print('=======================\n')
    reco.reshape((512,512,3))
    print(reco)
    print('=======================\n')
    reco = np.squeeze(reco,axis=0)
    print(reco)
    print('=======================\n')
    print(np.size(reco))
    cv2.imwrite('reconstructed'+str(i)+'.jpg', reco)
    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

for i in range(EPOCHS):
    print("Epoch %i, Generating corrupted samples..."%(i))
    decoder.fit(x=A_train, y=X_train, epochs=EPOCHS,validation_data=[A_test, X_test])
    visualize(decoder, i)
