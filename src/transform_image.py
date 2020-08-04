import cv2
import os
import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import transform
from skimage import img_as_float
from scipy import ndimage, misc

from get_rotation_matrix import *

def perspective_trans(imagePoints,frame):
    rows,cols,ch = frame.shape

    key_landmarks_eye1 = np.mean((imagePoints[0][36],imagePoints[0][37],imagePoints[0][38],imagePoints[0][39],imagePoints[0][40],imagePoints[0][41]),axis = 0)
    key_landmarks_eye2 = np.mean((imagePoints[0][42],imagePoints[0][43],imagePoints[0][44],imagePoints[0][45],imagePoints[0][46],imagePoints[0][47]),axis = 0)
    key_landmarks_mouth = np.mean((imagePoints[0][60],imagePoints[0][61],imagePoints[0][62],imagePoints[0][63],imagePoints[0][64],imagePoints[0][65],imagePoints[0][66],imagePoints[0][67]),axis = 0)
    #key_landmarks_nose = imagePoints[0][33]
    
    # creation of keylandmarks
    key_landmarks = np.zeros((3,3))   
    key_landmarks[0,:] = key_landmarks_eye1 
    key_landmarks[1,:] = key_landmarks_eye2 
    key_landmarks[2,:] = key_landmarks_mouth
    #key_landmarks[3,:] = key_landmarks_nose
    K = np.delete(key_landmarks,2,1)
    print(K)
    #savetxt('key_landmarks.csv',K, delimiter=',')

    pts1 = np.float32(loadtxt('key_landmarks.csv', delimiter=','))
    pts2 = np.float32(K)

    M = cv2.getAffineTransform(pts1,pts2)
    print(M)
    iM = cv2.invertAffineTransform(M)
    dst = cv2.warpAffine(frame,iM,(512,512))

    plt.subplot(121),plt.imshow(frame),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    #plt.show()

    directory = '../dataset/standardize_pic'
    os.chdir(directory) 
    cv2.imwrite('standardize.png',cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
    directory = '../../lib/facealignment'
    os.chdir(directory) 
