import cv2
import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import transform
from skimage import img_as_float

from src.get_rotation_matrix import *


def get_affine_matrix_keylandmarks(imagePoints):
	
    key_landmarks_eye1 = np.mean((imagePoints[0][36],imagePoints[0][37],imagePoints[0][38],imagePoints[0][39],imagePoints[0][40],imagePoints[0][41]),axis = 0)
    key_landmarks_eye2 = np.mean((imagePoints[0][42],imagePoints[0][43],imagePoints[0][44],imagePoints[0][45],imagePoints[0][46],imagePoints[0][47]),axis = 0)
    key_landmarks_mouth = np.mean((imagePoints[0][60],imagePoints[0][61],imagePoints[0][62],imagePoints[0][63],imagePoints[0][64],imagePoints[0][65],imagePoints[0][66],imagePoints[0][67]),axis = 0)
    
    # creation of keylandmarks
    key_landmarks = np.zeros((3,3))   
    key_landmarks[0,:] = key_landmarks_eye1 
    key_landmarks[1,:] = key_landmarks_eye2 
    key_landmarks[2,:] = key_landmarks_mouth 

    #print(key_landmarks)
    #savetxt('key_landmarks.csv',key_landmarks, delimiter=',')
    A = np.asmatrix(loadtxt('key_landmarks.csv', delimiter=','))
    B = np.asmatrix(key_landmarks)
    
    # create column and row to add 
    columnofones = np.ones((3,1), dtype=int64)
    rowtoaddtoM = [0,0,0,1]

    # we need to find the refined matrix that transform best A to B
    A = np.append(A,columnofones,1)

    # we extract the x,y and z from matrix B
    vector_xp = B[:,0]
    vector_yp = B[:,1]
    vector_zp = B[:,2]

    # we solve the overdetermined system
    M1 = np.linalg.pinv(A).dot(vector_xp)
    M2 = np.linalg.pinv(A).dot(vector_yp)
    M3 = np.linalg.pinv(A).dot(vector_zp)

    # we construct matrix M
    M = np.zeros((4,4))
    M[0,:] = M1.transpose()
    M[1,:] = M2.transpose()
    M[2,:] = M3.transpose()
    M[3,:] = rowtoaddtoM
    
    # we extract matrix N
    N = np.delete(M,3,0)
    N = np.delete(N,3,1)
    # QR decomposition
    Q, R = np.linalg.qr(N)
    
    print(N)
    return N,Q,R

def affine_trans_image(matrix, img_path):
	img = img_as_float(img_path)
	print(matrix)
	tform = transform.ProjectiveTransform(matrix)
	print(tform)
	tf_img = transform.warp(img, tform.inverse)
	fig, ax = plt.subplots()
	ax.imshow(tf_img)
	ax.set_title('Projective transformation')
	plt.show()

def affine_trans_2(matrix, img_path):
	img = img_path
	rows,cols,ch = img.shape
	Minv = np.linalg.inv(matrix)
	Minv = np.delete(Minv,2,0)
	print(Minv)
	dst = cv2.warpAffine(img,Minv,(cols,rows))

	cv2.imshow('img',dst)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()