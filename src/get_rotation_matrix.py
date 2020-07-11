from numpy import *
import numpy as np
import cv2

## @package get_rotation_matrix
#  Documentation for get_rotation_matrix

## Get rotation matrix
#  Allow us to retrieve rotation matrix from affine transformation
def get_rotation_matrix(imagePoints):
    
    key_landmarks_eye1 = np.mean((imagePoints[0][36],imagePoints[0][37],imagePoints[0][38],imagePoints[0][39],imagePoints[0][40],imagePoints[0][41]),axis = 0)
    key_landmarks_eye2 = np.mean((imagePoints[0][42],imagePoints[0][43],imagePoints[0][44],imagePoints[0][45],imagePoints[0][46],imagePoints[0][47]),axis = 0)
    key_landmarks_mouth = np.mean((imagePoints[0][60],imagePoints[0][61],imagePoints[0][62],imagePoints[0][63],imagePoints[0][64],imagePoints[0][65],imagePoints[0][66],imagePoints[0][67]),axis = 0)
    ## row to delete
    listofrowtodelete = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,31,32,33,34,35,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69])
    imagePoints[0] = np.delete(imagePoints[0],listofrowtodelete,0)
    
    key_landmarks = np.zeros((3,3))   
    key_landmarks[0,:] = key_landmarks_eye1 
    key_landmarks[1,:] = key_landmarks_eye2 
    key_landmarks[2,:] = key_landmarks_mouth 

    print(key_landmarks)
    #savetxt('data.csv', imagePoints[0], delimiter=',')
    #savetxt('key_landmarks.csv',key_landmarks, delimiter=',')
    
    A = np.asmatrix(loadtxt('data.csv', delimiter=','))
    B = np.asmatrix((imagePoints[0]))
    
    # create column and row to add 
    columnofones = np.ones((18,1), dtype=int64)
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
    print(Q)
    print(R)
    # get rotation from rotation matrix
    (theta, phi, psi) = rotationMatrixToEulerAngles(Q) * 180 / np.pi
    print(theta,phi,psi)
    return Q

## Check if a matrix is a valid rotation matrix
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

## Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])