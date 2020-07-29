from numpy import *
import numpy as np
import cv2

## @package get_rotation_matrix
#  Documentation for get_rotation_matrix

## Get rotation matrix
#  Allow us to retrieve rotation matrix from affine transformation
def get_rotation_matrix(imagePoints):
    
    
    ## row to delete
    listofrowtodelete = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,31,32,33,34,35,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69])
    imagePoints[0] = np.delete(imagePoints[0],listofrowtodelete,0)
    
    #savetxt('data.csv', imagePoints[0], delimiter=',')
    
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

    #verif with cv2
    #pts1 = np.asmatrix(loadtxt('data.csv', delimiter=','))
    #pts2 = np.asmatrix(imagePoints[0])
    #Np = cv2.getAffineTransform(pts1,pts2)
    #print(Np)
    # QR decomposition
    Q, R = np.linalg.qr(N)
    print(N)
    print(Q)
    print(R)
    # get rotation from rotation matrix
    (theta, phi, psi) = rotationMatrixToEulerAngles(Q) * 180 / np.pi
    print(theta,phi,psi)
    with open("../../src/euler_angles.txt", "ab") as f:
            np.savetxt(f, [theta,phi,psi])
    return N,Q,R

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
