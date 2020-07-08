import face_alignment
import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from numpy import *
import numpy as np
from math import cos, sin
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--image', dest='image_path', help='Path of image')
    args = parser.parse_args()
    return args

def main():
    # Run the 3D face alignment with CPU on a test image : change cpu to cuda
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False, face_detector='sfd')

    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2D-Plot
    plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)

    # Get landmarks of the input image
    imagePoints = fa.get_landmarks_from_image(frame)

    # row to delete
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
    #test with estimateAffine3D
    src_point = loadtxt('data.csv', delimiter=',')
    dst_point = imagePoints[0] 
    (retval ,E,inliers)= cv2.estimateAffine3D(src_point,dst_point)
    E = np.delete(E,3,1)
    print("with estimateAffine3D" , E)
    # we extract matrix N
    N = np.delete(M,3,0)
    N = np.delete(N,3,1)
    print("with our method" ,N)

    print("the difference" , E-N)
    # QR decomposition
    Q, R = np.linalg.qr(N)
    print(Q)
    print(R)
    # we have the rotation matrix
    x_axis = Q[:,0]
    y_axis = Q[:,1]
    z_axis = Q[:,2]

    if(imagePoints is not None):
        # Get the array
        imagePoints = imagePoints[0]
        # Compute the Mean-Centered-Scaled Points
        mean = np.mean(imagePoints, axis=0)

        # Draw the computed data
        for imagePoint in imagePoints:
            # Draw the Point Predictions
            cv2.circle(frame, (imagePoint[0], imagePoint[1]), 3, (0,255,0))

    # x_axis
    cv2.line(frame, tuple(mean[:2].astype(int)), 
                        tuple((mean+(x_axis * 100.0))[:2].astype(int)), (0, 0, 255), 3)

    # y_axis                   
    cv2.line(frame, tuple(mean[:2].astype(int)), 
                        tuple((mean-(y_axis * 100.0))[:2].astype(int)), (0, 255, 0), 3)
    # z_axis                   
    cv2.line(frame, tuple(mean[:2].astype(int)), 
                        tuple((mean+(z_axis * 100.0))[:2].astype(int)), (255, 0, 0), 3)

    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'nose': pred_type(slice(26, 31), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3))
                }    

    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(frame)

    # that's the part where we draw lines
    for pred_type in pred_types.values():
        ax.plot(imagePoints[pred_type.slice, 0],
                imagePoints[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    # 3D-Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(imagePoints[:, 0] * 1.2,
                    imagePoints[:, 1],
                    imagePoints[:, 2],
                    c='cyan',
                    alpha=1.0,
                    edgecolor='b')

    for pred_type in pred_types.values():
        ax.plot3D(imagePoints[pred_type.slice, 0] * 1.2,
                imagePoints[pred_type.slice, 1],
                imagePoints[pred_type.slice, 2], color='blue')

    
    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()
    

        
if __name__ == '__main__':
    args = parse_args()
    image_path = "../../"+args.image_path
    main()
    cv2.destroyAllWindows()
