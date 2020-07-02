import face_alignment
import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from numpy import *
import numpy as np
from math import cos, sin
from rigid_transform_3D import rigid_transform_3D


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--image', dest='image_path', help='Path of image')
    args = parser.parse_args()
    return args

def main():
    # Run the 3D face alignment with CUDA on a test image : change to cpu to test with your cpu.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)

    frame =  cv2.imread(image_path)

    preds = fa.get_landmarks(frame)[-1]
    # 2D-Plot
    plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)


    # Clear the indices frame : create an array filled with zero, with the size of frame
    canonical = np.zeros(frame.shape)
    # Get landmarks of the input image
    imagePoints = fa.get_landmarks_from_image(frame)
    if(imagePoints is not None):
        # Get the array
        imagePoints = imagePoints[0]
        # Compute the Mean-Centered-Scaled Points
        mean = np.mean(imagePoints, axis=0)
        scaled = (imagePoints / np.linalg.norm(imagePoints[42] - imagePoints[39])) * 0.06 # Set the inner eye distance to 6cm
        # Scaled the mean
        centered = scaled - np.mean(scaled, axis=0)

        # Transform array of landmarks points to matrixes
        '''
        X = np.asmatrix(imagePoints)
        y = np.asmatrix(imagePoints)
        # Get the number of column
        n = X.shape[1]
        # Get the rank of the matrice
        r = np.linalg.matrix_rank(X)
        # Find the equivalent to our matrix of features using singular value decomposition
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)
        # D^+ can be derived from sigma
        D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))
        # V is equal to the transpose of its transpose
        V = VT.T
        X_plus = V.dot(D_plus).dot(U.T)
        # Least square solution
        w = X_plus.dot(y)
        # Error of the least square solution
        error = np.linalg.norm(X.dot(w) - y, ord=2) ** 2

        A = np.asmatrix(imagePoints)
        B = np.asmatrix(imagePoints)
        ret_R, ret_t = rigid_transform_3D(w,w)

        print("Recovered rotation")
        print(ret_R)
        print("")

        print("Recovered translation")
        print(ret_t)
        print("")'''

        # Construct a "rotation" matrix
        rotationMatrix = np.empty((3,3))
        # eye line
        rotationMatrix[0,:] = (centered[16] - centered[0])/np.linalg.norm(centered[16] - centered[0])
        # nose line
        rotationMatrix[1,:] = (centered[8] - centered[27])/np.linalg.norm(centered[8] - centered[27])
        # cross product of eye line and nose line in order to have projection axe
        rotationMatrix[2,:] = np.cross(rotationMatrix[0, :], rotationMatrix[1, :])

        # Draw the X Axis
        cv2.line(frame, tuple(mean[:2].astype(int)), 
                        tuple((mean+(rotationMatrix[0,:] * 100.0))[:2].astype(int)), (0, 0, 255), 3)
        # Draw the Y Axis
        cv2.line(frame, tuple(mean[:2].astype(int)), 
                        tuple((mean-(rotationMatrix[1,:] * 100.0))[:2].astype(int)), (0, 255, 0), 3)
        # Draw the Z Axis
        cv2.line(frame, tuple(mean[:2].astype(int)), 
                        tuple((mean+(rotationMatrix[2,:] * 100.0))[:2].astype(int)), (255, 0, 0), 3)

    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }

    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(frame)

    for pred_type in pred_types.values():
        ax.plot(preds[pred_type.slice, 0],
                preds[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    # 3D-Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds[:, 0] * 1.2,
                    preds[:, 1],
                    preds[:, 2],
                    c='cyan',
                    alpha=1.0,
                    edgecolor='b')

    for pred_type in pred_types.values():
        ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                preds[pred_type.slice, 1],
                preds[pred_type.slice, 2], color='blue')

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()

        
if __name__ == '__main__':
    args = parse_args()
    image_path = args.image_path
    main()
    cv2.destroyAllWindows()