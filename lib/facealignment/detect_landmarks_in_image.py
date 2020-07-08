import face_alignment
import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from numpy import *
import numpy as np

import sys
sys.path.append('../../')
from src.get_rotation_matrix import *

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

    if(imagePoints is not None):
        Q = get_rotation_matrix(imagePoints)
        #get rotation from rotation matrix
        (theta, phi, psi) = rotationMatrixToEulerAngles(Q)
        print(theta,phi,psi)
        # Get the array
        imagePoints = imagePoints[0]        
        # we have the rotation matrix
        x_axis = Q[:,0]
        y_axis = Q[:,1]
        z_axis = Q[:,2]

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
    '''for pred_type in pred_types.values():
        ax.plot(imagePoints[pred_type.slice, 0],imagePoints[pred_type.slice, 1],color=pred_type.color, **plot_style)'''

    ax.axis('off')

    # 3D-Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(imagePoints[:, 0] * 1.2,
                    imagePoints[:, 1],
                    imagePoints[:, 2],
                    c='cyan',
                    alpha=1.0,
                    edgecolor='b')

    '''for pred_type in pred_types.values():
        ax.plot3D(imagePoints[pred_type.slice, 0] * 1.2,
                imagePoints[pred_type.slice, 1],
                imagePoints[pred_type.slice, 2], color='blue')'''

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()
        
if __name__ == '__main__':
    args = parse_args()
    image_path = "../../"+args.image_path
    main()
    cv2.destroyAllWindows()
