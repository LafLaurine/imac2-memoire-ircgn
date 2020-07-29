import face_alignment
import argparse
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from numpy import *
import glob
import numpy as np

import sys
sys.path.append('../../')
from src.get_rotation_matrix import *
from src.transform_image import *
from src.lips import *

## Get arguments from user
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--directory', dest='directory_path', help='Path of directory')
    args = parser.parse_args()
    return args

def draw(frame, imagePoints):
    # 2D-Plot
    plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)

    if(imagePoints is not None):
        #N,Q,R = get_affine_matrix_keylandmarks(imagePoints)
        perspective_trans(imagePoints,frame)
        distancelips = get_distance_lips()
        print('distance lips = ' ,distancelips)
        N,Q,R = get_rotation_matrix(imagePoints)

        x_axis = Q[:,0]
        y_axis = Q[:,1]
        z_axis = Q[:,2]
        imagePoints = imagePoints[0]        
        # compute the Mean-Centered-Scaled Points
        mean = np.mean(imagePoints, axis=0)
        # draw the computed data
        for imagePoint in imagePoints:
            # draw the Point Predictions
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


def main():
    # Run the 3D face alignment with CPU on a test image : change cpu to cuda
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False, face_detector='sfd')
    for dirs in glob.glob(directory_path):
        for filename in os.listdir(dirs):
            frame = cv2.imread(os.path.join(dirs, filename),1)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Get landmarks of the input image
                imagePoints = fa.get_landmarks_from_image(frame)
                draw(frame,imagePoints)
    
if __name__ == '__main__':
    args = parse_args()
    directory_path = "../../"+args.directory_path
    main()
    cv2.destroyAllWindows()
