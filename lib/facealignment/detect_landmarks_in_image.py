import face_alignment
import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from numpy import *
import numpy as np
from math import cos, sin

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
        # Get the array
        imagePoints = imagePoints[0]

        # Draw the computed data
        for imagePoint in imagePoints:
            # Draw the Point Predictions
            cv2.circle(frame, (imagePoint[0], imagePoint[1]), 3, (0,255,0))

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