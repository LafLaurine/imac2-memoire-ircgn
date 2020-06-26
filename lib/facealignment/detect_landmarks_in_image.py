import face_alignment
import argparse
import sys
import cv2
sys.path.append('..')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import numpy as np
from math import cos, sin

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--image', dest='image_path', help='Path of image')
    args = parser.parse_args()
    return args

def main():
    # Run the 3D face alignment with CUDA on a test image : change to cpu to test with your cpu.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=True)

    input_img = io.imread(image_path)

    preds = fa.get_landmarks(input_img)[-1]
    # 2D-Plot
    plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)

    frame =  cv2.imread(image_path)
    # Clear the indices frame
    canonical = np.zeros(frame.shape)
    imagePoints = fa.get_landmarks_from_image(input_img)
    if(imagePoints is not None):
        imagePoints = imagePoints[0]
        # Compute the Mean-Centered-Scaled Points
        mean = np.mean(imagePoints, axis=0) # <- This is the unscaled mean
        scaled = (imagePoints / np.linalg.norm(imagePoints[42] - imagePoints[39])) * 0.06 # Set the inner eye distance to 6cm
        centered = scaled - np.mean(scaled, axis=0) # <- This is the scaled mean

        # Construct a "rotation" matrix 
        rotationMatrix = np.empty((3,3))
        rotationMatrix[0,:] = (centered[16] - centered[0])/np.linalg.norm(centered[16] - centered[0])
        rotationMatrix[1,:] = (centered[8] - centered[27])/np.linalg.norm(centered[8] - centered[27])
        rotationMatrix[2,:] = np.cross(rotationMatrix[0, :], rotationMatrix[1, :])
        invRot = np.linalg.inv(rotationMatrix)

        # Object-space points, these are what you'd run OpenCV's solvePnP() with
        objectPoints = centered.dot(invRot)

        # Draw the computed data
        for i, (imagePoint, objectPoint) in enumerate(zip(imagePoints, objectPoints)):
            # Draw the Point Predictions
            cv2.circle(frame, (imagePoint[0], imagePoint[1]), 3, (0,255,0))

            # Draw the X Axis
            cv2.line(frame, tuple(mean[:2].astype(int)), 
                            tuple((mean+(rotationMatrix[0,:] * 100.0))[:2].astype(int)), (0, 0, 255), 3)
            # Draw the Y Axis
            cv2.line(frame, tuple(mean[:2].astype(int)), 
                            tuple((mean-(rotationMatrix[1,:] * 100.0))[:2].astype(int)), (0, 255, 0), 3)
            # Draw the Z Axis
            cv2.line(frame, tuple(mean[:2].astype(int)), 
                            tuple((mean+(rotationMatrix[2,:] * 100.0))[:2].astype(int)), (255, 0, 0), 3)
            
            # Draw the indices in Object Space
            cv2.putText(canonical, str(i), 
                        ((int)((objectPoint[0] * 1000.0) + 320.0), 
                            (int)((objectPoint[1] * 1000.0) + 240.0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    #cv2.imshow('Webcam View', frame)

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