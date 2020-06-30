import face_alignment
import argparse
import dlib
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from imutils import face_utils
import numpy as np
from math import cos, sin

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--image', dest='image_path', help='Path of image')
    args = parser.parse_args()
    return args

def main():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=True)

    frame =  cv2.imread(image_path)

    preds = fa.get_landmarks(frame)[-1]
    # Clear the indices frame : create an array filled with zero, with the size of frame
    canonical = np.zeros(frame.shape)
    # Get landmarks of the input image
    imagePoints = fa.get_landmarks_from_image(frame)

    fa3D = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False)
    model_points = fa3D.get_landmarks_from_image(frame)
    model_points = model_points[0]

    size = frame.shape

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    print ("Camera Matrix :\n {0}".format(camera_matrix))
    B = np.reshape(imagePoints, (-1, 2))
    X = np.asmatrix(B)

    model = np.asmatrix(model_points)

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model, X, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    print ("Rotation Vector:\n {0}".format(rotation_vector))
    print ("Translation Vector:\n {0}".format(translation_vector))
    
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    for p in X:
        cv2.circle(frame, (p[0].astype(int), p[1].astype(int)), 3, (0,0,255), -1)

    p1 = ( int(X[0][0]), int(X[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    
    cv2.line(frame, p1, p2, (255,0,0), 2) 
    # Display image
    cv2.imshow("Output", frame)
    cv2.waitKey(0)

    while(1):
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

        
if __name__ == '__main__':
    args = parse_args()
    image_path = args.image_path
    main()
    cv2.destroyAllWindows()