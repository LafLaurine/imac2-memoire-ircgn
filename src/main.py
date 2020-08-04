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

from get_rotation_matrix import *
from transform_image import *
from get_distance_lips import *

import sys
sys.path.append('../')
from lib.facealignment.detect_landmarks_in_image import *
from lib.FacialExpressionRecognition.visualize import get_expression

## Get arguments from user
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--filename', dest='filename', help='Path of image')
    args = parser.parse_args()
    return args

def draw(frame, imagePoints):
    # 2D-Plot
    plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)

    if(imagePoints is not None):
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

def main():
    # Face orientation
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False, face_detector='sfd')
    '''if(os.path.isdir(directory_path)):
        for filename in glob.glob(directory_path+'/*.jpg'):'''
    frame = cv2.imread(filename)
    print(frame)
    imagePoints = detect_landmarks(frame, fa)
    draw(frame,imagePoints)
    get_expression(frame)
    '''else:
        print("This directory doesn't exist")'''
    
if __name__ == '__main__':
    args = parse_args()
    filename = args.filename
    main()
    cv2.destroyAllWindows()
