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
from mask import create_mask

import sys
sys.path.append('../')
from lib.facealignment.detect_landmarks_in_image import *
from lib.FacialExpressionRecognition.visualize import get_expression

vailed_ext = [".jpg",".png"]
f_list = []

## Get arguments from user
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--directory', dest='directory_path', help='Path of directory', required=True)
    args = parser.parse_args()
    return args

def draw(filename, frame, imagePoints):
    if(imagePoints is None):
        os.remove(directory_path+"/"+filename)
        print("Deleted... %s", filename)
    if(imagePoints is not None):
        perspective_trans(imagePoints,frame)
        distancelips = get_distance_lips()
        if(distancelips):
            with open(directory_path+"/lips_dist.txt", "ab") as f:
                np.savetxt(f, [distancelips])
        N,Q,R = get_rotation_matrix(imagePoints)
        (theta, phi, psi) = rotationMatrixToEulerAngles(Q) * 180 / np.pi
        if((theta, phi, psi)):
            with open(directory_path+"/euler_angles.txt", "ab") as f:
                    np.savetxt(f, [[theta,phi,psi]])
        x_axis = Q[:,0]
        y_axis = Q[:,1]
        z_axis = Q[:,2]
        imagePoints = imagePoints[0]        
        # compute the Mean-Centered-Scaled Points
        mean = np.mean(imagePoints, axis=0)

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return filenames, images

def main():
    filenames, images = load_images_from_folder(directory_path)
    filenames.sort()
    length = len(images)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False, face_detector='sfd')
    for i in range(length):
        frame = images[i]
        print(filenames[i])
        imagePoints = detect_landmarks(frame, fa)
        draw(filenames[i], frame, imagePoints)
        expr = get_expression(frame)
        result = create_mask(frame)
        dirpath = os.path.split(os.path.split(directory_path)[1])[1]
        cv2.imwrite('extraction/masks/'+dirpath+str(i)+'.jpg',result)
        if(expr):
            with open(directory_path+"/expression.txt", "ab") as f:
                np.savetxt(f, [expr],  delimiter=' ')   
    
if __name__ == '__main__':
    args = parse_args()
    directory_path = args.directory_path
    main()
    cv2.destroyAllWindows()
