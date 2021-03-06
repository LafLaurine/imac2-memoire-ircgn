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
    parser = argparse.ArgumentParser(description='Detecting landmarks')
    parser.add_argument('--directory', dest='directory_path', help='Path of directory', required=True)
    args = parser.parse_args()
    return args

def draw(filename, frame, imagePoints, fa):
    if(imagePoints):
        perspective_trans(imagePoints,frame)
        distancelips = get_distance_lips(fa, imagePoints)
        if(distancelips):
            with open(directory_path+"/lips_dist.txt", "ab") as f:
                np.savetxt(f, [distancelips])
        else:
            with open(directory_path+"/lips_dist.txt", "ab") as f:
                np.savetxt(f, [0])
        N,Q,R = get_rotation_matrix(imagePoints)
        (theta, phi, psi) = rotationMatrixToEulerAngles(Q) * 180 / np.pi
        if(theta,phi,psi):
            with open(directory_path+"/theta.txt", "ab") as f:
                np.savetxt(f, [theta])
            with open(directory_path+"/phi.txt", "ab") as f:
                np.savetxt(f, [phi]) 
            with open(directory_path+"/psi.txt", "ab") as f:
                np.savetxt(f, [psi]) 
        else:
            with open(directory_path+"/theta.txt", "ab") as f:
                np.savetxt(f, [theta])
            with open(directory_path+"/phi.txt", "ab") as f:
                np.savetxt(f, [phi]) 
            with open(directory_path+"/psi.txt", "ab") as f:
                np.savetxt(f, [psi]) 
        x_axis = Q[:,0]
        y_axis = Q[:,1]
        z_axis = Q[:,2]
        imagePoints = imagePoints[0]        
        mean = np.mean(imagePoints, axis=0)
    else:
        os.remove(directory_path+"/"+filename)

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
        draw(filenames[i], frame, imagePoints, fa)
        if(imagePoints):
            expr = get_expression(frame)
            expression1 = expr[0]
            expression2 = expr[1]
            expression3 = expr[2]
            expression4 = expr[3]
            expression5 = expr[4]
            expression6 = expr[5]
            expression7 = expr[6]
            result = create_mask(frame)
            dirpath = os.path.split(os.path.split(directory_path)[1])[1]
            if not os.path.exists('extraction/masks'):
                os.makedirs('extraction/masks')
            cv2.imwrite('extraction/masks/'+dirpath+str(i)+'.jpg',result)
            if(expr):
                with open(directory_path+"/expression1.txt", "ab") as f:
                    np.savetxt(f, [expression1])
                with open(directory_path+"/expression2.txt", "ab") as f:
                    np.savetxt(f, [expression2])
                with open(directory_path+"/expression3.txt", "ab") as f:
                    np.savetxt(f, [expression3])
                with open(directory_path+"/expression4.txt", "ab") as f:
                    np.savetxt(f, [expression4])
                with open(directory_path+"/expression5.txt", "ab") as f:
                    np.savetxt(f, [expression5])
                with open(directory_path+"/expression6.txt", "ab") as f:
                    np.savetxt(f, [expression6])
                with open(directory_path+"/expression7.txt", "ab") as f:
                    np.savetxt(f, [expression7])
        
    
if __name__ == '__main__':
    args = parse_args()
    directory_path = args.directory_path
    main()
    cv2.destroyAllWindows()
