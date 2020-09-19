import cv2
import csv
import math
import face_alignment
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

from get_rotation_matrix import *

def get_distance_lips(fa, imagePoints):
    image_path = "../dataset/standardize_pic/standardize.png"
    frame = cv2.imread(image_path,1)

    # Get landmarks of the input image
    if(imagePoints):
        height, width ,aux= frame.shape
        x_upper_lips_center = imagePoints[0][62][0]
        x_lower_lips_center = imagePoints[0][66][0]
        y_upper_lips_center = imagePoints[0][62][1]
        y_lower_lips_center = imagePoints[0][66][1]
        distance = math.sqrt((x_lower_lips_center - x_upper_lips_center)**2 +(y_lower_lips_center - y_upper_lips_center)**2)
        return distance

    
    