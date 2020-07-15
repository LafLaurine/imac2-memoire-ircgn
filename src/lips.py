import cv2
import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import transform
from skimage import img_as_float
from scipy import ndimage, misc

from src.get_rotation_matrix import *

def get_distance_lips(imagePoints,frame):
    height, width ,aux= frame.shape
    x_upper_lips_center = imagePoints[0][62][0]
    x_lower_lips_center = imagePoints[0][66][0]
    y_upper_lips_center = imagePoints[0][62][1]
    y_lower_lips_center = imagePoints[0][66][1]
    distance = math.sqrt((x_lower_lips_center - x_upper_lips_center)**2 +(y_lower_lips_center - y_upper_lips_center)**2) 
    alpha= 225/height
    return (alpha * distance)

    
    