import csv
import numpy as np
import collections
import sys
from os import listdir
from os.path import isfile, join
from glob import glob

directories = glob("extraction/extracted_faces/*/")

for directory in directories:
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    col_lips = np.float32(np.loadtxt(directory+'/lips_dist.txt', delimiter=','))
    col_angles = np.float32(np.loadtxt(directory+'/euler_angles.txt'))
    col_expression = np.float32(np.loadtxt(directory+'/expression.txt'))
    length = len(col_lips)
    print(length)

'''with open("all_data.csv", 'w') as outfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(['File_name','Lips_distance', 'Euler\'s_angles','Expression'])
    for i in range(length):
        # Add the data row
        csvwriter.writerow(['00000'+str(i+1)+'.jpg',col_lips[i] ,col_angles[i] ,col_expression[i]])'''
