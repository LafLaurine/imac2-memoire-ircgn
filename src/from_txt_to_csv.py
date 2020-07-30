import csv
import numpy as np
import os
import collections
import sys

col_lips = np.float32(np.loadtxt('lips_dist.txt', delimiter=','))
col_angles = np.float32(np.loadtxt('euler_angles.txt'))
col_expression = np.float32(np.loadtxt('expression.txt'))


with open("all_data.csv", 'w') as outfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(outfile, delimiter=',')
    csvwriter.writerow(['File_name','Lips_distance', 'Euler\'s_angles'])
    csvwriter.writerows([['frame30.jpg','frame60.jpg','frame90.jpg','frame120.jpg','frame150.jpg']])
    csvwriter.writerows([col_lips])
    csvwriter.writerows([col_angles])