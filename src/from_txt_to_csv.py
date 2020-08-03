import csv
import numpy as np
import os
import collections
import sys

col_lips = np.float32(np.loadtxt('lips_dist.txt', delimiter=','))
col_angles = np.float32(np.loadtxt('euler_angles.txt'))
col_expression = np.float32(np.loadtxt('expression.txt'))


with open("data.csv", 'w') as outfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(['File_name','Lips_distance', 'Euler\'s_angles','Expression'])
    csvwriter.writerows([['000001.jpg']])
    csvwriter.writerows([['000002.jpg']])
    csvwriter.writerows([['000003.jpg']])
    csvwriter.writerows([['000004.jpg']])
    csvwriter.writerows([['000005.jpg']])
    csvwriter.writerows([col_lips])
    csvwriter.writerows([col_angles])
    csvwriter.writerows([col_expression])