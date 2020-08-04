import csv
import numpy as np
import os
import collections
import sys
col_lips = np.float32(np.loadtxt('lips_dist.txt', delimiter=','))
col_angles = np.float32(np.loadtxt('euler_angles.txt'))
col_expression = np.float32(np.loadtxt('expression.txt'))
lenth = len(col_lips)

with open("all_data.csv", 'w') as outfile:
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(['File_name','Lips_distance', 'Euler\'s_angles','Expression'])
    for i in range(lenth):
        # Add the data row
        csvwriter.writerow(['00000'+str(i+1)+'.jpg',col_lips[i] ,col_angles[i] ,[col_expression[7*i],col_expression[7*i+1],col_expression[7*i+2],col_expression[7*i+3],col_expression[7*i+4],col_expression[7*i+5],col_expression[7*i+6]]])