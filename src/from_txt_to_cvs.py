import csv
import numpy as np
import os
import collections
import sys

col_lips = np.float32(np.loadtxt('lips_dist.txt', delimiter=','))
col_angles = np.float32(np.loadtxt('euler_angles.txt'))
col_expression = np.float32(np.loadtxt('expression.txt'))


with open("all_data.csv", 'w') as csvfile:  
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile)
            # writing the fields  
            csvwriter.writerow(["lips distance"])  
            # writing the data rows  
            csvwriter.writerows([col_lips]) 

            csvwriter.writerow(["euler's angles"])

            csvwriter.writerows([col_angles])

            csvwriter.writerow(["expression"])

            csvwriter.writerows([col_expression])





        