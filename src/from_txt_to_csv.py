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
            # writing the data rows  
            csvwriter.writerow(["pic name"])
            csvwriter.writerows([['frame30.jpg']])
            csvwriter.writerows([['frame60.jpg']])
            csvwriter.writerows([['frame90.jpg']])
            csvwriter.writerows([['frame120.jpg']])
            csvwriter.writerows([['frame150.jpg']])
            # writing the fields  
            csvwriter.writerow(["lips distance"])  
            csvwriter.writerows([col_lips]) 
            csvwriter.writerow(["euler's angles"])
            csvwriter.writerows([col_angles])
            csvwriter.writerow(["expression"])
            csvwriter.writerows([col_expression])