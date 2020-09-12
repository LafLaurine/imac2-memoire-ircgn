import csv
import pandas as pd
import os
import argparse
import numpy as np
from glob import glob

## Get arguments from user
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Create CSV')
    parser.add_argument('--directory', dest='directory_path', help='Path of directory', required=True)
    args = parser.parse_args()
    return args

args = parse_args()
directory_path = args.directory_path
dirpath = os.path.split(os.path.split(directory_path)[1])[1]
onlyfiles = glob('extraction/masks/'+dirpath+'*.jpg')
onlyfiles.sort()

col_lips = np.genfromtxt(directory_path+'/lips_dist.txt')
col_theta= np.genfromtxt(directory_path+'/theta.txt')
col_phi= np.genfromtxt(directory_path+'/phi.txt')
col_psi= np.genfromtxt(directory_path+'/psi.txt')
col_expression1 = np.genfromtxt(directory_path+'/expression1.txt')
col_expression2 = np.genfromtxt(directory_path+'/expression2.txt')
col_expression3 = np.genfromtxt(directory_path+'/expression3.txt')
col_expression4 = np.genfromtxt(directory_path+'/expression4.txt')
col_expression5 = np.genfromtxt(directory_path+'/expression5.txt')
col_expression6 = np.genfromtxt(directory_path+'/expression6.txt')
col_expression7 = np.genfromtxt(directory_path+'/expression7.txt')
col_center = np.genfromtxt(directory_path+'/center.txt')
col_bounding = np.genfromtxt(directory_path+'/bounding_box.txt')

base = os.path.basename(directory_path)

precision = 4

with open('csv/'+base+'.csv', 'w') as outfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(['File_name','Lips_distance','Theta','Phi','Psi','Expression1','Expression2','Expression3','Expression4','Expression5','Expression6','Expression7','Center1','Center2','Bounding_box'])
    for i in range(len(col_lips)):
        nbCenter = col_center[i]
        nbCenter[0] = round(nbCenter[0],precision)
        nbCenter[1] = round(nbCenter[1],precision)
        center1 = nbCenter[0]
        center2 = nbCenter[1]
        csvwriter.writerow([onlyfiles[i],col_lips[i],col_theta[i],col_phi[i],col_psi[i],col_expression1[i],col_expression2[i],col_expression3[i],col_expression4[i],col_expression5[i],col_expression6[i],col_expression7[i],center1,center2,col_bounding[i]])