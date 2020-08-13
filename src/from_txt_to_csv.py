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
onlyfiles = glob(directory_path+'/*.jpg')
onlyfiles.sort()

col_lips = np.float32(np.loadtxt(directory_path+'/lips_dist.txt', delimiter=','))
col_angles = np.float32(np.loadtxt(directory_path+'/euler_angles.txt'))
col_expression = np.float32(np.loadtxt(directory_path+'/expression.txt'))
col_center = np.float32(np.loadtxt(directory_path+'/center.txt'))
col_bounding = np.float32(np.loadtxt(directory_path+'/bounding_box.txt'))



base=os.path.basename(directory_path)

with open('csv/'+base+'.csv', 'w') as outfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(['File_name','Lips_distance', 'Euler\'s_angles','Expression','Center','Bounding_box'])
    for i in range(len(col_lips)):
        csvwriter.writerow([onlyfiles[i],col_lips[i],col_angles[i],col_expression[i],col_center[i],col_bounding[i]])

all_files = glob("csv/*.csv")
combined_csv = pd.concat([pd.read_csv(f) for f in all_files ])
combined_csv.to_csv( "all_data.csv", index=False)