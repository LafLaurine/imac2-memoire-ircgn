import csv
import pandas as pd
import os
from glob import glob

if os.path.exists('all_data.csv'):
    os.remove('all_data.csv')

all_files = glob("csv/*.csv")
combined_csv = pd.concat([pd.read_csv(f) for f in all_files])
combined_csv.to_csv("all_data.csv", index=False)