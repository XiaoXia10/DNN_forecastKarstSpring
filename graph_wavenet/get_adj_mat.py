# Get an adj mat for GNN training

import pandas as pd
import numpy as np 
from os import chdir

chdir(r'G:\My Drive\Neuchatel_Project\Betteraz\python_codes')
from adj_mat import weighted_directed_adj

df = pd.read_csv(r'betteraz_mod_points.csv', index_col=0)

east= df['easting']
north = df['northing']
elevation = df['elevation'] 

weighted_adj = weighted_directed_adj(east, north, elevation)


