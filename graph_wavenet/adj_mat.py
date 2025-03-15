
import pandas as pd 
import numpy as np
from scipy.spatial.distance import euclidean

def directed_binary_adj(elevation):
    # This only works for simple topography
    # Direction is general groundwater flow downhill
    
    temp = [e - elevation for e in elevation ]
    ele_diff = np.array(temp)
    ele_diff = np.identity(len(ele_diff))+ele_diff # the nodes are connected to itself

    adj_binary = np.where(ele_diff < 0, ele_diff, 1) # it is connected to itself
    adj_binary = np.where(ele_diff > 0, adj_binary, 0) # difference is less than 0
    
    return adj_binary

def weighted_directed_adj(east, north, elevation):
    
    # weight this binary adj matrix with distance with easting, northing, and elevation parameters
    # this take a pairwaise euclidean distance and weight it using np.exp(-np.square(distance_matrix/std))
    
    mat = pd.concat([east, north, elevation], axis = 1).to_numpy()
    mat_trans = mat.T   
    
    l2=[]
    for i in range(len(mat)):
        temp= []
        for j in range(len(mat)):
            temp.append(np.abs(euclidean(mat[i,:], mat_trans[:,j])))
        l2.append(temp)

    A = np.vstack(l2) 
    scaling = A.max()# for scaling 
    
    adj_binary = directed_binary_adj(elevation)
    
    dis_adj = np.exp(-np.square(A/scaling))
    weighted_adj = dis_adj*adj_binary
    
    return weighted_adj

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    