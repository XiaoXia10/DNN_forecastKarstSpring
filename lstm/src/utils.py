#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:52:59 2026

@author: xl3138
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from durbango import pickle_save
import pandas as pd
from os.path import join, dirname, abspath

def destandardize_pred(args, df):

    data = pd.read_csv(join(args.data_dir, "train_val_data.csv"), index_col=0, parse_dates=True)
    std = data.std()
    mean = data.mean()

    df_copy = df.copy()
    df_destand = (df_copy*std.values)+mean.values

    return df_destand

def standardize_df(args, df):

    data = pd.read_csv(join(args.data_dir, "train_val_data.csv"), index_col=0, parse_dates=True)
    std = data.std()
    mean = data.mean()
    
    df_copy = df.copy()
    df_std = (df_copy-mean.values)/std.values

    return df_std

def log_df(df):

    df = np.log10(df)
    
    return df

# def exp10_pred(args, df):
    
#     data = pd.read_csv(join(args.dir, "df.csv"), index_col=0, parse_dates=True)
    
#     yhat = np.power(10.0, yhat)
#     realy = np.power(10.0, realy)
    
#     yhat = pd.DataFrame(yhat)

    
#     yhat.index = time
#     realy.index = time
    
#     return yhat, realy
    

def data_loader(args, loader):
    
    x = np.load(join(args.output_dir, "x_"+loader+".npy"))
    y = np.load(join(args.output_dir, "y_"+loader+".npy"))    

    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    
    train_dataset = TensorDataset(x, y)
    
    if loader == "train":
        
        data_loader = DataLoader(
                                  dataset=train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True
                                 )
        
    else:
        data_loader = DataLoader(
                                  dataset=train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=False
                                 )
    return data_loader

def get_shared_arg_parser():
    
    freq = "12h" ##[1h, 4h, 8h, 12h, 24h]
    
    ###[karst_data, sw_data, gw_data]
    # dataset = "sw_data"

    seq_length_x = 6 #[3,6,12]
    seq_length_y = 6
    shift = 6
    
    current_dir = dirname(abspath(__file__))
    base_dir = dirname(current_dir)
    
    data_dir = join(base_dir, "data", freq)
    output_dir = join(data_dir, "timestep"+str(seq_length_x)+str(seq_length_y)+str(shift))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size') 
    parser.add_argument('--input_dim', type=int, default=4, help='This is the number of input features in the data') 
    parser.add_argument('--output_dim', type=int, default=4, help='This is the number of output features from model prediction') 
    parser.add_argument('--hidden_dim', type=int, default=10, help='Number of hidden dimensions') 
    parser.add_argument('--num_layers', type=int, default=3, help='Number of model layers')
    
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Recurrent dropout') 
    
    parser.add_argument('--patience', type=int, default=20, help='quit if no improvement after this many iterations')
    
    parser.add_argument("--data_dir", type=str, default=data_dir, help="Data directory")
    
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Data for model training and model output.",)
    
    args = parser.parse_args()
    # pickle_save(args, f'{args.dir}/{args.data_dir}/args.pkl')
    
    return args