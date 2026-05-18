#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:21:28 2026

@author: xl3138
"""

import torch
from bayesian_lstm import BayesianLSTM
from utils import data_loader, get_shared_arg_parser
from os.path import join
import numpy as np
import pandas as pd


def predict_bayesian_dropout(model, mean, std, input_data, num_real):
    model.train() # Crucial: set model to training mode during inference
    predictions = []

    for _ in range(num_real):
        with torch.no_grad():
            output = model(input_data)
            output_destand = (output*std)+mean
            predictions.append(output_destand)
            
    # Compute mean, variance and standard deviation of the number of realizations
    predictions = torch.stack(predictions)
    mean_prediction = predictions.mean(dim=0)
    std_prediction = predictions.std(dim=0)
    variance_prediction = predictions.var(dim=0)
    
                                     
    return mean_prediction, variance_prediction
def get_uncertainty_prediction(args):
    

    x = np.load(join(args.output_dir, "x_test.npy"))
    y = np.load(join(args.output_dir, "y_test.npy"))
    time = np.load(join(args.output_dir, "test_time.npy"))
    
    data = pd.read_csv(join(args.data_dir, "train_val_data.csv"), index_col=0, parse_dates=True)
    data_std = (data.std()).values
    data_mean = (data.mean()).values
    
    x = torch.tensor(x).float()
    # y = torch.tensor(test_y).float()
    
    best_model = BayesianLSTM(args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.dropout_rate)
    best_model.load_state_dict(torch.load(f'{args.output_dir}/best_bayesian_lstm.pth'))
    
 
    
    mean_pred, uncertainty = predict_bayesian_dropout(best_model, data_mean, data_std, x, num_real=100)
    uncertainty = uncertainty.numpy()
   
    test_pred = np.reshape(mean_pred, (mean_pred.shape[0]*mean_pred.shape[1], mean_pred.shape[2]))
    test_uncert= np.reshape(uncertainty, (uncertainty.shape[0]*uncertainty.shape[1], uncertainty.shape[2]))
    test_time = time.flatten()
    
    test_y = np.reshape(y, (y.shape[0]*y.shape[1], y.shape[2]))
    test_y = (test_y*data_std)+data_mean
    
    test_pred = pd.DataFrame(test_pred)
    test_pred.index = test_time
    
    test_y = pd.DataFrame(test_y)
    test_y.index = test_time
    
    test_uncert = pd.DataFrame(test_uncert) 
    test_uncert.index = test_time
    
    test_uncert.to_csv(join(args.output_dir, "test_uncert.csv"))
    test_y.to_csv(join(args.output_dir, "test_realy.csv"))
    test_pred.to_csv(join(args.output_dir, "test_predy.csv"))
    
    print("Best model loaded, test dataset prediction and uncertainty are saved.")
    
    
if __name__ == "__main__":
    
    args = get_shared_arg_parser()
    get_uncertainty_prediction(args)
    