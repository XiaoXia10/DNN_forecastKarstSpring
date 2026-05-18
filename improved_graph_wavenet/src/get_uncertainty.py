#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
# from bayesian_lstm import BayesianLSTM
# from utils import data_loader, get_shared_arg_parser
import util
from model2 import *
from os.path import join
import numpy as np
import pandas as pd
import os
from engine import Trainer

def predict_bayesian_dropout(model, device, test_loader, scaler, realy, seq_length, mean_, std_, n_realizations=100):
    
    mean = torch.tensor(mean_, device=device, dtype=torch.float32).view(1, -1, 1) 
    std  = torch.tensor(std_,  device=device, dtype=torch.float32).view(1, -1, 1)


    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()

    all_realizations = []

    for _ in range(n_realizations):
        outputs = []
        for x, _ in test_loader.get_iterator():
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testx = nn.functional.pad(testx, (1, 0, 0, 0))  
            with torch.no_grad():
                preds = model(testx).transpose(1, 3)
            outputs.append(preds[:, 0, :, :])             

        yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]  
        yhat = scaler.inverse_transform(yhat)
        yhat = (yhat * std) + mean
        all_realizations.append(yhat)

    all_realizations = torch.stack(all_realizations, dim=0)
    yhat_mean = all_realizations.mean(dim=0)
    yhat_std  = all_realizations.std(dim=0)
    yhat_var  = all_realizations.var(dim=0)
    return yhat_mean, yhat_var



def get_uncertainty_prediction(args, loader='test', **model_kwargs):
    
    data = pd.read_csv(join(args.data_dir, "df.csv"), index_col=0, parse_dates=True)
    std = (data.std()).values
    mean = (data.mean()).values
    
    device = util.default_device()
    adjinit, supports = util.make_graph_inputs(args, device)

    # Create model 
    best_model = GWNet.from_args(args, device, supports, adjinit, dropout=0.3, **model_kwargs)
    best_model.to(device)

    # Load best trained model
    best_model.load_state_dict(torch.load(join(args.output_dir, 'best_model.pth')))

    data = util.load_dataset(args.output_dir, 
                             args.batch_size, 
                             args.batch_size, 
                             args.batch_size, 
                             n_obs=args.n_obs, 
                             fill_zeroes=args.fill_zeroes
                             )
     
    scaler = data['scaler']
    realy = torch.Tensor(data[f'y_{loader}']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    yhat_mean, yhat_std = predict_bayesian_dropout(best_model, device,  data[f'{loader}_loader'], scaler, realy, args.seq_length, mean, std)
    df_real, df_pred, df_uncert = util.make_pred_df_wells2(realy, yhat_mean, yhat_std , scaler, args.seq_length, args.shift) # Compile and save the GWN prediction data into a dataframe

    df_real = (df_real*std)+mean

    time = np.load(join(args.output_dir, "test_time.npy"))
    time_ind = time.flatten()
    
    df_real.index = time_ind
    df_pred.index = time_ind
    df_uncert.index = time_ind
    
if __name__ == "__main__":
    
    parser = util.get_shared_arg_parser()

    args = parser.parse_args()
    
    get_uncertainty_prediction(args)
    