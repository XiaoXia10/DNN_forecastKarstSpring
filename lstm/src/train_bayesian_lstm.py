#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 20:19:36 2026

@author: xl3138
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from os.path import join
import torch.optim as optim
import argparse
from bayesian_lstm import BayesianLSTM
from durbango import pickle_save
import time
from utils import data_loader, get_shared_arg_parser
from get_uncertainty import get_uncertainty_prediction
from get_metrics import get_metrics
import matplotlib.pyplot as plt



def main(args):
    
    model = BayesianLSTM(args.input_dim, args.hidden_dim, args.output_dim, args.num_layers, args.dropout_rate)

    # Make the best validation loss an infinite float, it will get rewritten
    best_val_loss = float('inf')
    counter = 0

    train_loader = data_loader(args, loader="train")
    val_loader = data_loader(args, loader="val")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    #### Which loss is the best???? 
    # criterion = nn.MSELoss() # MSE 
    criterion = nn.L1Loss() # MAE
    
    save_best_model = f'{args.output_dir}/best_bayesian_lstm.pth'
    
    train_loss_list = []
    val_loss_list = []
    
    # model.train()
    for epoch in range(args.epochs):
        # Train the model
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss_list.append(train_loss)
        
        # Validate the model
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_outputs = model(x_val)
                val_loss += criterion(val_outputs, y_val).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
       
        # Early stopping with number of patience
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save best model weights
            torch.save(model.state_dict(), save_best_model)
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break
            
    print(f"Best validation loss: {avg_val_loss}")
    plt.plot(train_loss_list, label="Train Loss") 
    plt.plot(val_loss_list, label = "Val Loss")
    plt.legend()
    
if __name__ == "__main__":

    args = get_shared_arg_parser()

    pickle_save(args, f'{args.output_dir}/args.pkl')
    
    t1 = time.time()
    main(args)
    t2 = time.time()
    minutes = (t2 - t1) / 60
    print(f"Total time spent: {minutes:.2f} minutes")
    
    get_uncertainty_prediction(args)
    metrics = get_metrics(args)
    
    