#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:04:13 2026

@author: xl3138
"""

import pandas as pd 
from os.path import join 
from util import get_shared_arg_parser
import argparse
import numpy as np
import pickle
from permetrics.regression import RegressionMetric

def get_metrics(args):
    
    true = pd.read_csv(join(args.output_dir, "uncert_realy.csv"), index_col=0, parse_dates=True)
    pred = pd.read_csv(join(args.output_dir, "uncert_predy.csv"), index_col=0, parse_dates=True)
    
    ####### Evaluate each well seperately #########
    list_metrics = ["RMSE", "MAE", "NSE", "KGE"]
    evaluator = RegressionMetric(true.to_numpy(), pred.to_numpy())
    results = evaluator.get_metrics_by_list_names(list_metrics)
    
    ######## Save results into a file ########
    ######## Beware!!!! This is a pickle file, can only be open with the same environment
    with open(join(args.output_dir, "metric.json"), 'wb') as file:
        pickle.dump(results, file)
        
    f = open(join(args.output_dir, "metric.csv"), 'w')
    f.write ("RMSE (l/s),MAE (l/s),NSE,KGE\n" )  
    
    ###### metric mean of all wells
    for metric in list_metrics:
        vars()[metric] = results[metric].mean()
        if metric != "KGE":
            f.write(str(vars()[metric])+",")
        else:
            f.write(str(vars()[metric]))
            f.write("\n") 
            f.close() 
        
    return results

if __name__ == "__main__":
    
    parser = get_shared_arg_parser()
    args = parser.parse_args()
    results = get_metrics(args)