import pandas as pd
import numpy as np
import argparse
from os.path import join
import pickle
import os
from get_parser import get_shared_arg_parser


def train_val_split(df, train_percent, val_percent):
    
    num_samples = df.shape[0]
    num_val = round(num_samples * val_percent)
    num_train = round(num_samples *train_percent)
    num_test = num_samples - num_val - num_train
    
    time = df.index
    
    train, val, test = df[:num_train], df[num_train: num_train + num_val], df[-num_test:]
   
    train_time, val_time, test_time = time[:num_train], time[num_train: num_train + num_val], time[-num_test:]
    
    time_dic = {"train_time":np.array(train_time), "val_time":np.array(val_time), "test_time":np.array(test_time)} 
    
    return np.array(train), np.array(val), np.array(test), time_dic


def sequence_data_preparation(data, data_time, seq_length_x, seq_length_y, shift):

    num_samples = len(data)
    x, y, time_list = [], [], []
    #shift = seq_length_x
    max_t = num_samples - (seq_length_y+shift)
   
    for t in range(0, max_t, shift):  # t is the index of the last observation.
        
        total_window_len = data[t:seq_length_x+seq_length_y+t]
        x.append(total_window_len[:seq_length_x, :])
        y.append(total_window_len[seq_length_x:, :]) 
        
        time_window_len = data_time[t:seq_length_x+seq_length_y+t]
        time_list.append(time_window_len[seq_length_x:]) # Only collecting the time for y dataset
        
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    time_y = np.stack(time_list, axis=0)
    
    return x, y, time_y


def standardize_df(data_path):
    
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    std = df.std()
    mean = df.mean()
    
    df_std = (df-mean.values)/std.values
    
    return df_std


def main(args):
    
    df_std = standardize_df(args.df)
    
    train, val, test, time_dic = train_val_split(df_std, args.train_percent, args.val_percent)

    time_for_train = time_dic["train_time"]
    time_for_val = time_dic["val_time"]
    time_for_test = time_dic["test_time"]

    x_train, y_train, train_time_y = sequence_data_preparation(train, time_for_train, args.seq_length_x, args.seq_length_y, args.shift)
    x_val, y_val, val_time_y = sequence_data_preparation(val, time_for_val, args.seq_length_x, args.seq_length_y, args.shift)
    x_test, y_test, test_time_y = sequence_data_preparation(test, time_for_test, args.seq_length_x, args.seq_length_y, args.shift)
    
    np.save(join( args.save_dir, "x_train.npy"), x_train)
    np.save(join( args.save_dir, "y_train.npy"), y_train)
    np.save(join( args.save_dir, "x_val.npy"), x_val)
    np.save(join( args.save_dir, "y_val.npy"), y_val)
    np.save(join( args.save_dir, "x_test.npy"), x_test)
    np.save(join( args.save_dir, "y_test.npy"), y_test)
    np.save(join(args.save_dir, "test_time_y.npy"), test_time_y)
    np.save(join(args.save_dir, "val_time_y.npy"), val_time_y)
    np.save(join(args.save_dir, "train_time_y.npy"), train_time_y)
    
 

if __name__ == "__main__":
    
    timestep = "D" 
    seq_length_x = 3
    seq_length_y = 3
    shift = 3
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length_x", type=int, default=seq_length_x, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=seq_length_y, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=shift, help="Default is seq_length_x", ) # this is a sequence window shift
    parser.add_argument("--timestep", type=str, default=timestep, help="Timestep", )
    parser.add_argument("--df", type=str, default="data/betteraz_"+str(timestep)+"_sims.csv", help="Water level readings.")
    parser.add_argument("--save_dir", type=str, default="experiment_GRU_"+str(timestep)+"_"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Output prediction data directory.")
    
    
    parser.add_argument('--train_percent', type=float, default=0.7, help='The percentage of data used for model training')
    parser.add_argument('--val_percent', type=float, default=0.15, help='The percentage of data used for model validation')
    args = parser.parse_args()
    
    # if os.path.exists(args.save_dir):
    #     reply = str(input(f'{args.save_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
    #     if reply[0] != 'y': exit
    # else:
    #     os.makedirs(args.save_dir)
        
    main(args)

    