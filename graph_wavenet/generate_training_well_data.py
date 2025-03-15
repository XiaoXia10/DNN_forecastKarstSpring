import argparse
import numpy as np
import os
import pandas as pd
from os.path import join
from scipy import signal

def get_leap_years(year_list):
    leaplist=[]
    for year in year_list:
        if ((year%4==0 and year%100!=0) or (year%400==0)):
            leaplist.append(year)
            
    return leaplist

def _standardize_df(df):
    std = df.std()
    mean = df.mean()
    df_standardized = (df-mean.values)/std.values
    
    return df_standardized

# def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, scaler=None):
def _generate_graph_seq2seq_io_data(df, seq_length_x, seq_length_y, shift, scaler=None):    
    """
    Generate samples from
    :param df:
    :param x_offsets: the x offsets are the x indices of the sequence.
    e.g. if seq_length_x is 3 then x_offsets = [-2,-1,0]
    
    :param y_offsets: the y offsets are the y indices of the sequence.
    e.g. if seq_length_y is 3 then x_offsets = [1, 2, 3]
   
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df, axis=-1)
    feature_list = [data]
    
    df_time = df.index

       
    years = df.index.values.astype('datetime64[Y]').astype(int) + 1970
    days = df.index.values.astype("datetime64[D]") - df.index.values.astype("datetime64[Y]")
    days = days.astype("int32")+1

    leap_years = get_leap_years(np.unique(years))

    scaled_days = [] 
    
    for day, year in zip(days, years):
        if year in leap_years:
            scaled_days.append(day/366)
        else:
            scaled_days.append(day/365)
            
    time_in_year = np.tile(scaled_days, [1, num_nodes, 1]).transpose((2, 1, 0))
    feature_list.append(time_in_year)
    
    data = np.concatenate(feature_list, axis=-1)
    x, y, time_list = [], [], []
    #shift = seq_length_x
    max_t = num_samples - (seq_length_y+shift)
   
    for t in range(0, max_t, shift):  # t is the index of the last observation.
        
        total_window_len = data[t:seq_length_x+seq_length_y+t]
        x.append(total_window_len[:seq_length_x, ...])
        y.append(total_window_len[seq_length_x:, ...]) 
        
        time_window_len = df_time[t:seq_length_x+seq_length_y+t]
        time_list.append(time_window_len[seq_length_x:]) # Only collecting the time for y dataset
        
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    time = np.stack(time_list, axis=0)
    
    
    return x, y, time

def generate_train_val_test(args):
    """

    Parameters
    ----------
    args : From parser
       
    Returns
    -------
    x_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    x_val : TYPE
        DESCRIPTION.
    y_val : TYPE
        DESCRIPTION.
    x_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    """
    
    seq_length_x, seq_length_y, shift = args.seq_length_x, args.seq_length_y, args.shift
    df = pd.read_csv(args.df, index_col=0, parse_dates=True)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    
    
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # print('x offset ' + str(x_offsets))
    
    y_offsets = np.sort(np.arange(1, (seq_length_y + 1), 1))
    # print('y offset ' + str(y_offsets))
  
    print(x_offsets)
    print(y_offsets)
    
    df_std = _standardize_df(df)
    x, y, time = _generate_graph_seq2seq_io_data(
        df_std,
        seq_length_x,
        seq_length_y,
        shift
    )

    # print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * args.test_split)
    num_train = round(num_samples * args.train_split)
    num_val = num_samples - num_test - num_train
    x_train, y_train, time_train = x[:num_train], y[:num_train], time[:num_train]
   
    x_val, y_val,time_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val], time[num_train: num_train + num_val]

    x_test, y_test, time_test = x[-num_test:], y[-num_test:], time[-num_test:]
    print(x_train.shape)
    print(y_train.shape)

    # for cat in ["train", "val", "test"]:
    #     _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    #     print(cat, "x: ", _x.shape, "y:", _y.shape)
    #     np.savez_compressed(
    #         os.path.join(args.output_dir, f"{cat}.npz"),
    #         x=_x,
    #         y=_y,
    #         x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
    #         y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    #     )
    
    # Write data into csv files for reconstruction of data after model training 
    # This is for plotting and debugging
    # np.save(join( args.output_dir, "x_train.npy"), x_train)
    # np.save(join( args.output_dir, "y_train.npy"), y_train)
    # np.save(join( args.output_dir, "x_val.npy"), x_val)
    # np.save(join( args.output_dir, "y_val.npy"), y_val)
    # np.save(join( args.output_dir, "x_test.npy"), x_test)
    # np.save(join( args.output_dir, "y_test.npy"), y_test)
    # np.save(join(args.output_dir, "test_time.npy"), time_test)
    # np.save(join(args.output_dir, "val_time.npy"), time_val)
    # np.save(join(args.output_dir, "train_time.npy"), time_train)
    
    print("The files are saved in output directory")
    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    # parser = util.get_shared_arg_parser()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", type=str, default="data/data_12H/data_12", help="Output directory.")
    parser.add_argument("--df", type=str, default="data/data_D/betteraz_D_sims.csv", help="input data.",)
    parser.add_argument("--seq_length_x", type=int, default=3, help="X Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=3, help="Y Sequence Length.",)
    parser.add_argument("--shift", type=int, default=3, help="Default is seq_length_x", ) # this is a sequence window shift
    parser.add_argument("--train_split", type=float, default=0.70, help="The percentage split for training datsset.",)
    parser.add_argument("--test_split", type=float, default=0.15, help="The percentage split for testing dataset.",)

    args = parser.parse_args()
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
        
    x_train, y_train, x_val, y_val, x_test, y_test = generate_train_val_test(args)
    # generate_train_val_test(args)
    