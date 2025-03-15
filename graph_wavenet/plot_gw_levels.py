import argparse
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np


def _load_loader_data(loader):
    input_dir =args.input_dir
    
    yhat = pd.read_csv(join(input_dir, loader+"_predy.csv"))
    realy = pd.read_csv(join(input_dir, loader+"_realy.csv"))
    
    return yhat, realy

def _destandardize_pred(args, loader):
    
    df = pd.read_csv(args.df, parse_dates=True, index_col=0)
    std = df.std()
    mean = df.mean()
    
    yhat, realy = _load_loader_data(loader)
    
    yhat = (yhat*std.values)+mean.values
    realy = (realy*std.values)+mean.values
    
    time = np.load(join(args.time_dir, loader+"_time.npy"), allow_pickle=True)
    time = pd.to_datetime(time.flatten())
           
    yhat.index = time
    realy.index = time
       
    return yhat, realy

def plot_one_loader_segment(args):
    loader = args.loader_type
    names = args.list_names
    
    yhat, realy = _destandardize_pred(args, loader)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,figsize=(20, 20), sharex=True,)
    ax1.plot(realy.index, realy.iloc[:,0], label="Measured-Betteraz", color="r", linewidth=2)
    ax1.plot(yhat.index, yhat.iloc[:,0], label="Predicted-Betteraz", color = "g", linewidth=2)
    ax1.set_title("Test Dataset", fontsize = 20)
    ax1.set_ylabel("Water Level (MASL)", fontsize = 20)
    
    ax2.plot(realy.index, realy.iloc[:,1], label="Measured-CA2", color="r", linewidth=2)
    ax2.plot(yhat.index, yhat.iloc[:,1], label="Predicted-CA2", color = "g", linewidth=2)
    ax2.set_ylabel("Water Level (MASL)", fontsize = 20)
    
    ax3.plot(realy.index, realy.iloc[:,2], label="Measured-SRM2", color="r", linewidth=2)
    ax3.plot(yhat.index, yhat.iloc[:,2], label="Predicted-SRM2", color = "g", linewidth=2)
    ax3.set_ylabel("Water Level (MASL)", fontsize = 20)
    
    ax4.plot(realy.index, realy.iloc[:,3], label="Measured-CP1", color="r", linewidth=2)
    ax4.plot(yhat.index, yhat.iloc[:,3], label="Predicted-CP1", color = "g", linewidth=2)
    ax4.set_xlabel("Date", fontsize = 20)
    ax4.set_ylabel("Water Level (MASL)", fontsize = 20)
    
    ax1.tick_params(labelsize=20)
    ax1.legend(fontsize = 20)
    ax2.tick_params(labelsize=20)
    ax2.legend(fontsize = 20)
    ax3.tick_params(labelsize=20)
    ax3.legend(fontsize = 20)
    ax4.tick_params(labelsize=20)
    ax4.legend(fontsize = 20)
    
    plt.tight_layout()
    
    # for i in range(0, yhat.shape[1]):
    #     plt.figure(figsize=(15,5))
    #     plt.plot(realy.index, realy.iloc[:,i], label="True", color="r", linewidth=2)
    #     plt.plot(yhat.index, yhat.iloc[:,i], label="Pred", color = "g", linewidth=2)
    #     plt.title(names[i]+" - Test dataset", fontsize=18)
    #     plt.xlabel('Date', fontsize=15)
    #     plt.ylabel('Water Level (MASL)', fontsize=15)
    #     plt.legend(fontsize=15)

        
def plot_entire_timerseries(args):
    names = args.list_names
    well_num = args.plot_well_num
    
    loaders = ["train","val","test"]
    df_all_yhat = pd.DataFrame()
    df_all_realy = pd.DataFrame()
    
    plt.figure(figsize=(25,5))
    for loader in loaders:
        yhat, realy = _destandardize_pred(args, loader)
        
        df_all_yhat = pd.concat([df_all_yhat, yhat])
        df_all_realy = pd.concat([df_all_realy, realy])
        
        # plt.plot(realy.index, realy.iloc[:,0], label=loader, linewidth=2)
        plt.plot(realy.index, realy.iloc[:,well_num], color="r", linewidth=2)
        plt.plot(yhat.index, yhat.iloc[:,well_num], label=str(loader)+" pred.", linewidth=2)
    
    plt.title(str(names[well_num]), fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Water Level (MASL)', fontsize=15)
    plt.legend(fontsize=20)
    
    
    # for i in range(0, df_all_yhat.shape[1]):
        
    #     plt.figure(figsize=(15,5))
    #     plt.plot(df_all_realy.index, df_all_realy.iloc[:,i], color = "r", label="True")
    #     plt.plot(df_all_yhat.index, df_all_yhat.iloc[:,i], label="Pred")
    #     plt.title(names[i], fontsize=18)
    #     plt.xlabel('Date', fontsize=15)
    #     plt.ylabel('MASL', fontsize=15)
    #     plt.legend()
        
        
def generate_plot(args):
    
    if args.one_loader == True:
        plot_one_loader_segment(args)
    
    else:
        plot_entire_timerseries(args)

if __name__ == "__main__":
    
    timestep = "12H"
    forecast = 3
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="", help="Input prediction data directory.")
    parser.add_argument("--time_dir", type=str, default=""), help="Directory for time.",)
    parser.add_argument("--df", type=str, default="", help="df water level readings.",)
    
    parser.add_argument("--plt_save", type=str, default= "", help="Path for to save plot.",)
    parser.add_argument("--seq_length_x", type=int, default=forecast, help="X Sequence Length.",)
    # parser.add_argument("--seq_length_y", type=int, default=3, help="Y Sequence Length.",)
    # parser.add_argument("--shift", type=int, default=3, help="Default is seq_length_x", ) # this is a sequence window shift
    # parser.add_argument("--train_split", type=float, default=0.8, help="The percentage split for training datsset.",)
    # parser.add_argument("--test_split", type=float, default=0.10, help="The percentage split for testing dataset.",)
    parser.add_argument("--list_names", type=list, default=["Batteraz Spring","Station CA2","Station SRM2","Station CP1" ], help="List of names from measuring stations",) #Keep double quotes 
    
    parser.add_argument("--one_loader", default=True, type=str, help="If true, will only plot the specified loader type.",)
    parser.add_argument("--plot_well_num", type=int, default=0, help="Plot entier time series, if one loader is false",)
    parser.add_argument("--loader_type", type=str, default="test", help="Type of loaders - train, val, test.",)
    
    args = parser.parse_args()


generate_plot(args)