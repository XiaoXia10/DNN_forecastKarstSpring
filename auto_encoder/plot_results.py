import argparse
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error


def _load_loader_data(args, loader):
   
    yhat = pd.read_csv(join(args.save_dir, loader+"_predy.csv"))
    realy = pd.read_csv(join(args.save_dir, loader+"_realy.csv"))
    
    return yhat, realy

def _destandardize_pred(args, loader):
    
    df = pd.read_csv(args.df, parse_dates=True, index_col=0)
    
    std = df.std()
    mean = df.mean()
    
    yhat, realy = _load_loader_data(args,loader)
    
    yhat = (yhat*std.values)+mean.values
    realy = (realy*std.values)+mean.values

    time = np.load(join(args.save_dir, loader+"_time_y.npy"), allow_pickle=True)
    time = pd.to_datetime(time.flatten())
           
    yhat.index = time
    realy.index = time
       
    return yhat, realy

def plot_one_loader_segment(args):
    loader = args.loader

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
    ax2.tick_params(labelsize=20)
    ax2.legend(fontsize = 20)
    ax3.tick_params(labelsize=20)
    ax3.legend(fontsize = 20)
    ax4.tick_params(labelsize=20)
    ax4.legend(fontsize = 20)
    plt.tight_layout()
    # plt.savefig(join(args.plot_save, args.timestep+"_"+str(args.seq_length_x)+str(args.seq_length_y)+str(args.shift)))


def plot_entire_timerseries(args):
    names = args.list_names
    
    loaders = ["train","val","test"]
    df_all_yhat = pd.DataFrame()
    df_all_realy = pd.DataFrame()
    
    plt.figure(figsize=(15,5))
    for loader in loaders:
        yhat, realy = _destandardize_pred(args, loader)
        
        df_all_yhat = pd.concat([df_all_yhat, yhat])
        df_all_realy = pd.concat([df_all_realy, realy])
        
        plt.plot(realy.index, realy.iloc[:,2], label=loader, linewidth=3)

    plt.ylabel('Water Level (MASL)', fontsize=20)
    plt.legend(fontsize=20)
    
def generate_plot(args):
    
    if args.one_loader == True:
        plot_one_loader_segment(args)
    
    else:
        plot_entire_timerseries(args)


if __name__ == "__main__":
    
    timestep = "D" # The time between each step, H=hour, 4H=4hours
    seq_length_x = 12
    seq_length_y = 12
    shift = 12

    parser = argparse.ArgumentParser()
    parser.add_argument("--list_names", type=list, default=["Betteraz Spring","Station CA2","Station SRM2","Station CP1"], help="List of names for measuring stations",) #Keep double quotes or sh*t
    
    parser.add_argument("--one_loader", default=True, type=str, help="If true, will only plot the specified loader type.",)
    parser.add_argument("--loader", type=str, default="test", help="Type of loaders - train, val, test.",)
    
    parser.add_argument("--save_dir", type=str, default="experiment_GRU_"+str(timestep)+"_"+str(seq_length_x)+str(seq_length_y)+str(shift), help="Prediction data directory.")
    parser.add_argument("--df", type=str, default="data/betteraz_"+str(timestep)+"_sims.csv", help="Input data.")
    parser.add_argument("--plot_save", type=str, default= r"G:/My Drive/Plots_compare_manuscript/"+str(timestep)+"_GRU_auto", help="Save plot path")
    
    args = parser.parse_args()
    
generate_plot(args)