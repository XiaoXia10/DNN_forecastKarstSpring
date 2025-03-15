import util
from model import *
from os.path import join
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

def _destandardize_pred(args, realy, yhat):
    
    df = pd.read_csv(args.df, parse_dates=True, index_col=0)
    # df = df.drop(columns=["milamont"])
    
    std = df.std()
    mean = df.mean()
      
    yhat_destd = (yhat*std.values)+mean.values
    realy_destd = (realy*std.values)+mean.values
       
    return realy_destd, yhat_destd


def main(args, 
         loader='test', 
         save_pred_path='predy.csv', 
         save_real_path='realy.csv',
         **model_kwargs):
         
    device = torch.device(args.device)
    
    adjinit, supports = util.make_graph_inputs(args, device)
    
    # Create model 
    model = GWNet.from_args(args, device, supports, adjinit, **model_kwargs)
    model.to(device)
    
    # Load best trained model
    model.load_state_dict(torch.load(join(args.save, 'best_model.pth')))
    model.eval()
    print('model loaded successfully')
    
    data = util.load_dataset(args.data, 
                             args.batch_size, 
                             args.batch_size, 
                             args.batch_size, 
                             n_obs=args.n_obs, 
                             fill_zeroes=args.fill_zeroes
                             )
    
    scaler = data['scaler']
    realy = torch.Tensor(data[f'y_{loader}']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    met_df, yhat = util.calc_tstep_metrics(model, device, data[f'{loader}_loader'], scaler, realy, args.seq_length)
    df_pred, df_real = util.make_pred_df_wells(realy, yhat, scaler, args.seq_length, args.shift) # Compile and save the GWN prediction data into a dataframe
   
    met_df.to_csv(join(save_path, "last_test_metrics.csv"))
    df_pred.to_csv(save_pred_path, index=False)
    df_real.to_csv(save_real_path, index=False)

    return df_pred, df_real 

#Keep the double quotes or shit. If you only want one, then keep the loader you want. Default is Test dataset loader
list_loader = ["train", "val", "test"] 

  
if __name__ == "__main__":
    parser = util.get_shared_arg_parser()
    args = parser.parse_args()
    save_path = args.save
    
    filepath = join(save_path, "best_model_metrics.csv")
    f = open(filepath, 'w')
    f.write("loader, RMSE, R2, MAPE, MAE \n" )
    
    for loader in list_loader:

        vars()[loader+'_predy'], vars()[loader+'_realy'] = main(args,
                                            loader=loader, 
                                            save_pred_path= save_path+"/"+loader+"_predy.csv", 
                                            save_real_path= save_path+"/"+loader+"_realy.csv"
                                            )
        
        vars()[loader+'_realy'], vars()[loader+'_predy'] = _destandardize_pred(args, vars()[loader+'_realy'], vars()[loader+'_predy'])
        
        vars()[loader+'_RMSE'] = root_mean_squared_error(vars()[loader+'_realy'], vars()[loader+'_predy'])
        vars()[loader+'_R2'] = r2_score(vars()[loader+'_realy'], vars()[loader+'_predy'])
        vars()[loader+'_MAPE'] = mean_absolute_percentage_error(vars()[loader+'_realy'], vars()[loader+'_predy'])
        vars()[loader+'_MAE'] = mean_absolute_error(vars()[loader+'_realy'], vars()[loader+'_predy'])
        
        f.write(str(loader)+",")
        f.write(str(vars()[loader+'_RMSE'])+",")
        f.write(str(vars()[loader+'_R2'])+",")
        f.write(str(vars()[loader+'_MAPE'])+",")
        f.write(str(vars()[loader+'_MAE'])+",")
        f.write("\n") 
        
        print(loader, "\n")
        print("RMSE:")
        print(vars()[loader+'_RMSE'])
        print("R2:")
        print(vars()[loader+'_R2'])
        print("MAPE:")
        print(vars()[loader+'_MAPE'])
        print("MAE:")
        print(vars()[loader+'_MAE'], "\n")

    f.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    


