import tensorflow as tf
from model_lstm import auto_encoder_lstm, auto_encoder_gru
from get_parser  import get_shared_arg_parser
import pandas as pd
import numpy as np
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

def main(args):
    
    checkpoint_path = join(args.save_dir, args.save_model_name)

    train_x = np.load(join(args.save_dir, "x_train.npy"))
    train_realy = np.load(join(args.save_dir, "y_train.npy"))
    
    if args.lstm==True:
        model = auto_encoder_lstm(train_x, train_realy, args)
    else:
        model = auto_encoder_gru(train_x, train_realy, args)

    model.load_weights(checkpoint_path) 

    val_x = np.load(join(args.save_dir, "x_val.npy"))
    val_realy = np.load(join(args.save_dir, "y_val.npy"))
    
    test_x = np.load(join(args.save_dir, "x_test.npy"))
    test_realy = np.load(join(args.save_dir, "y_test.npy"))

    test_predy = model.predict(test_x)
    train_predy = model.predict(train_x)
    val_predy = model.predict(val_x)
    
    loss_train = model.evaluate(train_x, train_realy, verbose=2)
    print("Restored model, train loss: {:5.2f}".format(loss_train))

    loss_val = model.evaluate(val_x, val_realy, verbose=2)
    print("Restored model, val loss: {:5.2f}".format(loss_val))

    loss_test = model.evaluate(test_x, test_realy, verbose=2)
    print("Restored model, test loss: {:5.2f}".format(loss_test))
    

    test_predy = np.reshape(test_predy, (test_predy.shape[0]*test_predy.shape[1],test_predy.shape[2]))
    test_realy = np.reshape(test_realy, (test_realy.shape[0]*test_realy.shape[1], test_realy.shape[2]))
    
    val_predy = np.reshape(val_predy, (val_predy.shape[0]*val_predy.shape[1], val_predy.shape[2]))
    val_realy = np.reshape(val_realy, (val_realy.shape[0]*val_realy.shape[1], val_realy.shape[2]))
    
    train_predy = np.reshape(train_predy, (train_predy.shape[0]*train_predy.shape[1], train_predy.shape[2]))
    train_realy = np.reshape(train_realy, (train_realy.shape[0]*train_realy.shape[1], train_realy.shape[2]))
    
    test_predy = pd.DataFrame(test_predy)
    test_realy = pd.DataFrame(test_realy)
    
    val_predy = pd.DataFrame(val_predy)
    val_realy = pd.DataFrame(val_realy)
    
    train_predy = pd.DataFrame(train_predy)
    train_realy = pd.DataFrame(train_realy)
    
    train_realy.to_csv(join(args.save_dir, "train_realy.csv"), index =False)
    val_realy.to_csv(join(args.save_dir, "val_realy.csv"), index =False)
    test_realy.to_csv(join(args.save_dir, "test_realy.csv"), index =False)
    
    train_predy.to_csv(join(args.save_dir, "train_predy.csv"), index =False)
    val_predy.to_csv(join(args.save_dir, "val_predy.csv"), index =False)
    test_predy.to_csv(join(args.save_dir, "test_predy.csv"), index =False)

    filepath = join(args.save_dir, "best_model_metrics.csv")
    f = open(filepath, 'w')
    f.write("loader, RMSE, R2, MAPE, MAE \n" )
    
    list_loader = ["train", "val", "test"]
    for loader in list_loader:
        
        # vars()[loader+'_realy'], vars()[loader+'_predy'] = _destandardize_pred(args, vars()[loader+'_realy'], vars()[loader+'_predy'])
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


if __name__ == "__main__":
    parser = get_shared_arg_parser()
    args = parser.parse_args()
    main(args)
    