import torch
import numpy as np
import pandas as pd
import time
import util
from engine import Trainer
import os
from durbango import pickle_save
from fastprogress import progress_bar

from model2 import GWNet
from util import calc_tstep_metrics
from exp_results import summary, loss_curve, plot_loss_curve
from get_uncertainty import get_uncertainty_prediction
from get_metrics import get_metrics
#from compile_pred_data import get_loader_pred 

def main(args, **model_kwargs):
    
    device = util.default_device()
    
    data = util.load_dataset(args.output_dir, args.batch_size, args.batch_size, args.batch_size, n_obs=args.n_obs, fill_zeroes=args.fill_zeroes)
    scaler = data['scaler']
    aptinit, supports = util.make_graph_inputs(args, device)

    model = GWNet.from_args(args, device, supports, aptinit, **model_kwargs)
    if args.checkpoint:
        model.load_checkpoint(torch.load(args.checkpoint))
    
    model.to(device)
    engine = Trainer.from_args(model, scaler, args)
    metrics = []
    best_model_save_path = os.path.join(args.output_dir, 'best_model.pth')
    
    # keep this to be a high value, will get overwritten, it will break if this value is too small
    lowest_loss_yet = 1000
    mb = progress_bar(list(range(1, args.epochs + 1)))

    # epochs_since_best_loss = 0
    for _ in mb:
        train_loss, train_mape, train_rmse = [], [], []
        data['train_loader'].shuffle()
        for iter, (x, y) in enumerate(data['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            yspeed = trainy[:, 0, :, :]
            if yspeed.max() == 0: continue
        
            # Use True if you want to use extreme function and false to use MAE
            loss, mape, rmse = engine.train(trainx, yspeed, extreme_loss=args.extreme_loss)
            train_loss.append(loss)
            train_mape.append(mape)
            train_rmse.append(rmse)
            if args.n_iters is not None and iter >= args.n_iters:
                break
        engine.scheduler.step()
        _, valid_loss, valid_mape, valid_rmse = eval_(args, data['val_loader'], device, engine)
        m = dict(train_loss=np.mean(train_loss), train_mape=np.mean(train_mape),
                 train_rmse=np.mean(train_rmse), valid_loss=np.mean(valid_loss),
                 valid_mape=np.mean(valid_mape), valid_rmse=np.mean(valid_rmse))

        m = pd.Series(m)
        metrics.append(m)
        if m.valid_loss < lowest_loss_yet:
            torch.save(engine.model.state_dict(), best_model_save_path)
            lowest_loss_yet = m.valid_loss
            epochs_since_best_mae = 0
        else:
            epochs_since_best_mae += 1
        met_df = pd.DataFrame(metrics)
        mb.comment = f'best val_loss: {met_df.valid_loss.min(): .3f}, current val_loss: {m.valid_loss:.3f}, current train loss: {m.train_loss: .3f}'
        met_df.round(6).to_csv(f'{args.output_dir}/training_metrics.csv')
        if epochs_since_best_mae >= args.es_patience: break
    
    ###Metrics on test data
    engine.model.load_state_dict(torch.load(best_model_save_path))
    realy = torch.Tensor(data['y_test']).transpose(1, 3)[:, 0, :, :].to(device)
    test_met_df, yhat = calc_tstep_metrics(engine.model, device, data['test_loader'], scaler, realy, args.seq_length)
    test_met_df.round(6).to_csv(os.path.join(args.output_dir, 'test_metrics.csv'))
    print(summary(args.output_dir))

    if args.plot_history == True:
        
        tr_val = loss_curve(args.output_dir)
        plot_loss_curve(args.output_dir)

    
def eval_(args, ds, device, engine):
    """Run validation."""
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    s1 = time.time()
    for (x, y) in ds.get_iterator():
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        testy = torch.Tensor(y).to(device).transpose(1, 3)
        metrics = engine.eval(args, testx, testy[:, 0, :, :])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
    total_time = time.time() - s1
    return total_time, valid_loss, valid_mape, valid_rmse


if __name__ == "__main__":
    
    parser = util.get_shared_arg_parser()
    args = parser.parse_args()
   
    pickle_save(args, f'{args.output_dir}/args.pkl')
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print(f"Total time spent: {mins:.2f} minutes")
    
    get_uncertainty_prediction(args)
    results = get_metrics(args)
    
  