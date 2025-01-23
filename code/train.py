import os
import nni
import csv
import time
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from model import DDG_RDE_Network
from dataset import SkempiDatasetManager
from trainer import CrossValidation, recursive_to
from utils import set_seed, check_dir, eval_skempi_three_modes, save_code

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epoch):
    for fold in range(args.num_cvfolds):
        model, optimizer, _ = cv_mgr.get(fold)

        model.train()

        batch = recursive_to(next(dataloader.get_train_loader(fold)), device)
        loss, _ = model(batch)
        if loss == 0:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 1:
            print("\033[0;30;46m{} | [train] Epoch {} Fold {} | Loss {:.8f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, fold, loss.item()))
            log_file.write("{} | [train] Epoch {} Fold {} | Loss {:.8f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, fold, loss.item()))
            log_file.flush()


def pretrain(epoch):
    for fold in range(args.num_cvfolds):
        model, optimizer, _ = cv_mgr.get(fold)

        model.train()

        batch = recursive_to(next(dataloader.pretrain_loader), device)
        loss, _ = model(batch, mode='val')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 1:
            print("\033[0;30;43m{} | [train] Epoch {} Fold {} | Loss {:.8f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, fold, loss.item()))
            log_file.write("{} | [train] Epoch {} Fold {} | Loss {:.8f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), epoch, fold, loss.item()))
            log_file.flush()


def validate():
    results = []
    val_loss_list = []

    with torch.no_grad():
        for fold in range(args.num_cvfolds):
            model, _, _ = cv_mgr.get(fold)

            model.eval()

            for i, batch in enumerate(tqdm(dataloader.get_val_loader(fold), desc='validate', dynamic_ncols=True)):
                batch = recursive_to(batch, device)
                loss, output_dict = model(batch, mode='val')
                val_loss_list.append(loss.item())

                for complex, num_muts, ddg_true, ddg_pred in zip(batch['complex'], batch['num_muts'], output_dict['ddG_true'], output_dict['ddG_pred']):
                    results.append({
                        'complex': complex,
                        'num_muts': num_muts,
                        'ddG': ddg_true.item(),
                        'ddG_pred': ddg_pred.item()
                    })
    
    results = pd.DataFrame(results)
    results['method'] = 'RDE'
    df_metrics = eval_skempi_three_modes(results).to_numpy()

    print("\033[0;30;43m {} | [val] A-Pea {:.6f} A-Spe {:.6f} | RMSE {:.6f} MAE {:.6f} AUROC {:.6f} | P-Pea {:.6f} P-Spe {:.6f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), 
                                df_metrics[0][1], df_metrics[0][2], df_metrics[0][3], df_metrics[0][4], df_metrics[0][5], df_metrics[0][6], df_metrics[0][7]))
    log_file.write("{} | [val] A-Pea {:.6f} A-Spe {:.6f} | RMSE {:.6f} MAE {:.6f} AUROC {:.6f} | P-Pea {:.6f} P-Spe {:.6f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), 
                                df_metrics[0][1], df_metrics[0][2], df_metrics[0][3], df_metrics[0][4], df_metrics[0][5], df_metrics[0][6], df_metrics[0][7]))
    log_file.flush()

    for fold in range(args.num_cvfolds):
        _, _, scheduler = cv_mgr.get(fold)
        scheduler.step(np.mean(val_loss_list))
    
    return df_metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cvfolds', type=int, default=3) 
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--node_feat_dim', type=int, default=128)
    parser.add_argument('--pair_feat_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)

    parser.add_argument('--pre_epoch', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=10000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--ab_mode', type=int, default=0)
    parser.add_argument('--input_mode', type=int, default=0)
    parser.add_argument('--loss_weight', type=float, default=0.5)
    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())
    args = argparse.Namespace(**param)

    param_s = json.loads(open("../configs/train_config.json", 'r').read())
    args = argparse.Namespace(**param_s)

    set_seed(args.seed)

    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    save_dir = os.path.join('../results/', timestamp)
    check_dir(os.path.join(save_dir, 'checkpoint'))
    log_file = open(os.path.join(save_dir, "train_log.txt"), 'a+')
    save_code(save_dir)
    with open(os.path.join(save_dir, 'train_config.json'), 'w') as fout:
        json.dump(args.__dict__, fout, indent=2)

    print('Loading datasets...')
    dataloader = SkempiDatasetManager(config=args, num_cvfolds=args.num_cvfolds, num_workers=4)

    print('Building model...')
    cv_mgr = CrossValidation(config=args, num_cvfolds=args.num_cvfolds, model_factory=DDG_RDE_Network).to(device)

    if args.pre_epoch != 0:
        for epoch in range(1, args.pre_epoch+1):
            pretrain(epoch)

            if epoch % args.val_freq == 0:
                metrics = validate()
                valid_metric = metrics[0][1] + metrics[0][2] - metrics[0][3] - metrics[0][4] + metrics[0][5] + metrics[0][6] + metrics[0][7]
                print("\033[0;30;43m{} | cur_metric {}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), valid_metric))
                log_file.write("{} | cur_metric {}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), valid_metric))
                log_file.flush()

    patience = 0
    best_valid_epoch = 0
    best_valid_metric = -1e6 

    for epoch in range(1, args.max_epoch+1):
        train(epoch)

        if epoch % args.val_freq == 0:
            metrics = validate()

            valid_metric = metrics[0][1] + metrics[0][2] - metrics[0][3] - metrics[0][4] + metrics[0][5] + metrics[0][6] + metrics[0][7]
            if valid_metric > best_valid_metric:
                torch.save(cv_mgr.state_dict(args), os.path.join(save_dir, 'checkpoint', f'best.ckpt'))
                best_valid_metric = valid_metric
                best_valid_epoch = epoch
                patience = 0
            else:
                patience += 1

            if patience > 5:
                break

            print("\033[0;30;43m{} | cur_metric {} best_metric {} | best_epoch {}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), valid_metric, best_valid_metric, best_valid_epoch))
            log_file.write("{} | cur_metric {} best_metric {} | best_epoch {}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), valid_metric, best_valid_metric, best_valid_epoch))
            log_file.flush()


    cv_mgr.load_state_dict(torch.load(os.path.join(save_dir, 'checkpoint', f'best.ckpt')))
    metrics = validate()

    nni.report_final_result(best_valid_metric)
    outFile = open('../PerformMetrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')

    results = [timestamp]
    for v, k in param.items():
        results.append(k)
    results.append(args.split_seed)

    for i in range(3):
        if i == 0:
            results.append(str(metrics[0][6] + metrics[0][7]))
            results.append(str(best_valid_metric))
            results.append(str(best_valid_epoch))
        results.append(metrics[i][-1])
        for j in range(1, 8):
            results.append(str(metrics[i][j]))

    writer.writerow(results)
