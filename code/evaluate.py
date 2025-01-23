import csv
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from Bio.PDB.PDBParser import PDBParser
from torch.utils.data import DataLoader
from Bio.PDB.Polypeptide import index_to_one
from common_utils.protein.parsers import parse_biopython_structure

from model import DDG_RDE_Network
from skempi import MutationDataset, OPTDataset
from trainer import CrossValidation, recursive_to
from utils import set_seed, eval_skempi_three_modes
from dataset import SkempiDatasetManager, PaddingCollate, get_test_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate():
    results = []
    val_loss_list = []
    dataloader = SkempiDatasetManager(config=args, num_cvfolds=args.num_cvfolds, num_workers=0)

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
    
    return df_metrics


def explainer():
    shapely_value_list = np.zeros((len(args.mut_list), 20))
    count_list = np.zeros((len(args.mut_list), 20))
    mtdataset = MutationDataset(args, "../data/test_data")

    with torch.no_grad():
        for iter in range(args.iter_num):
            loader = DataLoader(mtdataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

            for fold in range(args.num_cvfolds):
                model, _, _ = cv_mgr.get(fold)
                model.eval()

                for _, batch in enumerate(tqdm(loader, desc='explain', dynamic_ncols=True)):
                    batch = recursive_to(batch, device)
                    ddg_pred = model(batch, mode='mutation')

                    for i in range(int(batch['aa'].shape[0]/2)):
                        pos, res = batch['idx'][2*i].split("_")
                        shapely_value_list[int(pos), int(res)] += (ddg_pred[2*i] - ddg_pred[2*i+1]).item()
                        count_list[int(pos), int(res)] += 1

            shapely_value = np.zeros((len(args.mut_list), 20))
            for k in range(len(args.mut_list)):
                for l in range(20):
                    shapely_value[k, l] = shapely_value_list[k, l] / (count_list[k, l]+1e-8)

            mtdataset.update_sampling_prob(shapely_value, alpha=args.alpha, tau=args.tau)

            prob_res = mtdataset.sampling_res_prob
            prob_mut = mtdataset.sampling_mut_prob

        return -shapely_value, prob_res, prob_mut


def optimization(prob_res, prob_mut):
    optdataset = OPTDataset(args, prob_res, prob_mut, "../data/test_data")
    loader = DataLoader(optdataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

    results_1 = []
    results_2 = []
    results_3 = []

    with torch.no_grad():
        for fold in range(args.num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()

            count = 0
            for _, batch in enumerate(tqdm(loader, desc='validate', dynamic_ncols=True)):
                batch = recursive_to(batch, device)
                ddg_pred = model(batch, mode='mutation')

                for aa, ddg in zip(batch['aa'], ddg_pred):
                    if fold == 0:
                        results_1.append((optdataset.entries[count]['aa'][args.mut_list], optdataset.entries[count]['aa_mut'][args.mut_list], ddg))
                    if fold == 1:
                        results_2.append((optdataset.entries[count]['aa'][args.mut_list], optdataset.entries[count]['aa_mut'][args.mut_list], ddg))
                    if fold == 2:
                        results_3.append((optdataset.entries[count]['aa'][args.mut_list], optdataset.entries[count]['aa_mut'][args.mut_list], ddg))
                    count += 1
        
    ddg_list = (np.array([item[2].item() for item in results_1]) + np.array([item[2].item() for item in results_2]) + np.array([item[2].item() for item in results_3])) / 3.0
    index = np.argsort(ddg_list)[:10]
    print(ddg_list[index])

    for k, i in enumerate(index):
        print(k, "DDG:", ddg_list[i])
        aa, aa_mut = results_1[i][0], results_1[i][1]
        idx = (aa != aa_mut).nonzero()[:, 0].numpy()
        for j in idx:
            print("{}: {}->{}".format(j, index_to_one(aa[j].item()), index_to_one(aa_mut[j].item())))


def test():
    testloader = get_test_data(args)

    with torch.no_grad():
        for fold in range(args.num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()

            results = []

            for _, batch in enumerate(tqdm(testloader, desc='validate', dynamic_ncols=True)):
                batch = recursive_to(batch, device)
                ddg_pred = model(batch, mode='mutation')

                for i in range(ddg_pred.shape[0]):
                    results.append(ddg_pred[i].item())

            if fold == 0:
                ddg_sum = np.array(results)
            else:
                ddg_sum += np.array(results)

    return ddg_sum / args.num_cvfolds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cplx_name', type=str, default='IFNAR1')
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--iter_num', type=int, default=10)
    parser.add_argument('--task_type', type=str, default='validation')
    args = parser.parse_args()
    param = args.__dict__
    param_model = json.loads(open("../configs/train_config.json", 'r').read())
    param.update(param_model)
    param['mut_list'] = np.arange(51, 58).tolist() + np.arange(96, 108).tolist()
    args = argparse.Namespace(**param)

    set_seed(args.seed)

    print('Building model...')
    cv_mgr = CrossValidation(config=args, num_cvfolds=args.num_cvfolds, model_factory=DDG_RDE_Network).to(device)
    cv_mgr.load_state_dict(torch.load("../trained_models/best.ckpt"))


    if args.task_type == 'validation':
        print("(Validation) | Perfromance on the validation set of Skempi v2.0 ...")
        metrics = validate()
    elif args.task_type == 'prediction':
        print("(Prediction) | The user inputs the mutations they want to predict DDGs ...")
        ddg_mean = test()
        print(ddg_mean)
    elif args.task_type == 'explaination':
        print("(Explaination) | The shaply value map for the sites of interest ...")
        shapely_value, prob_res, prob_mut = explainer()
        print(shapely_value)
    elif args.task_type == 'optimization':
        print("(Optimization) | 10 mutant antibodies with minimal ddg were searched ...")
        shapely_value, prob_res, prob_mut = explainer()
        optimization(prob_res, prob_mut)
    else:
        raise ValueError("task type {} is not supported, it can only be one of the four task types: validation, prediction, explaination, and optimization.".format(args.task_type))