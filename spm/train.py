import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from tqdm import tqdm

import scipy.stats
from scipy.stats import spearmanr, pearsonr

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import pickle
import glob

import networks
import util
from dataloader import dataloaders
from arguments import get_args


def train(args):

    dl = dataloaders(args.seq, args.stats, 
                     alphabet=args.alphabet, random_seed=args.data_split_random_seed, 
                     num_fold=args.num_fold, val_fold_idx=args.val_fold_idx, test_fold_idx=args.test_fold_idx,
                     data_split_dir=args.data_split_dir, split_type=args.split_type, max_pvalue=args.max_pvalue, 
                     use_rounds=args.use_rounds, batch_size=args.batch_size, device=args.device, 
                     subsample_train=args.subsample_train, train_on_all=args.train_on_all, embedding_file_list=args.embedding_file_list)
    if args.use_selection:
        train_X, train_Y, train_dl, val_X, val_Y, val_dl, test_X, test_Y, test_dl = dl.get_seq_round_count_loaders_for_selection()
        feature_dim = train_X.shape[1]+1
    else:
        train_X, train_Y, train_dl, val_X, val_Y, val_dl, test_X, test_Y, test_dl = dl.get_seq_round_count_loaders(round_encoding=args.round_encoding, positive_counts_only=args.positive_counts_only)
        feature_dim = train_X.shape[1]


    if args.model == 'OneLayerFCRegressionRELU':
        model = networks.OneLayerFCRegressionRELU(feature_dim, 1, args.hiddim, args.dropout)
    elif args.model == 'TwoLayerFCRegressionRELU':
        model = networks.TwoLayerFCRegressionRELU(feature_dim, 1, args.hiddim, args.dropout)
    elif args.model == 'ThreeLayerFCRegressionRELU':
        model = networks.ThreeLayerFCRegressionRELU(feature_dim, 1, args.hiddim, args.dropout)
    elif args.model == 'FourLayerFCRegressionRELU':
        model = networks.FourLayerFCRegressionRELU(feature_dim, 1, args.hiddim, args.dropout)
    model.to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    print('Training ...')
    model = util.train_nn_regressor(model, optim, criterion, train_dl, val_dl, device=args.device, 
                        lr=args.lr, n_epoch=args.max_epoch, batch_size=dl.batch_size_train, 
                        max_no_improve=args.max_no_improve, verbose=args.verbose, use_selection=args.use_selection, total_counts=dl.total_counts.to(args.device),
                        constrained_energy=args.constrained_energy, fit_init_round=args.fit_init_round)

    os.system('mkdir -p models')
    if args.train_on_all:
        torch.save(model.to('cpu').state_dict(), f'models/{args.model}_hiddim{args.hiddim}_dataseed{args.data_split_random_seed}_trainseed{args.training_random_seed}_trainall_lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}.pt')    
    else:
        torch.save(model.to('cpu').state_dict(), f'models/{args.model}_hiddim{args.hiddim}_dataseed{args.data_split_random_seed}_trainseed{args.training_random_seed}_valfold{args.val_fold_idx}_testfold{args.test_fold_idx}_lr{args.lr}_bs{args.batch_size}_dropout{args.dropout}.pt')    

if __name__ == '__main__':
    args = get_args()
    print (args)

    if args.gpu_id != -1:
        print(f'CAUTION: manually setting CUDA device to {args.gpu_id}')
        torch.cuda.set_device(args.gpu_id)

    torch.manual_seed(args.training_random_seed)
    train(args)


