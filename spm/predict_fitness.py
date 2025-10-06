import torch
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import networks
import util
from torch.utils.data import DataLoader, TensorDataset, Dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, help='numpy array of one-hot encoded sequences')
    parser.add_argument('--seq_onehot', action='store_true', help='sequences already one-hot encoded')
    parser.add_argument('--alphabet', type=str, default='FILMV', help='alphabet of sequences')
    parser.add_argument('--num_rounds', type=int, default=7, help='number of rounds of selection (including round0)')
    parser.add_argument('--device', type=str, default='cpu', help='device for training')
    parser.add_argument('--gpu_id', type=int, default=-1, help='overwrite to select which gpu to use (for non-slurm jobs)')
    parser.add_argument('--use_selection', action='store_true', help='use selection model with negative log likelihood loss')
    parser.add_argument('--constrained_energy', action='store_true', help='for selection model, impose monotonically increasing energies over rounds')
    parser.add_argument('--fit_init_round', action='store_true', help='for selection model, allow model to train on the inital round (Round 0) counts')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('--model', type=str, default='ThreeLayerFCRegressionRELU', help='model to use for training')
    parser.add_argument('--model_weights', type=str, default=None, help='.pt file with pretrained model weights for evaluation')
    parser.add_argument('--hiddim', type=int, default=100, help='hidden dimension for the neural network linear layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--save_round_fitness', action='store_true', help='if enabled, output fitness by rounds instead of the summed fitness per sequence')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    return args

def load_model(args, feature_dim):
    if args.model == 'OneLayerFCRegressionRELU':
        model = networks.OneLayerFCRegressionRELU(feature_dim, 1, args.hiddim, args.dropout)
    elif args.model == 'TwoLayerFCRegressionRELU':
        model = networks.TwoLayerFCRegressionRELU(feature_dim, 1, args.hiddim, args.dropout)
    elif args.model == 'ThreeLayerFCRegressionRELU':
        model = networks.ThreeLayerFCRegressionRELU(feature_dim, 1, args.hiddim, args.dropout)
    elif args.model == 'FourLayerFCRegressionRELU':
        model = networks.FourLayerFCRegressionRELU(feature_dim, 1, args.hiddim, args.dropout)
    model.to(args.device)
    model.load_state_dict(torch.load(args.model_weights))
    model.eval()
    return model


def eval_fitness(args):

    # prepare input data
    if not args.seq_onehot:
        seq = np.load(args.seq, allow_pickle=True)
        seq_onehot = util.encode_onehot(seq, alphabet=args.alphabet, verbose=args.verbose)
    else:
        seq_onehot = torch.load(args.seq)
    seq_onehot = torch.flatten(seq_onehot, 1, 2)

    if args.use_selection:
        dl = DataLoader(TensorDataset(seq_onehot), batch_size=args.batch_size, shuffle=False, drop_last=False)
        feature_dim = seq_onehot.shape[1]+1
        model = load_model(args, feature_dim)
        fitness_round, fitness_all = util.predict_fitness(model, dl, device=args.device, batch_size=args.batch_size, 
                                                num_rounds=args.num_rounds, 
                                                constrained_energy=args.constrained_energy, 
                                                fit_init_round=args.fit_init_round, verbose=args.verbose)
    else:
        fitness_round = []
        for r in args.num_rounds:
            featurized_round = torch.ones(seq_onehot.shape[0])*r # hardcoded predicting round6 counts
            featurized = torch.hstack((featurized_round[..., None], seq_onehot))
            dl = DataLoader(TensorDataset(featurized), batch_size=args.batch_size, shuffle=False, drop_last=False)
            feature_dim = featurized.shape[1]
            model = load_model(args, feature_dim)
            fitness_round.append(util.predict_counts(model, dl, device=args.device, batch_size=args.batch_size))
        fitness_round = torch.vstack(fitness_round)
        # fitness_all in original model is defined as the Round6 counts
        fitness_all = fitness_round[-1]

    if args.save_round_fitness:
        torch.save(fitness_round, 'fitness_round.pt')
    torch.save(fitness_all, 'fitness_all.pt')



if __name__ == '__main__':
    args = get_args()
    print (args)

    if args.gpu_id != -1:
        print(f'CAUTION: manually setting CUDA device to {args.gpu_id}')
        torch.cuda.set_device(args.gpu_id)

    eval_fitness(args)
