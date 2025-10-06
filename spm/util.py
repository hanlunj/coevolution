import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from collections import OrderedDict
import scipy.stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import selection

def encode(seqs, alphabet='-ACDEFGHIKLMNPQRSTVWY', verbose=True):
    aa_to_i = OrderedDict((aa, i) for i, aa in enumerate( alphabet ))
    if verbose:
        seq_iter = tqdm(seqs)
    else:
        seq_iter = seqs
    X = torch.tensor([[aa_to_i[x] for x in seq] 
                      for seq in seq_iter])
    return X, aa_to_i


def encode_onehot(msa_sequences, alphabet='FILMV', verbose=True):
    seqs_enc, aa_to_i = encode(msa_sequences, alphabet=alphabet, verbose=verbose)
    seqs_onehot = torch.nn.functional.one_hot(seqs_enc, num_classes=len(alphabet)).to(torch.float)
    return seqs_onehot


def encode_round_onehot(round_list, rounds=[]):
    if len(rounds)==0:
        rounds = sorted(list(set(round_list)))
    round_onehot = torch.nn.functional.one_hot(torch.tensor(round_list), num_classes=len(rounds)).to(torch.float)
    return round_onehot


def decode_onehot(seq_onehot, L, alphabet='FILMV', verbose=False):
    '''
    L: length of protein
    '''
    A = len(alphabet)
    seq_enc = torch.argmax(seq_onehot.reshape(seq_onehot.shape[0],L,A), dim=2)
    seq = []
    if verbose:
        seq_iter = tqdm(range(seq_enc.shape[0]))
    else:
        seq_iter = range(seq_enc.shape[0])
    for idx in seq_iter:
        seq.append(''.join([alphabet[x] for x in seq_enc[idx]]))
    return seq 


def train_nn_regressor(model, optim, criterion, train_dl, val_dl, device='cpu', 
             lr=1e-2, n_epoch=100, batch_size=10000, max_no_improve=5,  
             verbose=True, use_selection=False, total_counts=None, constrained_energy=True, fit_init_round=False):
    
    
    #epoch_iter = tqdm(range(n_epoch), total=n_epoch, position=0)
    train_losses = []
    val_losses = []

    best_train_loss = float('inf')
    best_val_loss = float('inf')

    if verbose:
        #print('epoch', 'train_loss', 'val_loss', 'best_train_loss')
        print('epoch', 'train_loss', 'val_loss', 'best_val_loss')

    no_improve = 0
    #for epoch in epoch_iter:
    for epoch in range(n_epoch):

        model.train()
        epoch_l = 0
        train_total = 0

        for batch in train_dl:
            batch_data = batch[0].to(device)
            batch_labels = batch[1].to(device)

            optim.zero_grad()

            if use_selection:
                #p_init_s = torch.ones(batch_size, 1)/batch_size  # for drop_last=True
                p_init_s = torch.ones(batch[0].shape[0], 1)/batch[0].shape[0] # for drop_last=False
                loss = selection.get_batch_nll(model, batch_data, batch_labels, total_counts, p_init_s, device=device, 
                                               constrained_energy=constrained_energy, fit_init_round=fit_init_round)
            else:
                pred = model(batch_data).squeeze(1)
                loss = criterion(pred, batch_labels)

            loss.backward()
            epoch_l += loss.item()
            train_total += batch_labels.size(0)
            optim.step()

        train_losses.append(epoch_l)

        '''
        # use train loss for early stop
        if epoch_l < best_train_loss:
            no_improve = 0
            best_train_loss = epoch_l 
        else:
            no_improve += 1
        '''

        model.eval()  
        val_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for batch in val_dl:
                batch_data = batch[0].to(device)
                batch_labels = batch[1].to(device)

                if use_selection:
                    #p_init_s = torch.ones(batch_size, 1)/batch_size  # for drop_last=True
                    p_init_s = torch.ones(batch[0].shape[0], 1)/batch[0].shape[0] # for drop_last=False
                    loss = selection.get_batch_nll(model, batch_data, batch_labels, total_counts, p_init_s, device=device,
                                                   constrained_energy=constrained_energy, fit_init_round=fit_init_round)
                else:
                    pred = model(batch_data).squeeze(1)
                    loss = criterion(pred, batch_labels)

                val_loss += loss.item()
                val_total += batch_labels.size(0)
                

        # use validation loss for early stop
        if val_loss < best_val_loss:
            no_improve = 0
            best_val_loss = val_loss
        else:
            no_improve += 1
        
        val_losses.append(val_loss)

        if verbose:
            #print(epoch, train_losses[-1]/len(train_dl), val_losses[-1]/len(val_dl), best_train_loss/len(train_dl))
            if use_selection:
                print(epoch, train_losses[-1]/len(train_dl)/batch_size, val_losses[-1]/len(val_dl)/batch_size, best_val_loss/len(val_dl)/batch_size)
            else:
                print(epoch, train_losses[-1]/len(train_dl), val_losses[-1]/len(val_dl), best_val_loss/len(val_dl))

        if no_improve >= max_no_improve:
            if verbose:
                #print(f'Stopped as no train loss improvement for {max_no_improve} epoches!')
                #print(f'Best train loss: {best_train_loss}/len(train_dl)')
                print(f'Stopped as no val loss improvement for {max_no_improve} epoches!')
                if use_selection:
                    print(f'Best validation loss: {best_val_loss/len(val_dl)/batch_size}')
                else:
                    print(f'Best validation loss: {best_val_loss/len(val_dl)}')
            break
            
            
    return model


def get_spearman_pearson_mse_r2(model, test_dls, device='cpu', batch_size=10000):

    '''
       test_dls is a dictionary {round#:test_dl, ...}, round#=-1 for all test data
    '''

    mse_loss = nn.MSELoss()
    model.eval()  

    spearmans, pearsons, mses = {}, {}, {}
    r2s = {}
    all_predictions_out, all_labels_out = {}, {}
    
    for r in sorted(list(test_dls.keys())):

        test_preds, test_labels = [], []
        with torch.no_grad():
 
            for batch in test_dls[r]:
                batch_data = batch[0].to(device)
                batch_labels = batch[1].to(device)
 
                pred = model(batch_data).squeeze(1)
                test_preds.append(pred)
                test_labels.append(batch[1])
                    
        all_predictions = torch.cat(test_preds).cpu().detach()
        all_labels = torch.cat(test_labels).cpu().detach()
        
        spearman = spearmanr(all_labels, all_predictions).correlation
        pearson = pearsonr(all_labels, all_predictions).correlation
        mse = mse_loss(all_labels, all_predictions).item()
        r2 = r2_score(all_labels, all_predictions)
 
        spearmans[r] = spearman
        pearsons[r] = pearson
        mses[r] = mse
        r2s[r] = r2
        all_predictions_out[r] = all_predictions
        all_labels_out[r] = all_labels

    return spearmans, pearsons, mses, r2s, all_predictions_out, all_labels_out


def get_spearman_pearson_mse_r2_selection(model, test_dl, device='cpu', batch_size=10000, total_counts=None, positive_counts_only=True, min_counts=-1, constrained_energy=True, fit_init_round=False):

    mse_loss = nn.MSELoss()
    test_preds, test_labels = [], []

    model.eval()  

    with torch.no_grad():

        for batch in test_dl:
            batch_data = batch[0].to(device)
            batch_labels = batch[1].to(device)

            #p_init_s = torch.ones(batch_size, 1)/batch_size  # for drop_last=True
            p_init_s = torch.ones(batch[0].shape[0], 1)/batch[0].shape[0] # for drop_last=False

            pred = selection.get_batch_nll(model, batch_data, batch_labels, total_counts, p_init_s, predict_counts=True, device=device, constrained_energy=constrained_energy, fit_init_round=fit_init_round)
            test_preds.append(pred)
            test_labels.append(batch_labels.T.cpu().detach())
                
    #all_predictions = torch.hstack(test_preds).cpu().detach()
    #all_labels = torch.hstack(test_labels).cpu().detach()
    all_predictions = torch.cat(test_preds, dim=1).cpu().detach()
    all_labels = torch.cat(test_labels, dim=1).cpu().detach()
    
    all_predictions_out, all_labels_out = {}, {}
    pearsons, spearmans, mses = {}, {}, {}
    r2s = {}
    for r in range(all_predictions.shape[0]):
        # only consider predictions for (sequence, round) with count > 0
        if min_counts > -1:
            all_labels_r = all_labels[r][torch.where(all_labels[r]>min_counts)]
            all_predictions_r = all_predictions[r][torch.where(all_labels[r]>min_counts)]
        elif positive_counts_only:
            all_labels_r = all_labels[r][torch.where(all_labels[r]>0)]
            all_predictions_r = all_predictions[r][torch.where(all_labels[r]>0)]
        else:
            all_labels_r = all_labels[r]
            all_predictions_r = all_predictions[r]
        all_labels_out[r] = all_labels_r
        all_predictions_out[r] = all_predictions_r
        if all_labels_r.shape[0]<2 or all_predictions_r.shape[0]<2:
            print(f'Warning: no data points left for round{r} after filtering by --min_counts={min_counts}!')
            pearson = np.nan
            spearman = np.nan
            mse = np.nan
            r2 = np.nan
        else:
            pearson = pearsonr(all_labels_r, all_predictions_r).correlation
            spearman = spearmanr(all_labels_r, all_predictions_r).correlation
            mse = mse_loss(all_labels_r, all_predictions_r).item()
            r2 = r2_score(all_labels_r, all_predictions_r)
        pearsons[r] = pearson
        spearmans[r] = spearman
        mses[r] = mse
        r2s[r] = r2
    # -1 for all rounds combined
    all_labels_all = all_labels.flatten()
    all_predictions_all = all_predictions.flatten()
    if min_counts > -1:
        all_labels_all_select = all_labels_all[torch.where(all_labels_all>min_counts)]
        all_predictions_all_select = all_predictions_all[torch.where(all_labels_all>min_counts)]
    elif positive_counts_only:
        all_labels_all_select = all_labels_all[torch.where(all_labels_all>0)]
        all_predictions_all_select = all_predictions_all[torch.where(all_labels_all>0)]
    else:
        all_labels_all_select = all_labels_all
        all_predictions_all_select = all_predictions_all
    all_labels_out[-1] = all_labels_all_select
    all_predictions_out[-1] = all_predictions_all_select
    pearsons[-1] = pearsonr(all_labels_all_select, all_predictions_all_select).correlation
    spearmans[-1] = spearmanr(all_labels_all_select, all_predictions_all_select).correlation
    mses[-1] = mse_loss(all_labels_all_select, all_predictions_all_select).item()
    r2s[-1] = r2_score(all_labels_all_select, all_predictions_all_select)
    return spearmans, pearsons, mses, r2s, all_predictions_out, all_labels_out


def predict_fitness(model, dl, device='cpu', batch_size=10000, num_rounds=7, 
                    constrained_energy=True, fit_init_round=False, verbose=False):

    test_preds = []
    model.eval()  

    with torch.no_grad():
        if verbose:
            for batch in tqdm(dl):
                batch_data = batch[0].to(device)
                pred_fitness = selection.get_batch_fitness(model, batch_data, num_rounds, constrained_energy=constrained_energy, fit_init_round=fit_init_round, device=device)
                test_preds.append(pred_fitness)
        else:
            for batch in dl:
                batch_data = batch[0].to(device)
                pred_fitness = selection.get_batch_fitness(model, batch_data, num_rounds, constrained_energy=constrained_energy, fit_init_round=fit_init_round, device=device)
                test_preds.append(pred_fitness)
                
    all_predictions = torch.cat(test_preds, dim=0).cpu().detach()
    all_predictions_round_mins, _ = torch.min(all_predictions, dim=0, keepdim=True)
    fitness_round = all_predictions - all_predictions_round_mins
    fitness_all = torch.sum(fitness_round, dim=1)
    return fitness_round, fitness_all


def predict_counts(model, dl, device='cpu', batch_size=10000): 

    test_preds = []
    model.eval()  

    with torch.no_grad():
        for batch in tqdm(dl):
            batch_data = batch[0].to(device)
            pred = model(batch_data).squeeze(1)
            test_preds.append(pred)

    all_predictions = torch.cat(test_preds).cpu().detach()
    return all_predictions


def predict_counts_selection(model, test_dl, device='cpu', batch_size=10000, total_counts=None, constrained_energy=True, fit_init_round=False, n_samples=1):

    test_preds = []

    model.eval()  

    with torch.no_grad():

        for batch in test_dl:
            batch_data = batch[0].to(device)
            batch_labels = batch[1].to(device)

            #p_init_s = torch.ones(batch_size, 1)/batch_size  # for drop_last=True
            p_init_s = torch.ones(batch[0].shape[0], 1)/batch[0].shape[0] # for drop_last=False

            pred = selection.predict_counts(model, batch_data, batch_labels, total_counts, p_init_s, device=device, constrained_energy=constrained_energy, fit_init_round=fit_init_round, n_samples=n_samples)
            test_preds.append(pred)
                
    all_predictions = np.concatenate(test_preds, axis=1)
    
    return all_predictions


