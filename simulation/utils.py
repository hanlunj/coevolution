import glob, os, sys
from pathlib import Path
import numpy as np

def save_traj(traj, start_ind, sdir):
    # can just hstack them and store them in a bin based on their starting index
    res = np.hstack(load_trajs(start_ind, sdir) + [traj])
    np.save(sdir+str(start_ind)+'.npy', res)
    del res

def save_trajs(traj_list, start_ind, sdir):
    # can just hstack them and store them in a bin based on their starting index
    res = np.hstack(load_trajs(start_ind, sdir) + traj_list)
    np.save(sdir+str(start_ind)+'.npy', res)
    del res

def load_trajs(start_ind, sdir):
    try:
        res = np.load(sdir+str(start_ind)+'.npy')
        return split_trajs(res, start_ind)
    except Exception as e:
        # print(e)
        return []

def split_trajs(trajs, start_ind):
    inds = np.arange(trajs.shape[0])[trajs == start_ind]
    return np.split(trajs, inds[1:]) # b/c inds[0] will be 0


def get_traj_lengths(trajs, start_ind): # list of inds
    lengths = []
    traj_list = trajs
    if not isinstance(trajs, list):
        traj_list = split_trajs(trajs, start_ind)
    for t in traj_list:
        lengths.append(len(t))
    return lengths
        
def mut_onehot(s, i, oh_len):
    '''
    Given a onehot sequence array and an index, 
    (copy) mutation at index i, return array too.
    '''
    chunk = i // oh_len
    res = s.flatten()
    res[chunk*oh_len:(chunk+1)*oh_len] = 0
    res[i] = 1
    return res

def mut_seq(s, i, alphabet, j):
    '''
    Given a sequence str and a position index, 
    (copy) mutation to AA j. No cp b/c str immutable.
    '''
    return s[:i] + alphabet[j] + s[i+1:]

def edit_distance_onehot(s, t):
    '''Both onehot rep.'''
    return np.sum(np.abs(s - t)) / 2
