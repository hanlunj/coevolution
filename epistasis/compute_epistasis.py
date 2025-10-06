import itertools
import pandas as pd
import sys
import torch
import numpy as np
from tqdm import tqdm
from itertools import combinations
from collections import OrderedDict
import glob
import term_extraction

fitness_fname = 'data/fitness_all.pt'
fitness_fname_pre = fitness_fname.split('/')[-1].replace('.pt','')
sequences_fname = 'data/all_sequences.npy'
seq_to_fitness_dict_fname = f'seq_to_fitness_dict_selection_{fitness_fname_pre}.npy'
term_extractor_fname = f'term_extractor_{fitness_fname_pre}.pickle'
alphabet = 'FILMV'


def encode(seqs, alphabet='-ACDEFGHIKLMNPQRSTVWY', verbose=True):
    aa_to_i = OrderedDict((aa, i) for i, aa in enumerate( alphabet ))
    if verbose:
        seq_iter = tqdm(seqs)
    else:
        seq_iter = seqs
    #X = torch.tensor([[aa_to_i[x] for x in seq]
    #                  for seq in seq_iter])
    X = [tuple(aa_to_i[x] for x in seq)
                      for seq in seq_iter]    
    return X, aa_to_i

try:
    seq_to_fitness_dict = np.load(seq_to_fitness_dict_fname, allow_pickle=True).item()
    print('Loaded ', seq_to_fitness_dict_fname)
except FileNotFoundError:
    print('Preparing ', seq_to_fitness_dict_fname)
    sequences = np.load(sequences_fname, allow_pickle=True)
    fitness = torch.load(fitness_fname).tolist()
    seqs_encoded, aa_to_i = encode(sequences, alphabet=alphabet)
    seq_to_fitness_dict = {}
    for s, f in tqdm(zip(seqs_encoded, fitness), total=len(seqs_encoded)):
        seq_to_fitness_dict[s] = f
    np.save(seq_to_fitness_dict_fname, seq_to_fitness_dict)

# Define sequence-set properties
num_sites            = len(list(seq_to_fitness_dict.keys())[0])
global_alphabet_size = len(alphabet)
cardinalities        = [global_alphabet_size]*num_sites

try:
    term_extractor = term_extraction.TermExtractor.from_file(term_extractor_fname)
    print('Loaded ', term_extractor_fname)
except FileNotFoundError:
    print('Preparing ', term_extractor_fname)
    term_extractor = term_extraction.TermExtractor(cardinalities=cardinalities, global_max_term_order=None)
    term_extractor.determine_marginals(state_fitness_dict=seq_to_fitness_dict, marginals_max_term_order=None)
    term_extractor.determine_factors(factors_max_term_order=None)
    # Store the term extractor
    term_extractor.to_file(term_extractor_fname)



