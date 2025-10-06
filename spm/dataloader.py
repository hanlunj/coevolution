import os
import numpy as np
import pandas as pd
import util
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from tqdm import tqdm


class dataloaders():
    '''
       seq: numpy array of sequences
       stats: [seq_index... , round_index..., ['count','p-value','frequency','log_frequency']]
              frequency computed by normalizing each seq's count by total count in each round 
       num_fold: for KFold split, the fold of test_fold_idx is the holdout set for testing, while the rest are for training
    '''

    def __init__(self, seq_fn, stats_fn, alphabet='FILMV', random_seed=0, num_fold=3, val_fold_idx=0, test_fold_idx=1, 
                 data_split_dir='data', split_type='lastseenround', max_pvalue=1., use_rounds='', batch_size=10000, device='cpu', subsample_train=1.0, train_on_all=False, embedding_file_list=''):

        self.seq = np.load(seq_fn, allow_pickle=True)
        self.stats = np.load(stats_fn, allow_pickle=True)
        assert(self.seq.shape[0]==self.stats.shape[0])
        self.num_seq = self.seq.shape[0]
        self.num_round = self.stats.shape[2]
        if use_rounds == '':
            self.use_rounds = list(range(self.num_round))
        else:
            self.use_rounds = [int(x) for x in use_rounds.split(',')]
        self.total_counts = torch.from_numpy(self.stats[:,0,self.use_rounds].sum(axis=0))
        self.alphabet = alphabet
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.num_fold = num_fold
        self.val_fold_idx = val_fold_idx
        self.test_fold_idx = test_fold_idx
        self.data_split_dir = data_split_dir
        assert(split_type in ['stratified','lastseenround','random'])
        self.split_type = split_type
        self.max_pvalue = max_pvalue
        os.system(f'mkdir -p {self.data_split_dir}')
        self.split_fn = f'{self.data_split_dir}/{self.num_fold}-fold_split_seed_{self.random_seed}_{self.split_type}_maxp{self.max_pvalue}.pkl' 
        self.batch_size = batch_size
        self.batch_size_train = batch_size
        self.batch_size_val = batch_size
        self.batch_size_test = batch_size
        self.device = device

        self.embedding = None
        if embedding_file_list!='':
            self.embedding = []
            with open(embedding_file_list,'r') as ef_fin:
                print('Loading sequence embeddings ...')
                for ef in tqdm(ef_fin.readlines()):
                    self.embedding.append(torch.load(ef.strip()))
            self.embedding = torch.vstack(self.embedding)

        try:
            df_fold = pd.read_pickle(self.split_fn)
            print(f'Loaded dataset splits from {self.split_fn}')

        except FileNotFoundError:
            # group sequences by the last round where they are seen, then kfold split each round
            seqid_by_last_seen_rounds = {}
            if self.max_pvalue < 1:
                seqid_by_last_seen_rounds[self.num_round-1] = np.where((self.stats[:,0,self.num_round-1]>0) & (self.stats[:,1,self.num_round-1]<=self.max_pvalue))[0]  
            else:
                seqid_by_last_seen_rounds[self.num_round-1] = np.where(self.stats[:,0,self.num_round-1]>0)[0]
            claimed_seqids = seqid_by_last_seen_rounds[self.num_round-1]
            for x in range(1, self.num_round):
                r = self.num_round-x-1
                if self.max_pvalue < 1:
                    seqids_r = np.where((self.stats[:,0,r]>0) & (self.stats[:,1,r]<=self.max_pvalue))[0]
                else:
                    seqids_r = np.where(self.stats[:,0,r]>0)[0]
                seqids_r = seqids_r[~np.isin(seqids_r, claimed_seqids)]
                seqid_by_last_seen_rounds[r] = seqids_r
                claimed_seqids = np.append(claimed_seqids, seqids_r)
            all_seqids = np.arange(self.stats.shape[0])
            unclaimed_seqids = all_seqids[~np.isin(all_seqids, claimed_seqids)]
            print(f'Found {unclaimed_seqids.shape[0]} sequences with 0 counts in all rounds! Removed them from dataset.')


            print('Grouping sequencing by the last round where they are seen:')
            for r in range(self.num_round):
                print(r, seqid_by_last_seen_rounds[r].shape[0])


            terms = ['Fold','LastSeenRound','Seq_id']
            data = {x:[] for x in terms}
            if self.split_type=='stratified':
                print('Error: stratified data split to be implemented...')
                sys.exit(1)

            elif self.split_type=='lastseenround':
                kf = KFold(n_splits=self.num_fold, shuffle=True, random_state=self.random_seed)
                kf_split = {}
                for r in range(self.num_round):
                    kf_split[r] = kf.split(seqid_by_last_seen_rounds[r])
                for r in range(self.num_round):
                    for fold, (seqid_xi, seqid_yi) in enumerate(kf_split[r]):
                        seqid_x, seqid_y = seqid_by_last_seen_rounds[r][seqid_xi], seqid_by_last_seen_rounds[r][seqid_yi]
                        data['Seq_id'] += list(seqid_y)
                        data['LastSeenRound'] += [r]*len(seqid_y)
                        data['Fold'] += [fold]*len(seqid_y)

            elif self.split_type=='random':
                kf = KFold(n_splits=self.num_fold, shuffle=True, random_state=self.random_seed)
                terms = ['Fold','LastSeenRound','Seq_id']
                data = {x:[] for x in terms}
                fold = 0
                for seqid_x, seqid_y in kf.split(np.arange(self.num_seq)):
                    lastseenrounds = []
                    for id_y in seqid_y:
                        lastseenround = 0
                        for r in range(self.num_round):
                            if self.stats[id_y,0,self.num_round-1-r]>0:
                                lastseendround = r
                                break
                        lastseenrounds.append(lastseenround)
                    data['Seq_id'] += list(seqid_y)
                    data['LastSeenRound'] += lastseenrounds
                    data['Fold'] += [fold]*len(seqid_y) 
                    fold += 1

            df_fold = pd.DataFrame(data, columns=terms)
            print(f'Writing {self.split_fn}')
            df_fold.to_pickle(f'{self.split_fn}')
            
        # assign val_fold_idx fold to validation set, test_fold_idx fold to test set, and the rest to training set
        if not train_on_all:
            self.df_fold_val = df_fold[df_fold['Fold']==self.val_fold_idx]
            self.df_fold_test = df_fold[df_fold['Fold']==self.test_fold_idx]
            self.df_fold_train = df_fold[~df_fold['Fold'].isin([self.val_fold_idx, self.test_fold_idx])]
        else:
            self.df_fold_val = df_fold
            self.df_fold_test = df_fold
            self.df_fold_train = df_fold
        if subsample_train < 1:
            print('Subsampling training set by: ', subsample_train)
            self.df_fold_train = self.df_fold_train.loc[np.random.choice(self.df_fold_train.index, int(subsample_train*self.df_fold_train.shape[0]), replace=False)] 
        print(f'Training set sequences: {self.df_fold_train.shape[0]}')
        print(f'Validation set sequences: {self.df_fold_val.shape[0]}')
        print(f'Test set sequences: {self.df_fold_test.shape[0]}')

        if self.batch_size == -1:
            print(f'Setting batch_size_train to the size of training set: {self.df_fold_train["Seq_id"].unique().shape[0]}!')
            self.batch_size_train = self.df_fold_train['Seq_id'].unique().shape[0]
            print(f'Setting batch_size_val to the size of training set: {self.df_fold_val["Seq_id"].unique().shape[0]}!')
            self.batch_size_val = self.df_fold_val['Seq_id'].unique().shape[0]
            print(f'Setting batch_size_test to the size of training set: {self.df_fold_test["Seq_id"].unique().shape[0]}!')
            self.batch_size_test = self.df_fold_test['Seq_id'].unique().shape[0]


    def get_seq_last_seen_count_loaders(self):
        '''
           'one count per sequence':
           input features: onehot encoded sequences
           label: counts of the last seen round
        '''
        seq_onehot = util.encode_onehot(self.seq[self.df_fold_train['Seq_id']], alphabet=self.alphabet, verbose=False)
        train_X = torch.flatten(seq_onehot, 1, 2)
        train_Y = torch.from_numpy(self.stats[self.df_fold_train['Seq_id']][np.arange(len(self.df_fold_train['Round'].tolist())),0,np.array(self.df_fold_train['Round'].tolist())]).float()
        print(train_X.shape, train_Y.shape)


    def prepare_dataset_from_df(self, df, round_encoding='direct', positive_counts_only=True, min_counts=-1):
        seqs = []
        rounds = []
        labels = []
        for seq_id in df['Seq_id'].unique():
            for r in self.use_rounds:
                if min_counts > -1:
                    if self.stats[seq_id,0,r]>min_counts and (self.max_pvalue>=1 or self.stats[seq_id,1,r]<=self.max_pvalue):
                        seqs.append(self.seq[seq_id])
                        rounds.append(r)
                        labels.append(self.stats[seq_id,0,r])
                elif positive_counts_only:
                    if self.stats[seq_id,0,r]>0 and (self.max_pvalue>=1 or self.stats[seq_id,1,r]<=self.max_pvalue):
                        seqs.append(self.seq[seq_id])
                        rounds.append(r)
                        labels.append(self.stats[seq_id,0,r])
                else:
                    if self.max_pvalue>=1 or self.stats[seq_id,1,r]<=self.max_pvalue:
                        seqs.append(self.seq[seq_id])
                        rounds.append(r)
                        labels.append(self.stats[seq_id,0,r])
        if self.embedding is not None:
            featurized_seq = self.embedding[df['Seq_id'].unique()]
        else:
            seq_onehot = util.encode_onehot(seqs, alphabet=self.alphabet, verbose=False)
            featurized_seq = torch.flatten(seq_onehot, 1, 2)
        if round_encoding == 'direct':
            featurized_round = torch.tensor(rounds)
            featurized = torch.hstack((featurized_round[..., None], featurized_seq))
        else:
            print(f'Error: round_encoding({round_encoding}) is not supported ...')
            sys.exit(1)
        labels = torch.tensor(labels).float()
        return featurized, labels


    def get_seq_round_count_loaders(self, round_encoding='direct', positive_counts_only=True, min_counts=-1, drop_last=True):
        '''
           for each sequence, prepare all pairs of (round number, counts at that round) 
           input features: round encoding, onehot encoded sequences
           label: counts of input sequence at the input round
        '''
        assert(round_encoding in ['direct'])
        print('Encoding training dataset ...')
        train_X, train_Y = self.prepare_dataset_from_df(self.df_fold_train, round_encoding=round_encoding, positive_counts_only=positive_counts_only, min_counts=min_counts)
        train_dl = DataLoader(TensorDataset(train_X, train_Y), batch_size=self.batch_size_train, shuffle=True, drop_last=drop_last)

        print('Encoding validate dataset ...')
        val_X, val_Y = self.prepare_dataset_from_df(self.df_fold_val, round_encoding=round_encoding, positive_counts_only=positive_counts_only, min_counts=min_counts)
        val_dl = DataLoader(TensorDataset(val_X, val_Y), batch_size=self.batch_size_val, shuffle=True, drop_last=drop_last)

        print('Encoding test dataset ...')
        test_X, test_Y = self.prepare_dataset_from_df(self.df_fold_test, round_encoding=round_encoding, positive_counts_only=positive_counts_only, min_counts=min_counts)
        test_dl = DataLoader(TensorDataset(test_X, test_Y), batch_size=self.batch_size_test, shuffle=False, drop_last=False)

        return train_X, train_Y, train_dl, val_X, val_Y, val_dl, test_X, test_Y, test_dl


    def get_seq_round_count_test_loader(self, round_encoding='direct', positive_counts_only=True, min_counts=-1, drop_last=False):
        '''
           for each sequence, prepare all pairs of (round number, counts at that round), sort them by round#
           input features: round encoding, onehot encoded sequences
           label: counts of input sequence at the input round
        '''
        assert(round_encoding in ['direct'])
        print('Encoding test dataset ...')
        test_X, test_Y = self.prepare_dataset_from_df(self.df_fold_test, round_encoding=round_encoding, positive_counts_only=positive_counts_only, min_counts=min_counts)
        test_Xs, test_Ys, test_dls = {}, {}, {}
        test_Xs[-1] = test_X
        test_Ys[-1] = test_Y
        test_dls[-1] = DataLoader(TensorDataset(test_Xs[-1], test_Ys[-1]), batch_size=self.batch_size_test, shuffle=False, drop_last=drop_last)
        for r in self.use_rounds:
            test_Xs[r] = test_X[torch.where(test_X[:,0]==r)]
            test_Ys[r] = test_Y[torch.where(test_X[:,0]==r)]
            test_dls[r] = DataLoader(TensorDataset(test_Xs[r], test_Ys[r]), batch_size=self.batch_size_test, shuffle=False, drop_last=drop_last)
        return test_Xs, test_Ys, test_dls


    def prepare_dataset_from_df_for_selection(self, df):
        if self.embedding is not None:
            featurized_seq = self.embedding[df['Seq_id'].unique()]
        else:
            seq_onehot = util.encode_onehot(self.seq[df['Seq_id'].unique()], alphabet=self.alphabet, verbose=False)
            featurized_seq = torch.flatten(seq_onehot, 1, 2)
        if self.max_pvalue<1: 
            seq_i_highp, r_i_highp = np.where(self.stats[:,1,:]>self.max_pvalue)
            self.stats[seq_i_highp,0,r_i_highp] = 0 # CAUTION: setting counts with high pvalues to zeros
        labels = self.stats[df['Seq_id'].unique(),0]
        labels = labels[:,self.use_rounds]
        labels = torch.tensor(labels).float()
        return featurized_seq, labels


    def get_seq_round_count_loaders_for_selection(self, drop_last=True):
        '''
           encode all the sequences and their counts over all the rounds for SPM
           input features: onehot encoded sequences
           label: vector of counts from each round
        '''
        print('Encoding training dataset ...')
        train_X, train_Y = self.prepare_dataset_from_df_for_selection(self.df_fold_train)
        train_dl = DataLoader(TensorDataset(train_X, train_Y), batch_size=self.batch_size_train, shuffle=True, drop_last=drop_last)

        print('Encoding validate dataset ...')
        val_X, val_Y = self.prepare_dataset_from_df_for_selection(self.df_fold_val)
        val_dl = DataLoader(TensorDataset(val_X, val_Y), batch_size=self.batch_size_val, shuffle=True, drop_last=drop_last)

        print('Encoding test dataset ...')
        test_X, test_Y = self.prepare_dataset_from_df_for_selection(self.df_fold_test)
        test_dl = DataLoader(TensorDataset(test_X, test_Y), batch_size=self.batch_size_test, shuffle=False, drop_last=False)

        return train_X, train_Y, train_dl, val_X, val_Y, val_dl, test_X, test_Y, test_dl


    def get_seq_round_count_test_loader_for_selection(self, drop_last=False):
        '''
           encode all the sequences and their counts over all the rounds for SPM
           input features: onehot encoded sequences
           label: vector of counts from each round
        '''
        print('Encoding test dataset ...')
        test_X, test_Y = self.prepare_dataset_from_df_for_selection(self.df_fold_test)
        test_dl = DataLoader(TensorDataset(test_X, test_Y), batch_size=self.batch_size_test, shuffle=False, drop_last=drop_last)

        return test_X, test_Y, test_dl
