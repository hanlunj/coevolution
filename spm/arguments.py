import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, help='numpy array of sequences')
    parser.add_argument('--stats', type=str, help='numpy ndarray of counts,pvalue,frequency and log_frequency for each sequence in each round [seq_index... , round_index..., [count,p-value,frequency,log_frequency]]')
    parser.add_argument('--alphabet', type=str, default='FILMV', help='alphabet of sequences')
    parser.add_argument('--data_split_random_seed', type=int, default=0, help='random seed for dataset spliting')
    parser.add_argument('--training_random_seed', type=int, default=0, help='random seed for training')
    parser.add_argument('--num_fold', type=int, default=5, help='number of kfold dataset split')
    parser.add_argument('--val_fold_idx', type=int, default=0, help='index of the validation dataset from kfold dataset split')
    parser.add_argument('--test_fold_idx', type=int, default=1, help='index of the test dataset from kfold dataset split')
    parser.add_argument('--subsample_train', type=float, default=1.0, help='subsample by this fraction of the training dataset')
    parser.add_argument('--train_on_all', action='store_true', help='if enabled, train on all data (ignoring the validation/test fold idx)')
    parser.add_argument('--split_type', type=str, default='lastseenround', help='dataset split method (stratified,lastseenround,random)')
    parser.add_argument('--max_pvalue', type=float, default=1.0, help='only use counts with p value (from hypergeometric test) equal or less than this cutoff')
    parser.add_argument('--use_rounds', type=str, default='', help='only use specified rounds (comma-sperated, first round is 0) of selection data for training; default is to use all')
    parser.add_argument('--device', type=str, default='cpu', help='device for training')
    parser.add_argument('--gpu_id', type=int, default=-1, help='overwrite to select which gpu to use')
    parser.add_argument('--data_split_dir', type=str, default='data', help='directory for storing dataset splits')
    parser.add_argument('--embedding_file_list', type=str, default='', help='file that lists (esm) embeddings')

    # parameters for non-SPM model
    parser.add_argument('--round_encoding', type=str, default='direct', help='encoding method for round number')
    parser.add_argument('--positive_counts_only', action='store_true', help='only use sequence with counts > 0')

    # parameters for SPM
    parser.add_argument('--use_selection', action='store_true', help='use selection model with negative log likelihood loss')
    parser.add_argument('--constrained_energy', action='store_true', help='for selection model, impose monotonically increasing energies over rounds')
    parser.add_argument('--fit_init_round', action='store_true', help='for selection model, allow model to train on the inital round (Round 0) counts')

    # for evaluation only
    parser.add_argument('--min_counts', type=int, default=-1, help='only use sequence with counts > this number')
    parser.add_argument('--write_prediction', action='store_true', help='if enabled, write prediction and labels in .npy')

    # for counts prediction and bootstrapping only
    parser.add_argument('--n_samples', type=int, default=1, help='number of samples to draw using multinomial.rvs. If set to 0, compute the counts directly from probability (i.e. no sampling via multinomial.rvs)')

    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('--max_no_improve', type=int, default=3, help='stop training when no improvement on training loss after this number of epoch')
    parser.add_argument('--max_epoch', type=int, default=200, help='max number of epoches for training')
    parser.add_argument('--model', type=str, default='ThreeLayerFCRegressionRELU', help='model to use for training')
    parser.add_argument('--model_weights', type=str, default=None, help='.pt file with pretrained model weights for evaluation')
    parser.add_argument('--model_weights_list', type=str, default=None, help='list of .pt file with pretrained model weights for evaluation')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for Adam optimizer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--hiddim', type=int, default=100, help='hidden dimension for the neural network linear layers')
    parser.add_argument('--verbose', action='store_true', help='verbose')

    args = parser.parse_args()

    return args
