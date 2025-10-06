#!/bin/bash

python ../train.py \
--seq=../data/synthetic_all_rounds_seqs.npy \
--stats=../data/synthetic_all_rounds_stats.npy \
--alphabet=FILMV \
--data_split_random_seed=0 \
--training_random_seed=0 \
--num_fold=5 \
--train_on_all \
--split_type=lastseenround \
--device=cuda \
--gpu_id=0 \
--data_split_dir=data \
--use_selection \
--constrained_energy \
--batch_size=10000 \
--max_no_improve=10 \
--max_epoch=500 \
--model=FourLayerFCRegressionRELU \
--lr=0.001 \
--weight_decay=1e-5 \
--dropout=0.1 \
--hiddim=100 \
--verbose 
