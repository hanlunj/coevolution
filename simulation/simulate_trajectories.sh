#!/bin/bash

gunzip data/all_sequences.npy.gz 

python simulate_trajectories.py \
--rng_seed=0 \
--num_procs=20 \
--chunk_size=100 \
--num_traj=10000000 \
--max_traj_steps=500 \
--tol=0.0 \
--sequence_array=data/all_sequences.npy \
--fitness_array=data/fitness.npz \
--output_dir=./ \
--exp_name=0 \
--alphabet=FILMV

