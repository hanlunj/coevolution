#!/bin/bash

gunzip all_sequences_onehot_FILMV.pt.gz

mkdir -p energy_barrier

python compute_energy_barriers.py \
--fitness_array=data/fitness.npz \
--alphabet=FILMV \
--output_dir=energy_barrier/ \
--sequence_array=data/all_sequences.npy \
--sequence_onehot=data/all_sequences_onehot_FILMV.pt \
--seq_to_ind_dict=traj_10000000_tol0.0-0/seq_to_ind.dict \
--sink_freqs=traj_10000000_tol0.0-0/sink_freqs.npz \
--n_top_sinks=20
