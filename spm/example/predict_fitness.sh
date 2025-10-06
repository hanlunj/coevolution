#!/bin/bash

python ../predict_fitness.py \
--seq=data/test_sequences_onehot_FILMV.pt \
--alphabet=FILMV \
--seq_onehot \
--num_rounds=7 \
--device=cuda \
--gpu_id=0 \
--use_selection \
--constrained_energy \
--batch_size=10000 \
--model=FourLayerFCRegressionRELU \
--model_weights=models/FourLayerFCRegressionRELU_hiddim100_dataseed0_trainseed0_trainall_lr0.001_bs10000_dropout0.1.pt \
--hiddim=100  \
--verbose 



