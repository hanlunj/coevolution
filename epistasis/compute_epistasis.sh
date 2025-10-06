#!/bin/bash

gunzip all_sequences.npy.gz 

python compute_epistasis.py 
