import numpy as np
import glob
from functools import partial
from tqdm import tqdm, trange
import multiprocessing as mp
import os, sys
import argparse
import pickle
from utils import mut_seq

parser = argparse.ArgumentParser(description="Run trajectory simulation with multiprocessing.")

parser.add_argument("--rng_seed", type=int, required=True, help="Random seed (required positional argument)")
parser.add_argument("--num_procs", type=int, default=1,
                    help="Number of processes to use for multiprocessing (default: 1)")
parser.add_argument("--chunk_size", type=int, default=100,
                    help="Chunk size for multiprocessing (default: 100)")
parser.add_argument("--num_traj", type=int, default=10000000,
                    help="Number of trajectories to simulate (default: 10000000)")
parser.add_argument("--max_traj_steps", type=int, default=500,
                    help="Maximum number of steps per trajectory (default: 500)")
parser.add_argument("--tol", type=float, default=0.0,
                    help="Tolerance for noise in fitness landscape (default: 0.0)")
parser.add_argument("--sequence_array", type=str, required=True,
                    help="Numpy array of all sequences in .npy format")
parser.add_argument("--fitness_array", type=str, required=True,
                    help="Numpy array of fitness of all sequences in .npy or .npz format")
parser.add_argument("--output_dir", type=str, default='./',
                    help="Directory for output")
parser.add_argument("--exp_name", type=str, default="0",
                    help="Experiment name or ID (default: '0')")
parser.add_argument("--alphabet", type=str, default="ACDEFGHIKLMNPQRSTVWY",
                    help="Amino acid alphabet used for all the sequences")

args = parser.parse_args()

# Assign to variables
RNG_SEED = args.rng_seed
NUM_PROCS = args.num_procs
CHUNK_SIZE = args.chunk_size
NUM_TRAJ = args.num_traj
MAX_TRAJ_STEPS = args.max_traj_steps
TOL = args.tol
output_dir = args.output_dir
sequence_array = args.sequence_array
fitness_array = args.fitness_array
EXP_NAME = args.exp_name
alphabet = args.alphabet

if __name__ == '__main__':
    # Step 0: Set up environment and multiprocessing.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    mp.set_start_method('fork')
    rkey = np.random.default_rng(RNG_SEED)

    # Step 1 (in README.md): Load data. Setup save dir.
    print(f'Step 1: Loading data.')

    seq = np.load(sequence_array, allow_pickle=True)
    ### NOTE: alphabet could be made per-position, but currently is not.
    ### This is accounted for in `single_trajectory` -- dict lookup will fail if not in library.
    if fitness_array.endswith('.npy'):
        fitness = np.load(fitness_array)
    elif fitness_array.endswith('.npz'):
        fitness = np.load(fitness_array, allow_pickle=True)['arr_0']
    else:
        raise('Error: unrecognized fitness_array file format', fitness_array)
   
    n, n_pos, n_alph = seq.shape[0], len(seq[0]), len(alphabet)
    n_neighbors = n_pos*n_alph
    print(f'\tLoaded data: {n} sequences, {n_pos} positions, {n_alph} alphabet size.')
    savedir = output_dir + f'traj_{NUM_TRAJ}_tol{TOL}-{EXP_NAME}/'
    print(f'\tExperiment will be saved to {savedir}')
    os.makedirs(savedir, exist_ok=True)
    
    # Step 2 (in README.md): Generate the sequence to index mapping.
    print(f'Step 2: Generating sequence to index mapping.')

    ### Step 2a: Make sure hashing is consistent. Not strictly necessary.
    hashseed = os.getenv('PYTHONHASHSEED')
    if not hashseed:
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    seq_to_ind_dict = {}
    count, collision_set = 0, set()

    ### Step 2b: Populate the dictionary. Optionally, can check it. See `hash_str_to_ind.py`.
    for i in trange(0, n):
        h = hash(seq[i])
        if h in collision_set:
            print('Collision:', i, h, seq[i])
        collision_set.add(h)
        count += 1
        seq_to_ind_dict[h] = i

    with open(f'{savedir}seq_to_ind.dict', 'wb') as fout:
        pickle.dump(seq_to_ind_dict, fout)
    print(f"\tSaved to {savedir}seq_to_ind.dict. {n - len(collision_set)} hashing collisions.")
    print(f"{n - len(collision_set)} hashing collisions.")
    del collision_set

    # Step 3 (in README.md): Run trajectories.
    print(f'Step 3: Running {NUM_TRAJ} trajectories.')

    ### Step 3a: Get a random permutation of indices to start trajectories from.
    ### NOTE: if you want NUM_TRAJ > n, you'll need to repeat indices. Just do k*n, k>1.
    ordering = rkey.permutation(n)[:NUM_TRAJ]

    ### Step 3b: Define various utility functions.
    def get_ind_from_seq(seq):
        h = hash(seq)
        return int(seq_to_ind_dict[h])

    def single_trajectory(start_ind, max_steps): 
        n_steps = 0
        traj_inds = [start_ind%n] # mod n in case NUM_TRAJ > n
        while n_steps < max_steps:
            found_improvement = False
            ordering = rkey.permutation(n_neighbors)
            mut_inds = [(a,b) for a in range(n_pos) for b in range(n_alph)] # (n_pos, n_alph)
            for j in ordering:
                try: # make seq space mutation, not in-place.
                    s = mut_seq(seq[traj_inds[-1]], mut_inds[j][0], alphabet, mut_inds[j][1])
                    ind = get_ind_from_seq(s) # get ind instead (easier storage)
                except Exception as e: # Not all neighbors are in the library definition.
                    continue # Means that get_ind_from_seq lookup failed.
                if fitness[ind] > fitness[traj_inds[-1]] - TOL:
                    traj_inds.append(ind)
                    found_improvement = True
                    break # take any fitness-improving mutation, not necessarily the best one
            if not found_improvement:
                break # local maximum reached!
            n_steps += 1
        return traj_inds[0], np.array(traj_inds, dtype=np.int32)

    def run_single_trajectory(start_ind):
        return single_trajectory(start_ind, MAX_TRAJ_STEPS)

    ### Step 3c: Launch trajectories in parallel.
    print(f'\tStarting multiprocessing with {NUM_PROCS} processes, {CHUNK_SIZE} chunk size.')
    with mp.Pool(NUM_PROCS) as pool:
        ind_traj_list = list(tqdm(
            pool.imap_unordered(
                run_single_trajectory,
                ordering, 
                chunksize=CHUNK_SIZE
            ), total=NUM_TRAJ
        ))
    ### Convert list to dict, of lists. May have duplicate keys.
    ind_to_traj_dict = {}
    for ind, traj in ind_traj_list:
        if ind not in ind_to_traj_dict:
            ind_to_traj_dict[ind] = []
        ind_to_traj_dict[ind].append(traj)
    n_starts = len(ind_to_traj_dict)
    #print(f'\tTrajectories finished. Saved to {savedir}. {n_starts} unique start sequences.')
    print(f'\tTrajectories finished. {n_starts} unique start sequences.')

    ### Step 3d: Aggregate trajectories and compute statistics.
    print(f'\tAggregating trajectories and computing statistics.')
    ### NOTE: statistic for each sequence is saved as value @ its ind in array.
    #visited_arr = np.zeros_like(fitness).astype(int) # not needed for now
    sinks_arr = np.zeros_like(fitness).astype(int)
    #lens_arr = np.ones((len(f_inds),))*(-1)  # not needed for now
    for i, (start_ind, traj_list) in tqdm(enumerate(ind_to_traj_dict.items()), total=n_starts):
        for traj in traj_list:
            inds_unique = np.unique(traj[1:]) # could pass >1 if tol>0.
            if inds_unique.size > 0: # could randomly start at a sink
                #visited_arr[inds_unique] += 1  # not needed for now
                p = fitness[inds_unique]
                best = np.argmax(p)
                sinks_arr[inds_unique[best]] += 1
                #lens_arr[i] = len(traj)  # not needed for now
    #np.save(f"{savedir}visit_freqs.npy", visited_arr) # not needed for now
    #np.save(f"{savedir}sink_freqs.npy", sinks_arr)
    np.savez_compressed(f"{savedir}sink_freqs.npz", sinks_arr)
    #np.save(f"{savedir}traj_lengths.npy", lens_arr)  # not needed for now

    sinks = {}
    nz = sinks_arr.nonzero()
    assert len(nz) == 1
    for ind in nz[0]:
        sinks[ind] = sinks_arr[ind]
    with open(f"{savedir}sinks.dict.pkl", 'wb') as fout:
        pickle.dump(sinks, fout)
    print(f'\tSaved visit frequencies, sink frequencies, trajectory lengths, and sinks dict to {savedir}.')
    print('Trajectories complete.')
