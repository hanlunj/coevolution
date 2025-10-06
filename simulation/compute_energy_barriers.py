import numpy as np
from glob import glob
from tqdm import tqdm
import sys, os
import multiprocessing as mp
import torch
import utils
from heapq import heapify, heappush, heappop
import pickle
import argparse

parser = argparse.ArgumentParser(description="Run energy barrier computation with multiprocessing.")

parser.add_argument("--fitness_array", type=str, required=True,
                    help="Numpy array of fitness of all sequences in .npy or .npz format")
parser.add_argument("--alphabet", type=str, required=True,
                    help="Amino acid alphabet used for all the sequences")
parser.add_argument("--sequence_array", type=str, required=True,
                    help="Numpy array of all sequences in .npy format")
parser.add_argument("--sequence_onehot", type=str, required=True,
                    help="1D pytorch tensor of onehot encoded sequences")
parser.add_argument("--seq_to_ind_dict", type=str, required=True,
                    help="Hastable for sequence to index")
parser.add_argument("--interest_seq_idx", type=str, default='',
                    help="1D numpy array of indices for sequences of interest")
parser.add_argument("--sink_freqs", type=str, default='',
                    help="1D numpy array of sink frequencies for identify top sinks")
parser.add_argument("--n_top_sinks", type=int, default=0,
                    help="Number of top accessible sinks for energy barrier analysis")
parser.add_argument("--output_dir", type=str, default='./',
                    help="Directory for output")

args = parser.parse_args()

fitness_array = args.fitness_array
alphabet = args.alphabet

if __name__ == '__main__':

    hashseed = os.getenv('PYTHONHASHSEED')
    if not hashseed:
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable] + sys.argv)

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    mp.set_start_method('fork')

    rkey = np.random.default_rng(821038)

    # choose problem to work on
    
    if fitness_array.endswith('.npy'):
        fitness = np.load(fitness_array)
    elif fitness_array.endswith('.npz'):
        fitness = np.load(fitness_array, allow_pickle=True)['arr_0']
    else:
        raise('Error: unrecognized fitness_array file format', fitness_array)
    preds = -fitness
    print('Loading sequences...')
    seq = np.load(args.sequence_array, allow_pickle=True)
    seq_onehot = torch.load(args.sequence_onehot).numpy()
    n, n_pos, n_alph = seq_onehot.shape
    n_neighbors = n_pos*n_alph
    print('Loading seq_to_ind_dict...')
    with open(args.seq_to_ind_dict, 'rb') as fin:
        hseq_to_ind = pickle.load(fin)

    # Load sequences of interest (sinks)
    if args.n_top_sinks!=0 and args.sink_freqs!='':
        sink_freqs = np.load(args.sink_freqs, allow_pickle=True)['arr_0']
        interest_seq_idx = sink_freqs.argsort()[-args.n_top_sinks:] 
    elif args.interest_seq_idx!='':
        interest_seq_idx = np.load(args.interest_seq_idx)
    else:
        raise('Error: must specify args.sink_freqs or args.interest_seq_idx!')
    sink_inds = interest_seq_idx
    print('Sinks for analysis: ', sink_inds.shape)
    print(sink_inds)
    
    # Set save string for output files
    save_str = args.output_dir
    
    ############ helpers ###############
    
    def get_ind_from_seq(seq):
        h = hash(seq)
        return int(hseq_to_ind[h])

    def get_neighbors(curr):
        '''curr is index into seq_onehot'''
        neighbor_list = []
        mut_inds = [(a,b) for a in range(n_pos) for b in range(n_alph)] # (n_pos, n_alph)
        assert len(mut_inds) == n_neighbors
        for i in range(n_neighbors):
            try:
                # make seq space mutation
                s = utils.mut_seq(seq[curr], mut_inds[i][0], alphabet, mut_inds[i][1]) 
                # get ind from sequence instead
                ind = get_ind_from_seq(s)
                neighbor_list.append(ind)
            except Exception as e: # not all neighbors from onehot are in the library definition
                # print(f'get_neighbors: {e}, skipping.'); 
                continue
        return neighbor_list

    def energy_difference(source, target):
        return preds[target] - preds[source]

    def energy_barrier(source, target):
        return float(max(0, energy_difference(source, target)))

    def energy_barrier_sum(path):
        e = 0
        for i in range(1, len(path)):
            e += energy_barrier(path[i-1], path[i])
        return e

    def energy_difference_abssum(path):
        e = 0
        for i in range(1, len(path)):
            e += abs(energy_difference(path[i-1], path[i]))
        return e

    def energy_difference_allsum(path):
        e = 0
        for i in range(1, len(path)):
            e += energy_difference(path[i-1], path[i])
        return e

    # heap stuff + dijkstra's

    REMOVED = -1     # placeholder for a removed task

    def add_task(task, priority, pq, entry_finder):
        'Add a new task or update the priority of an existing task'
        if task in entry_finder:
            remove_task(task, entry_finder)
        entry = [priority, task]
        entry_finder[task] = entry
        heappush(pq, entry)

    def remove_task(task, entry_finder):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = entry_finder.pop(task)
        entry[-1] = REMOVED

    def pop_task(pq, entry_finder):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while pq:
            priority, task = heappop(pq)
            if task is not REMOVED:
                del entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    def shortest_path(source, target):
        '''
        source and target are indices of sequences.
        Return: python List, path of indices [source, ..., target].
        '''
        prev = {} # node : source of node
        dist = {} # node : distance from source node; if not present => INF
        ######## DEFINE PRIORITY QUEUE ########
        pq = []              # elements have (priority, index); priority = dist from source
        entry_finder = {}    # mapping of tasks to entries
        pq_add = lambda task, priority: add_task(task, float(priority), pq, entry_finder)
        pq_remove = lambda task: remove_task(task, entry_finder)
        pq_pop = lambda : pop_task(pq, entry_finder)
        ######################################
        visited = set()
        pq_add(source, 0)
        prev[source] = -1
        dist[source] = 0
        while pq:
            try:
                curr = pq_pop()
            except Exception as e:
                print(e)
                print(pq)
            visited.add(curr)
            if curr == target:
                break
            else:
                neighbors = get_neighbors(curr)
                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    # only count increases in energy
                    alt = dist[curr] + energy_barrier(curr, neighbor)
                    if neighbor not in dist or alt < dist[neighbor]:
                        dist[neighbor] = alt
                        prev[neighbor] = curr
                        pq_add(neighbor, dist[neighbor])
        if target not in prev:
            print('Failed to find target')
        # reconstruct path from prev
        path = [target]
        try:
            while prev[path[-1]] != -1:
                path.append(prev[path[-1]])
        except Exception as e:
            print(f'Failed to find a path: {e}')
            return path[::-1], prev, dist
        return path[::-1], prev, dist

    ####

    # run dijkstra's
    # NOTE: this is actually not symmetric, because we use
    # asymmetric energy_barrier() to determine "shortness" of paths
    try:
        e_barriers = np.load(save_str+'energy_barrier.npy')
        assert e_barriers.size == sink_inds.size**2 # assume rest are same size
        rev_e_barriers = np.load(save_str+'energy_barrier_rev.npy')
        e_abs_diff = np.load(save_str+'energy_absdiff.npy')
        e_sum_diff = np.load(save_str+'energy_sumdiff.npy')
        print(f'Successfully loaded in from {save_str}_*.npy')
        print(f'abs_diff: {np.count_nonzero(~np.isnan(e_abs_diff))} elements already done.')
        with open(save_str+'paths.pkl', 'rb') as fin:
            paths = pickle.load(fin)
        p = 0
        for k in paths:
            p += len(paths[k])
        print(f'paths: {p} elements already done (this takes precedent).')
    except Exception as e:
        print(e)
        print('Setting all zeros, starting from scratch.')
        paths = {} # sink : sink2 : path List[]
        e_barriers = np.ones((sink_inds.size, sink_inds.size)) * np.nan
        rev_e_barriers = np.ones((sink_inds.size, sink_inds.size)) * np.nan
        e_abs_diff = np.ones((sink_inds.size, sink_inds.size)) * np.nan
        e_sum_diff = np.ones((sink_inds.size, sink_inds.size)) * np.nan
    
    print(f'{sink_inds.size**2} elements to run total')
    # NOTE: this could also be done via multiprocessing to speed up...
    path_lens = []
    for i in tqdm(range(sink_inds.size)):
        for j in tqdm(range(sink_inds.size), leave=False):
            a, b = sink_inds[i], sink_inds[j]
            if a in paths and b in paths[a] and ~np.isnan(e_abs_diff[i,j]):
                print('pair already done, continuing')
                continue
            path, prev, dist = shortest_path(a, b)
            if a not in paths:
                paths[a] = {}
            paths[a][b] = path
            path_lens.append(len(path))
            e_barriers[i,j] = energy_barrier_sum(path)
            rev_e_barriers[i,j] = energy_barrier_sum(path[::-1])
            e_abs_diff[i,j] = energy_difference_abssum(path)
            e_sum_diff[i,j] = energy_difference_allsum(path)

    p = 0
    for k in paths:
        p += len(paths[k])
    print(p, np.count_nonzero(e_abs_diff), np.count_nonzero(e_sum_diff), np.count_nonzero(e_barriers))
    print(np.mean(path_lens), np.min(path_lens), np.max(path_lens))

    np.save(save_str+'energy_barrier.npy', e_barriers)
    np.save(save_str+'energy_barrier_rev.npy', rev_e_barriers)
    np.save(save_str+'energy_absdiff.npy', e_abs_diff)
    np.save(save_str+'energy_sumdiff.npy', e_sum_diff)
    with open(save_str+'_paths.pkl','wb') as fout:
        pickle.dump(paths, fout)
    
