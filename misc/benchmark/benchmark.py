# SPDX-License-Identifier: MIT

import argparse
import gc
import time
import math
import subprocess
import sys
sys.path.insert(0, '/people/ondel/Repositories/pychain')

import torch
import simplefst
import pychain
from pychain.graph import ChainGraph, ChainGraphBatch
from pychain.loss import ChainLoss
import pychain_C

pychain_C.set_verbose_level(0)
torch.set_num_threads(1)

def make_hmm(fstfile, B, log_domain=True):
    #state = 0
    #with open('hmm.txt', 'w') as f:
    #    for s in range(1, S):
    #        print(f'{s} {s} {s} {s} {-math.log(1/2)}', file=f)
    #        print(f'{s} {s+1} {s} {s+1} {-math.log(1/2)}', file=f)
    #    print(f'{S} {S} {S} {S} 0', file=f)
    #    print(f'{S}', file=f) #subprocess.run(['fstcompile', 'hmm.txt', 'hmm.fst'])

    #fst = simplefst.StdVectorFst.read('hmm.fst')
    #graph = ChainGraph(fst, log_domain=log_domain)

    fst = simplefst.StdVectorFst.read(fstfile)
    graph = ChainGraph(fst, log_domain=log_domain)
    return ChainGraphBatch(graph, batch_size=B)

def main(fstfile, N, B):
    lang = 'python'
    precision = 'single' # we always use float32 in the following

    graphs = make_hmm(fstfile, B)
    S = graphs.num_states-1
    data = -torch.ones(B, N, 84, dtype=torch.float32).contiguous()
    lengths = [N for i in range(B)]
    data_lengths = torch.tensor(lengths, dtype=torch.int32)

    for device in ['cpu', 'cuda:0']:

        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            data, data_lengths, batch_first=True,
        )
        batch_sizes = packed_data.batch_sizes

        graphs = make_hmm(fstfile, B)
        forward_transitions = graphs.forward_transitions.to(device)
        forward_transition_indices = graphs.forward_transition_indices.to(device)
        forward_transition_probs = graphs.forward_transition_probs.to(device)
        backward_transitions = graphs.backward_transitions.to(device)
        backward_transition_indices = graphs.backward_transition_indices.to(device)
        backward_transition_probs = graphs.backward_transition_probs.to(device)
        initial_probs = graphs.initial_probs.to(device)
        final_probs = graphs.final_probs.to(device)
        start_state = graphs.start_state.to(device)
        data = data.to(device)
        data_lengths = data_lengths.to(device)

        t1 = time.time()
        objf, log_probs_grad, ok = pychain_C.forward_backward_log_domain(
            forward_transitions,
            forward_transition_indices,
            forward_transition_probs,
            backward_transitions,
            backward_transition_indices,
            backward_transition_probs,
            initial_probs,
            final_probs,
            start_state,
            data,
            batch_sizes,
            data_lengths,
            graphs.num_states,
        )
        t2 = time.time()
        dev = 'gpu' if device == 'cuda:0' else 'cpu'
        print(f'{lang}\t{precision}\t{B}\t{S}\t{N}\tpychain_log\t{dev}\t{t2 - t1}')

        data.exp_()
        graphs = make_hmm(fstfile, B, log_domain=False)
        forward_transitions = graphs.forward_transitions.to(device)
        forward_transition_indices = graphs.forward_transition_indices.to(device)
        forward_transition_probs = graphs.forward_transition_probs.to(device)
        backward_transitions = graphs.backward_transitions.to(device)
        backward_transition_indices = graphs.backward_transition_indices.to(device)
        backward_transition_probs = graphs.backward_transition_probs.to(device)
        leaky_probs = graphs.leaky_probs.to(device)
        initial_probs = graphs.initial_probs.to(device)
        final_probs = graphs.final_probs.to(device)
        start_state = graphs.start_state.to(device)
        data = data.to(device)
        data_lengths = data_lengths.to(device)

        t1 = time.time()
        objf, log_probs_grad, ok = pychain_C.forward_backward(
            forward_transitions,
            forward_transition_indices,
            forward_transition_probs,
            backward_transitions,
            backward_transition_indices,
            backward_transition_probs,
            leaky_probs,
            initial_probs,
            final_probs,
            start_state,
            data,
            batch_sizes,
            data_lengths,
            graphs.num_states,
            1e-3
        )
        t2 = time.time()
        dev = 'gpu' if device == 'cuda:0' else 'cpu'
        print(f'{lang}\t{precision}\t{B}\t{S}\t{N}\tpychain_leaky\t{dev}\t{t2 - t1}')


if __name__ == '__main__':
    main(
            'den_fsm_wsj.fst',
            128,
            700,
    )
