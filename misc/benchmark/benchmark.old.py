# SPDX-License-Identifier: MIT

import argparse
import gc
import time
import torch

torch.set_num_threads(1)

def make_hmm(T, S):
    p = 1/2
    A = torch.diag(torch.ones(S, dtype=T)*p)
    A += torch.diag(torch.ones(S-1, dtype=T)*p, 1)
    A = torch.log(A)
    A[-1,-1] = 0

    init = torch.zeros(S, dtype=T)
    init[0] = 1.0
    init = torch.log(init)

    final= torch.zeros(S, dtype=T)
    final[-1] = 1.0
    final = torch.log(final)

    return A, init, final


def forward(A, init, lhs):
    log_alphas = torch.zeros_like(lhs) - float('inf')
    log_alphas[0] = lhs[0] + init
    At = A.t()
    for i in range(1, lhs.shape[0]):
        log_alphas[i] = lhs[i] + torch.logsumexp(log_alphas[i-1] + At, dim=1).view(-1)
    return log_alphas

def backward(A, final, lhs):
    log_betas = torch.zeros_like(lhs) - float('inf')
    log_betas[-1] = final
    for i in reversed(range(lhs.shape[0]-1)):
        log_betas[i] = torch.logsumexp(A + log_betas[i+1] + lhs[i+1], dim=1).view(-1)
    return log_betas

def forward_backward(A, init, final, lhs):
    log_alphas = forward(A, init, lhs)
    log_betas = backward(A, final, lhs)
    lgammas = log_alphas + log_betas
    lognorm = torch.logsumexp(lgammas, dim=1)
    return lgammas - lognorm[:, None], torch.min(lognorm)

def main(args):
    N = args.num_frames
    S = args.num_states
    T = torch.float32 if args.single_precision else torch.float64

    lang = 'python'
    precision = 'double' if T == torch.float64 else 'single'

    lhs = torch.zeros(N, , dtype=T)
    A, init, final = make_hmm(T, S)

    t1 = time.time()
    lg, ttl = forward_backward(A, init, final, lhs)
    t2 = time.time()
    print(f'{lang}\t{precision}\t{S}\t{N}\tstandard\tdense\tcpu\t{t2 - t1}')

    device = torch.device('cuda')
    A = A.to(device)
    init = init.to(device)
    final = final.to(device)
    lhs = lhs.to(device)

    t1 = time.time()
    forward_backward(A, init, final, lhs)
    t2 = time.time()
    print(f'{lang}\t{precision}\t{S}\t{N}\tstandard\tdense\tgpu\t{t2 - t1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", "-N", type=int, default=1000,
                        help="number of observations frames")
    parser.add_argument("--num-states", "-S", type=int, default=1000,
                        help="number of states")
    parser.add_argument("--single-precision", action='store_true',
                        help="use single precision")
    args = parser.parse_args()
    main(args)
