#!/usr/bin/env python
# coding: utf-8

import utils as utils

import numpy as np
import scipy.sparse as sp
import scipy.linalg as spl
from tqdm import tqdm
from sklearn.preprocessing import normalize

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser("attack",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("--dataset", default="cora",
                    help="The dataset to be perturbed on [cora, citeseer, polblogs]")
parser.add_argument("--pert-rate", default=0.1, type=float,
                    help='Perturbation rate of edges to be flipped.')
parser.add_argument("--threshold", default=0.03, type=float,
                    help='Restart threshold of eigen-solutions.')
parser.add_argument("--save-dir", default="outputs",
                    help='File directory to save outputs.')
args = parser.parse_args()

dataset = args.dataset
prec_flips = args.pert_rate
threshold = args.threshold
save_dir = '{}/{}.flips.npy'.format(args.save_dir, dataset)


# Load network, basic setup
_A_obs, _X_obs, _z_obs = utils.load_npz('data/{}.npz'.format(dataset))
_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1
lcc = utils.largest_connected_components(_A_obs)

_A_obs = _A_obs[lcc][:,lcc]
_A_obs.setdiag(0)

assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

_N = _A_obs.shape[0]
if dataset == 'polblogs':
    _X_obs = np.ones((_N, _N))
else:
    _X_obs = _X_obs[lcc].astype('float32')
_z_obs = _z_obs[lcc]
_K = _z_obs.max()+1
_Z_obs = np.eye(_K)[_z_obs]
_E = int(np.sum(_A_obs)/2)

_An = utils.preprocess_graph(_A_obs)
sizes = [16, _K]
degrees = _A_obs.sum(0).A1

# generalized eigenvalues/eigenvectors
A_tilde = _A_obs + sp.eye(_A_obs.shape[0])
vals_org, vecs_org = spl.eigh(A_tilde.toarray(), np.diag(A_tilde.sum(0).A1))

n_flips = int(prec_flips*_E)


# Choose candidate set
n_candidates = 20000
candidates = np.random.randint(0, _N, [n_candidates * 5, 2])
candidates = candidates[candidates[:, 0] < candidates[:, 1]]
candidates = np.array(list(set(map(tuple, candidates))))
candidates_all = candidates[:n_candidates]
print("# of candidates:", len(candidates_all))


def eigen_restart_flips(adj_matrix, D_, candidates, n_flips, vals_org, vecs_org, th=0.07):
    N = len(D_)
    best_edges = []
    for t in tqdm(range(n_flips)):
        flip_indicator = 1 - 2 * np.array(adj_matrix[tuple(candidates.T)])[0]
        eigen_scores = np.zeros(len(candidates))
        delta_eigvals = []
        for x in range(len(candidates)):
            i, j = candidates[x]
            delta_vals_est = flip_indicator[x] * (
                    2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2))
            delta_eigvals.append(delta_vals_est)
            vals_est = vals_org + delta_vals_est
            loss_ij = (np.sqrt(np.sum(vals_org ** 2)) - np.sqrt(np.sum(vals_est ** 2))) ** 2
            eigen_scores[x] = loss_ij
        struct_scores = - np.expand_dims(eigen_scores, 1)
        best_edge_ix = np.argsort(struct_scores, axis=0)[0]
        best_edge = candidates[best_edge_ix].squeeze()
        best_edges.append(best_edge)

        vals_org += delta_eigvals[best_edge_ix[0]]

        C, D_, adj_matrix = utils.update_delta_C_and_D(adj_matrix.copy(), D_.copy(), best_edge)

        vecs_org = normalize(vecs_org, axis=0, norm='l2')
        vecs_org[best_edge[0]] = np.sign(vals_org)*vecs_org[best_edge[0]] + \
                                 np.dot(C[best_edge[0]], vecs_org) / abs(vals_org)
        vecs_org[best_edge[1]] = np.sign(vals_org)*vecs_org[best_edge[1]] + \
                                 np.dot(C[best_edge[1]], vecs_org) / abs(vals_org)

        scale = [np.sum(vecs_org[:, i] * vecs_org[:, i] * np.diag(D_)) for i in range(N)]
        vecs_org = vecs_org / np.sqrt(scale)
        orth_matrix = np.dot(vecs_org.T * np.diag(D_), vecs_org)
        acc_pert = (np.sum(abs(orth_matrix))-N) / (N*(N-1))
        if acc_pert >= th:
            vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), np.diag(adj_matrix.sum(0).A1))

        candidates = np.concatenate((candidates[:best_edge_ix[0]],
                                     candidates[best_edge_ix[0] + 1:]))

    best_edges = np.array(best_edges)
    return best_edges


print("START attacking")
our_restart_flips = eigen_restart_flips(A_tilde.copy(), np.diag(A_tilde.sum(0).A1),
                                        candidates_all, n_flips, vals_org, vecs_org,
                                        th=threshold)
np.save(save_dir, our_restart_flips)

