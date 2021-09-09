#!/usr/bin/env python
# coding: utf-8


import numpy as np
import networkx as nx
import sys, os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from models.node2vec import Node2vec
import utils as utils


parser = ArgumentParser("eval_node2vec",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("--dataset", default="cora",
                    help="the dataset to be perturbed on [cora, citeseer, polblogs].")
parser.add_argument("--pert-rate", default=0.1, type=float,
                    help='perturbation rate of edges to be flipped.')
parser.add_argument('--dimensions', type=int, default=32,
	                help='Output embedding dimensions of Node2vec. Default is 32.')
parser.add_argument('--window-size', type=int, default=5,
                    help='Context size for optimization. Default is 5.')
parser.add_argument('--walk-length', type=int, default=80,
                    help='Length of walk per source. Default is 80.')
parser.add_argument('--walk-num', type=int, default=10,
                    help='Number of walks per source. Default is 10.')
parser.add_argument('--p', type=float, default=4.0,
                    help='Parameter in node2vec. Default is 4.0.')
parser.add_argument('--q', type=float, default=1.0,
                    help='Parameter in node2vec. Default is 1.0.')
parser.add_argument('--worker', type=int, default=10,
                    help='Number of parallel workers. Default is 10.')
parser.add_argument("--load-dir", default="../outputs",
                    help='file directory to load adversarial edges.')
args = parser.parse_args()

dataset = args.dataset
prec_flips = args.pert_rate
dim = args.dimensions
window_size = args.window_size
walk_length = args.walk_length
walk_num = args.walk_num
p = args.p
q = args.q
workers = args.worker
load_dir = '{}/{}.flips.npy'.format(args.load_dir, dataset)


# Load network, basic setup
data_path = '../data/{}.npz'.format(dataset)
A_obs, X_obs, z_obs = utils.load_npz(data_path)
A_obs = A_obs + A_obs.T
A_obs[A_obs > 1] = 1
lcc = utils.largest_connected_components(A_obs)

A_obs = A_obs[lcc][:, lcc]
A_obs.setdiag(0)

assert np.abs(A_obs - A_obs.T).sum() == 0, "Input graph is not symmetric"
assert A_obs.max() == 1 and len(
    np.unique(A_obs[A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
assert A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

N = A_obs.shape[0]
z_obs = z_obs[lcc]
K = z_obs.max()+1
Z_obs = np.eye(K)[z_obs]
_E = int(np.sum(A_obs)/2)

_An = utils.preprocess_graph(A_obs)
sizes = [16, K]
degrees = A_obs.sum(0).A1

# load adversarial edges
n_flips = int(prec_flips * _E)
flips = np.load(load_dir)[:n_flips]

# split train/val/test
unlabeled_share = 0.8
val_share = 0.1
train_share = 1 - unlabeled_share - val_share

split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(tuple(np.expand_dims(np.arange(N), 0)),
                                                                             train_size=train_share,
                                                                             val_size=val_share,
                                                                             test_size=unlabeled_share,
                                                                             stratify=z_obs)

# Node2vec embedding (Before Attack)
graph_before = nx.from_scipy_sparse_matrix(A_obs.copy())
model_before = Node2vec(dimension=dim, walk_length=walk_length, window_size=window_size, iteration=1,
                 walk_num=walk_num, worker=workers, p=p, q=q)
embedding_before = model_before.train(graph_before)

features_train = embedding_before[np.concatenate([split_val, split_train], 0)]
features_test = embedding_before[split_unlabeled]
labels_train = z_obs[np.concatenate([split_val, split_train], 0)]
labels_test = z_obs[split_unlabeled]

lr_before = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
lr_before.fit(features_train, labels_train)

lr_z_predict = lr_before.predict(features_test)
test_f1_micro_before = f1_score(labels_test, lr_z_predict, average='micro')
test_f1_macro_before = f1_score(labels_test, lr_z_predict, average='macro')

print("Before Test micro-f1: {}; macro-f1: {}".format(test_f1_micro_before, test_f1_macro_before))


# Attack
A_obs_tack = A_obs.copy().tolil()
if flips is not None:
    for e in flips:
        if A_obs[e[0], e[1]] == 0:
            A_obs_tack[e[0], e[1]] = A_obs_tack[e[1], e[0]] = 1
        else:
            A_obs_tack[e[0], e[1]] = A_obs_tack[e[1], e[0]] = 0


# Node2vec embedding (After Attack)
graph_after = nx.from_scipy_sparse_matrix(A_obs_tack.copy())
model_after = Node2vec(dimension=dim, walk_length=walk_length, window_size=window_size, iteration=1,
                 walk_num=walk_num, worker=workers, p=p, q=q)
embedding_after = model_after.train(graph_after)

features_train = embedding_after[np.concatenate([split_val, split_train], 0)]
features_test = embedding_after[split_unlabeled]
labels_train = z_obs[np.concatenate([split_val, split_train], 0)]
labels_test = z_obs[split_unlabeled]

lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
lr.fit(features_train, labels_train)

lr_z_predict = lr.predict(features_test)
test_f1_micro = f1_score(labels_test, lr_z_predict, average='micro')
test_f1_macro = f1_score(labels_test, lr_z_predict, average='macro')

print("After Test micro-f1: {}; macro-f1: {}".format(test_f1_micro, test_f1_macro))
print("Drop: {}".format((test_f1_macro_before - test_f1_macro) / test_f1_macro_before))