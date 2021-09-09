#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sys, os, torch

from sklearn.metrics import f1_score

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
import utils as utils


parser = ArgumentParser("eval_gcn",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("--dataset", default="cora",
                    help="The dataset to be evaluated on [cora, citeseer, polblogs].")
parser.add_argument("--pert-rate", default=0.1, type=float,
                    help='Perturbation rate of edges to be flipped.')
parser.add_argument('--dimensions', type=int, default=16,
	                help='Dimensions of GCN hidden layer. Default is 16.')
parser.add_argument("--load-dir", default="../outputs",
                    help='File directory to load adversarial edges.')
args = parser.parse_args()

dataset = args.dataset
prec_flips = args.pert_rate
dim = args.dimensions
load_dir = '{}/{}.flips.npy'.format(args.load_dir, dataset)


# Load network, basic setup
_A_obs, _X_obs, _z_obs = utils.load_npz('../data/{}.npz'.format(dataset))
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
    _X_obs = np.identity(_N)
else:
    _X_obs = _X_obs[lcc].astype('float32')
_z_obs = _z_obs[lcc]
_K = _z_obs.max()+1
_Z_obs = np.eye(_K)[_z_obs]
_E = int(np.sum(_A_obs)/2)

_An = utils.preprocess_graph(_A_obs)
sizes = [16, _K]
degrees = _A_obs.sum(0).A1

# load adversarial edges
n_flips = int(prec_flips * _E)
flips = np.load(load_dir)[:n_flips]

# split train/val/test
unlabeled_share = 0.8
val_share = 0.1
train_share = 1 - unlabeled_share - val_share

split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(tuple(np.expand_dims(np.arange(_N),0)),
                                                                       train_size=train_share,
                                                                       val_size=val_share,
                                                                       test_size=unlabeled_share,
                                                                       stratify=_z_obs)


class GCN(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(n_input, dim)
        self.conv2 = GCNConv(dim, n_output)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


edge_index = []
edge_index.append(_A_obs.tocoo().row.tolist())
edge_index.append(_A_obs.tocoo().col.tolist())

edge = torch.LongTensor(edge_index)
if dataset == 'polblogs':
    feats = torch.FloatTensor(_X_obs)
else:
    feats = torch.FloatTensor(_X_obs.toarray())
labels = torch.LongTensor(_z_obs)
idx_train = torch.LongTensor(split_train)
idx_val = torch.LongTensor(split_val)
idx_test = torch.LongTensor(split_unlabeled)

model = GCN(_X_obs.shape[1], _z_obs.max() + 1)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)


best_performance = 0
early_stopping = 30
patience = 30
loss_type = 'torch'
epoch_n = 201
for epoch in range(1, epoch_n):
    optimizer.zero_grad()
    model.train()
    output = model.forward(feats, edge)
    if loss_type == 'tf':
        logp = torch.nn.functional.log_softmax(output[idx_train])
        logpy = torch.gather(logp, 1, labels[idx_train].view(-1,1))
        loss = -(logpy).mean()
        loss.backward()
    elif loss_type == 'torch':
        F.nll_loss(output[idx_train], labels[idx_train]).backward()
    optimizer.step()

    model.eval()
    output = model.forward(feats, edge)
    if loss_type == 'tf':
        logp = torch.nn.functional.log_softmax(output[idx_train])
        logpy = torch.gather(logp, 1, labels[idx_train].view(-1,1))
        val_loss = -(logpy).mean()
    elif loss_type == 'torch':
        val_loss = F.nll_loss(output[idx_val], labels[idx_val])
    val_f1_micro = f1_score(labels[idx_val], output[idx_val].max(1)[1].type_as(labels[idx_val]), average='micro')
    val_f1_macro = f1_score(labels[idx_val], output[idx_val].max(1)[1].type_as(labels[idx_val]), average='micro')
    test_f1_micro = f1_score(labels[idx_test], output[idx_test].max(1)[1].type_as(labels[idx_test]), average='micro')
    test_f1_macro = f1_score(labels[idx_test], output[idx_test].max(1)[1].type_as(labels[idx_test]), average='macro')
    print("Epoch: {}; Test micro-f1: {}; macro-f1: {}".format(epoch, test_f1_micro, test_f1_macro))

    perf_sum = val_f1_micro + val_f1_macro
    if perf_sum > best_performance:
        best_performance = perf_sum
        patience = early_stopping
    else:
        patience -= 1
    if (epoch-1) > early_stopping and patience <= 0:
        break

print("EARLY STOPPING-- Test micro-f1: {}; macro-f1: {}".format(test_f1_micro, test_f1_macro))

test_f1_micro_before = test_f1_micro
test_f1_macro_before = test_f1_macro

# attack
_A_obs_tack = _A_obs.copy().tolil()
for e in flips:
    if _A_obs[e[0], e[1]] == 0:
        _A_obs_tack[e[0], e[1]] = _A_obs_tack[e[1], e[0]] = 1
    else:
        _A_obs_tack[e[0], e[1]] = _A_obs_tack[e[1], e[0]] = 0

new_edge_index = []
new_edge_index.append(_A_obs_tack.tocoo().row.tolist())
new_edge_index.append(_A_obs_tack.tocoo().col.tolist())
edge_tack = torch.LongTensor(new_edge_index)

model.eval()
output = model.forward(feats, edge_tack)
test_f1_micro = f1_score(labels[idx_test],
                         output[idx_test].max(1)[1].type_as(labels[idx_test]),
                         average='micro')
test_f1_macro = f1_score(labels[idx_test],
                         output[idx_test].max(1)[1].type_as(labels[idx_test]),
                         average='macro')
print("gcn_retrain Test micro-f1: {}; macro-f1: {}".format(test_f1_micro, test_f1_macro))






