#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn.metrics import f1_score
import sys, os, copy, torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from models.labelprop import LabelPropagation
import utils as utils

parser = ArgumentParser("eval_lp",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("--dataset", default="cora",
                    help="The dataset to be evaluated on [cora, citeceer, polblogs].")
parser.add_argument("--pert-rate", default=0.1, type=float,
                    help='Perturbation rate of edges to be flipped.')
parser.add_argument("--load-dir", default="../outputs",
                    help='file directory to load adversarial edges.')

args = parser.parse_args()

dataset = args.dataset
prec_flips = args.pert_rate
load_dir = '{}/{}.flips.npy'.format(args.load_dir, dataset)


# Load network, basic setup
data_path = '../data/{}.npz'.format(dataset)
_A_obs, _X_obs, _z_obs = utils.load_npz(data_path)
_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1
lcc = utils.largest_connected_components(_A_obs)

_A_obs = _A_obs[lcc][:, lcc]
_A_obs.setdiag(0)

assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
assert _A_obs.max() == 1 and len(
    np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

_N = _A_obs.shape[0]
_z_obs = _z_obs[lcc]
_E = int(np.sum(_A_obs)/2)

# load adversarial edges
n_flips = int(prec_flips * _E)
flips = np.load(load_dir)[:n_flips]

# split train/val/test
unlabeled_share = 0.8
val_share = 0.1
train_share = 1 - unlabeled_share - val_share

split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(tuple(np.expand_dims(np.arange(_N), 0)),
                                                                             train_size=train_share,
                                                                             val_size=val_share,
                                                                             test_size=unlabeled_share,
                                                                             stratify=_z_obs)

labels_train = copy.deepcopy(_z_obs)
labels_train[split_unlabeled] = -1
labels_test = _z_obs[split_unlabeled]
labels_train_t = torch.LongTensor(labels_train)
labels_test_t = torch.LongTensor(labels_test)


# Label Propagation (Before Attack)
model_before = LabelPropagation(torch.FloatTensor(_A_obs.toarray()))
model_before.fit(labels_train_t)

lp_predict = model_before.predict_classes()
lp_predict_test = lp_predict[split_unlabeled]
test_f1_micro_before = f1_score(labels_test_t, lp_predict_test, average='micro')
test_f1_macro_before = f1_score(labels_test_t, lp_predict_test, average='macro')
print("Before Test micro-f1: {}; macro-f1: {}".format(test_f1_micro_before, test_f1_macro_before))


# Attack
_A_obs_tack = _A_obs.copy().tolil()
if flips is not None:
    for e in flips:
        if _A_obs[e[0], e[1]] == 0:
            _A_obs_tack[e[0], e[1]] = _A_obs_tack[e[1], e[0]] = 1
        else:
            _A_obs_tack[e[0], e[1]] = _A_obs_tack[e[1], e[0]] = 0

model = LabelPropagation(torch.FloatTensor(_A_obs_tack.toarray()))
model.fit(labels_train_t)


# Label Propagation (After Attack)
lp_predict = model.predict_classes()
lp_predict_test = lp_predict[split_unlabeled]
test_f1_micro = f1_score(
    labels_test_t, lp_predict_test, average='micro')
test_f1_macro = f1_score(
    labels_test_t, lp_predict_test, average='macro')
if flips is None:
    test_f1_micro_before = test_f1_micro
    test_f1_macro_before = test_f1_macro
print("After Test micro-f1: {}; macro-f1: {}".format(test_f1_micro, test_f1_macro))
print("Drop: {}".format(100*(test_f1_macro_before-test_f1_macro)/test_f1_macro_before))
