from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import utils


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


from models import UnsupervisedRGCN
from training import train_via_ranking, train_via_classification
from evaluation import RankingEvaluation
from datasets import load_graph, get_adj_dict, get_relational_adj_dict, load_image_features


root_dir = '../data/fb15k-237/Release'
#root_dir = '../data/wn18rr'
#root_dir = '../data/yago3-10'

num_nodes, num_relations, train_triples, val_triples, test_triples, all_triples, entity_map, relation_map = load_graph(root_dir)
# TODO: Maybe put those two methods into one with relations=True.
relational_adj_dict = get_relational_adj_dict(train_triples)
#adj_dict = get_adj_dict(train_triples, undirected=True)


# Set up dataset splits and ranking evaluation.
#train_triples, val_triples, test_triples = utils.train_val_test_split(all_triples, val_size=5000, test_size=5000, random_state=0)

filtered = False
train_ranker = RankingEvaluation(train_triples[:5000], num_nodes, filter_triples=all_triples if filtered else None, show_progress=True)
val_ranker = RankingEvaluation(val_triples, num_nodes, filter_triples=all_triples if filtered else None, show_progress=True)
#test_ranker = RankingEvaluation(test_triples, num_nodes, filter_triples=all_triples if filtered else None, show_progress=True)


force_cpu = False
history = utils.History()

#node_features = load_image_features(num_nodes, entity_map)
node_features = None

# Option 1: R-GCN
utils.seed_all(0)
net = UnsupervisedRGCN(num_nodes, num_relations, relational_adj_dict, train_triples, embedding_size=500, dist_mult_dropout=0.5,
                       num_sample_train=10, num_sample_eval=10, activation=F.elu, regularization='basis',
                       node_features=node_features)
embedding_func, scoring_func = net.graph_conv1, net.decoder  # required for ranking
if torch.cuda.is_available() and not force_cpu:
    net.cuda()
    print('Moved network to GPU')
optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, net.parameters()), lr=0.001)
batch_size = 64


### TODO: Handle the embedding_func/scoring_func here better.
#       Maybe through function train_ranking_func = lambda: train_ranker(embedding_func, scoring_func) above.
#       Or define training functions in here and use most variables from the global namespace.
#       Then: train_via_ranking(net, optimizer, num_epochs, batch_size, margin, history=None)
train_via_ranking(net, train_triples, val_triples, optimizer, num_nodes, train_ranker, val_ranker,
                  embedding_func, scoring_func, num_epochs=35, batch_size=batch_size, batch_size_val=16,
                  margin=1, history=history, dry_run=False)
