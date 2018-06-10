from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import utils


import torch
import torch.nn.functional as F


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


use_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

# Set up dataset splits and ranking evaluation.
#train_triples, val_triples, test_triples = utils.train_val_test_split(all_triples, val_size=5000, test_size=5000, random_state=0)


filtered = False
train_ranker = RankingEvaluation(train_triples[:5000], num_nodes, filter_triples=all_triples if filtered else None, device=device, show_progress=True)
val_ranker = RankingEvaluation(val_triples, num_nodes, filter_triples=all_triples if filtered else None, device=device, show_progress=True)
#test_ranker = RankingEvaluation(test_triples, num_nodes, filter_triples=all_triples if filtered else None, show_progress=True)



history = utils.History()

#node_features = load_image_features(num_nodes, entity_map)
node_features = None

# Option 1: R-GCN
utils.seed_all(0)
net = UnsupervisedRGCN(num_nodes, num_relations, relational_adj_dict, train_triples, embedding_size=500, dropout=0.5,
                       num_sample_train=10, num_sample_eval=10, activation=F.elu, regularization='basis',
                       node_features=node_features, device=device).to(device)
optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, net.parameters()), lr=0.001)

train_via_ranking(net, train_triples, val_triples, optimizer, num_nodes, train_ranker, val_ranker,
                  num_epochs=35, batch_size=64, batch_size_val=512,
                  margin=1, device=device, history=history, dry_run=False)



to_plot = ['loss', 'acc', 'median_diff', 'mean_rank', 'mean_rec_rank', 'hits_1', 'hits_3', 'hits_10']
figsize = (8, 20)
history.plot(*to_plot, figsize=figsize)#, xlim=(0, 10))