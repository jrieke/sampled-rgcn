from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from sklearn.model_selection import ParameterGrid

import utils
from models import UnsupervisedRGCN, DistMult
from training import train_via_ranking, train_via_classification
from evaluation import RankingEvaluation
from datasets import load_graph, load_image_features
import os


# ---------------------- Settings ----------------------------
use_cuda = True
dry_run = False
ranking_eval = True
filtered = True
graph_dir = 'data/fb15k-237/Release'
#graph_dir = '../data/wn18rr'
#graph_dir = '../data/yago3-10'
# ------------------------------------------------------------


num_nodes, num_relations, train_triples, val_triples, test_triples, all_triples, entity_map, relation_map = load_graph(graph_dir)

device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
print('Using device:', device)

train_ranker = RankingEvaluation(train_triples[:5000], num_nodes, triples_to_filter=all_triples if filtered else None, device=device, show_progress=True)
val_ranker = RankingEvaluation(val_triples, num_nodes, triples_to_filter=all_triples if filtered else None, device=device, show_progress=True)
#test_ranker = RankingEvaluation(test_triples, num_nodes, filter_triples=all_triples if filtered else None, show_progress=True)

node_features = None



log_dir = 'logs/1'
# TODO: Fix this.
#log_index = utils.next_log_index('logs')
param_grid = {'embedding_size': [200, 500], 'dropout': [0.2, 0.5, 0.8], 'lr': [0.001, 0.0005], 'batch_size': [128]}

print()
print()

def run(i, params, history):
    # TODO: Make device parameter obsolete by moving everything to the device once .to(device) is called.
    # net = UnsupervisedRGCN(num_nodes, num_relations, train_triples, embedding_size=params['embedding_size'],
    #                        dropout=params['dropout'], num_sample_train=params['num_sample_train'],
    #                        num_sample_eval=params['num_sample_eval'], activation=F.elu,
    #                        node_features=node_features, device=device)
    net = DistMult(params['embedding_size'], num_nodes, num_relations, params['dropout'])
    net.to(device)
    optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, net.parameters()), lr=params['lr'])

    train_via_classification(net, train_triples, val_triples, optimizer, num_nodes, train_ranker, val_ranker,
                             num_epochs=10, batch_size=params['batch_size'], batch_size_eval=512, device=device,
                             history=history, dry_run=dry_run, ranking_eval=ranking_eval,
                             save_best_to=os.path.join(log_dir, '{}_best-model_epoch-{{epoch}}.pt'.format(i)))

g = utils.GridSearch(log_dir, param_grid)
g.run(run)




# print('Starting hyperparameter optimization (logging to index {})'.format(log_index))
# print('='*80)
# for i, params in enumerate(ParameterGrid(param_grid)):
#     print('Performing run {} with parameters {}'.format(i, params))
#     print('='*80)
#     history = utils.History(desc='Run {} of hyperparameter search started on {}'.format(i, utils.get_timestamp()), params=params)
#
#
#     # TODO: Make device parameter obsolete by moving everything to the device once .to(device) is called.
#     # net = UnsupervisedRGCN(num_nodes, num_relations, train_triples, embedding_size=params['embedding_size'],
#     #                        dropout=params['dropout'], num_sample_train=params['num_sample_train'],
#     #                        num_sample_eval=params['num_sample_eval'], activation=F.elu,
#     #                        node_features=node_features, device=device)
#     net = DistMult(params['embedding_size'], num_nodes, num_relations, params['dropout'])
#     net.to(device)
#     optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, net.parameters()), lr=params['lr'])
#
#     train_via_classification(net, train_triples, val_triples, optimizer, num_nodes, train_ranker, val_ranker,
#                              num_epochs=30, batch_size=params['batch_size'], batch_size_eval=512, device=device,
#                              history=history, dry_run=dry_run, ranking_eval=ranking_eval,
#                              save_best_to='logs/{}-{}_best-model_epoch-{{epoch}}.pt'.format(log_index, i))
#
#     history.save('logs/{}-{}.json'.format(log_index, i))