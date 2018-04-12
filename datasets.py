from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import collections

import utils


def load_dataset(root_dir):
    train_df = pd.read_csv(os.path.join(root_dir, 'train.txt'), sep='\t', names=['subject', 'relation', 'object'])
    val_df = pd.read_csv(os.path.join(root_dir, 'valid.txt'), sep='\t', names=['subject', 'relation', 'object'])
    test_df = pd.read_csv(os.path.join(root_dir, 'test.txt'), sep='\t', names=['subject', 'relation', 'object'])
    #print(len(train_df), len(val_df), len(test_df))
    
    entity_map = utils.IndexMap(train_df[['subject', 'object']], 
                            val_df[['subject', 'object']], 
                            test_df[['subject', 'object']])

    relation_map = utils.IndexMap(train_df['relation'], 
                                  val_df['relation'], 
                                  test_df['relation'])

    # TODO: Maybe rename num_nodes to num_entities.
    num_nodes = len(entity_map)
    num_relations = len(relation_map)
    
    def to_triples_array(df):
        subjects = df['subject'].map(entity_map.to_index)
        objects = df['object'].map(entity_map.to_index)
        relations = df['relation'].map(relation_map.to_index)
        return np.vstack([subjects, objects, relations]).T

    train_triples = to_triples_array(train_df)
    val_triples = to_triples_array(val_df)
    test_triples = to_triples_array(test_df)

    # Required for filtered ranking evaluation to check if corrupted triples appear anywhere in the dataset.
    all_triples = np.vstack([train_triples, val_triples, test_triples])
    
    return num_nodes, num_relations, train_triples, val_triples, test_triples, all_triples


def get_adj_dict(triples, undirected=False):
    adj_dict = collections.defaultdict(set)
    for s, o, r in triples:  # only use training set, so the network does not implicilty see the whole graph
        adj_dict[s].add(o)
        adj_dict[o].add(s)  # use non-directed edges
    return adj_dict


def get_relational_adj_dict(triples, undirected=False):
    relational_adj_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, o, r in triples:  # only use training set, so the network does not implicilty see the whole graph
        relational_adj_dict[s][o].append(r)
        # TODO: Try using undirected edges here as well and see if it improves the results.
    return relational_adj_dict