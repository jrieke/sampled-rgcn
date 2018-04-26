from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import random
from tqdm import tqdm
#from tqdm import tqdm_notebbok as tqdm
    
    
class RankingEvaluation(object):
    
    def __init__(self, triples, num_nodes, filter_triples=None, show_progress=False):
        self.triples = triples
        self.num_nodes = num_nodes
        self.filtered = (filter_triples is not None)

        print('Instantiating ranking evaluation with', len(triples), 'triples.')
        
        if self.filtered:

            print('Searching for corrupted triples that are actually true (these will be filtered later)...')
            
            self.true_triples_subject_corrupted_per_triple = []
            self.true_triples_object_corrupted_per_triple = []
            
            for s, o, r in tqdm(self.triples) if show_progress else self.triples:
                # TODO: Rename this.
                self.true_triples_subject_corrupted_per_triple.append(
                    filter_triples[np.logical_and(filter_triples[:, 1] == o, filter_triples[:, 2] == r)][:, 0])
                self.true_triples_object_corrupted_per_triple.append(
                    filter_triples[np.logical_and(filter_triples[:, 0] == s, filter_triples[:, 2] == r)][:, 1])
            
                #print(triple, len(self.true_triples_subject_corrupted_per_triple[-1]), len(self.true_triples_object_corrupted_per_triple[-1]))
                
            print('Subject-corrupted triples: Found on average', np.mean([len(arr) for arr in self.true_triples_subject_corrupted_per_triple]), 'triples that were actually true')
            print('Object-corrupted triples: Found on average', np.mean([len(arr) for arr in self.true_triples_object_corrupted_per_triple]), 'triples that were actually true')

        print()
                
    def _get_rank(self, scoring_model, subject_embeddings, object_embeddings, relations, n, true_triples=None):
        # TODO: torch.from_numpy(relations) is a dirty fix. Make everything with tensors instead.
        scores_gpu = scoring_model(subject_embeddings, object_embeddings, relations)
        scores = scores_gpu.cpu().data.numpy()
        #if n <= 20:
        #    print(scores[n], scores[:50])
        del scores_gpu  # TODO: Check if this improves GPU memory.
        #scores = utils.predict(scoring_model, [subject_embeddings, object_embeddings, relations], batch_size=16, move_to_cuda=True, move_to_cpu=True)
        score_true_triple = scores[n]
        
        # TODO: Maybe do not delete the scores, but set them to 0, like in ConvE code. 
        #       Especially in combination with the pytorch speedup of ranking, see below.
        if self.filtered:
            scores_corrupted_triples = np.delete(scores, true_triples)
            #print('Removed', len(scores) - len(scores_corrupted_triples), 'triples')
        else:
            scores_corrupted_triples = np.delete(scores, [n])
            
        # TODO: This takes up 53 % of the time in this function. Speed it up by doing this directly in pytorch.
        #       See ConvE code (evaluation.py):
        #       max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        #       argsort1 = argsort1.cpu().numpy()
        #       rank1 = np.where(argsort1[i]==e2[i, 0])[0][0]
        rank = sp.stats.rankdata(-np.hstack([score_true_triple, scores_corrupted_triples]), 'average')[0]  # apply negative so highest score is 
        #print('rank ordinal:', sp.stats.rankdata(-np.hstack([score_true_triple, scores_corrupted_triples]), 'ordinal')[0])
        #print('rank average:', sp.stats.rankdata(-np.hstack([score_true_triple, scores_corrupted_triples]), 'average')[0])
        
        #rank_unfiltered = sp.stats.rankdata(-scores, 'ordinal')[n]
        #print(rank, rank_unfiltered, '--> changed', rank_unfiltered-rank, 'ranks')
        
        return rank
    
    
    def __call__(self, embedding_model, scoring_model, verbose=False, show_progress=False, batch_size=32):
        
        #print('Running rank evaluation for triples:')
        #print(self.triples[:5])
        #print('...')
        #print(self.triples[-5:])
        
        # TODO: Maybe refactor this by giving all_node_embeddings as an argument here. Then each model can compute the node embeddings itselves (or if it's a simple embedding model, just give the embedding matrix), and this class only does the scoring. Then, add a function get_embedding_matrix() to RGC-layer that yield a tensor with the complete embedding matrix.
        #all_node_embeddings = embedding_model(np.arange(self.num_nodes))
        all_nodes = torch.arange(self.num_nodes).long()
        if torch.cuda.is_available():
            all_nodes = all_nodes.cuda()
        #print(batch_size)
        all_node_embeddings = utils.predict(embedding_model, all_nodes, batch_size=batch_size, to_tensor=True)
        ranks = []

        for i, triple in enumerate(tqdm(self.triples)) if show_progress else enumerate(self.triples):

            repeated_subject_embedding = all_node_embeddings[triple[0]].expand(self.num_nodes, -1)
            repeated_object_embedding = all_node_embeddings[triple[1]].expand(self.num_nodes, -1)
            # TODO: Handle cuda stuff better here.
            repeated_relation = Variable(torch.from_numpy(triple[2][None]).expand(self.num_nodes))
            if torch.cuda.is_available():
                repeated_relation = repeated_relation.cuda()

            # TODO: Refactor this.
            rank_subject_corrupted = self._get_rank(scoring_model, all_node_embeddings, repeated_object_embedding, repeated_relation, triple[0], self.true_triples_subject_corrupted_per_triple[i] if self.filtered else None)
            ranks.append(rank_subject_corrupted)
            
            rank_object_corrupted = self._get_rank(scoring_model, repeated_subject_embedding, all_node_embeddings, repeated_relation, triple[1], self.true_triples_object_corrupted_per_triple[i] if self.filtered else None)
            ranks.append(rank_object_corrupted)
            
            # TODO: Check if these are more or less the same.
            if verbose: print(rank_subject_corrupted, rank_object_corrupted)
            
        ranks = np.asarray(ranks)
        
        def hits_at(n):
            return np.sum(ranks <= n) / len(ranks)
        
        del all_node_embeddings

        return np.mean(ranks), np.mean(1 / ranks), hits_at(1), hits_at(3), hits_at(10)