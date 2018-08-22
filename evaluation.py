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


def rank(tensor, n, dim=None):
    """
    PyTorch function similar to `scipy.stats.rankdata`.
    Returns the rank of the element at index `n` within the `tensor`.

    If the element at index `n` is the highest value in `tensor`, it has rank 1,
    if it is the 2nd-highest value, it has rank 2, etc.

    Note: This cannot handle equal values in `tensor` in the same way that `scipy.stats.rankdata` can.
    """
    _, sorted_indices = torch.sort(tensor, dim=dim, descending=True)
    return (sorted_indices == n).nonzero()[0][0] + 1

    
class RankingEvaluation(object):
    
    def __init__(self, triples, num_nodes, triples_to_filter=None, device='cuda', show_progress=False):
        self.triples = triples
        self.num_nodes = num_nodes
        self.filtered = (triples_to_filter is not None)
        self.device = device

        print('Creating ranking evaluation with', len(triples), 'triples and', num_nodes, 'nodes in', 'filtered' if self.filtered else 'raw', 'setting')
        
        if self.filtered:

            print('Searching for corrupted triples that are actually true (these will be filtered later)...')
            
            self.true_triples_subject_corrupted_per_triple = []
            self.true_triples_object_corrupted_per_triple = []
            
            for s, o, r in tqdm(self.triples) if show_progress else self.triples:
                # TODO: Rename this.
                self.true_triples_subject_corrupted_per_triple.append(
                    triples_to_filter[np.logical_and(triples_to_filter[:, 1] == o, triples_to_filter[:, 2] == r)][:, 0])
                self.true_triples_object_corrupted_per_triple.append(
                    triples_to_filter[np.logical_and(triples_to_filter[:, 0] == s, triples_to_filter[:, 2] == r)][:, 1])
            
                #print(triple, len(self.true_triples_subject_corrupted_per_triple[-1]), len(self.true_triples_object_corrupted_per_triple[-1]))
                
            print('Subject-corrupted triples: Found on average', np.mean([len(arr) for arr in self.true_triples_subject_corrupted_per_triple]), 'triples that were actually true')
            print('Object-corrupted triples: Found on average', np.mean([len(arr) for arr in self.true_triples_object_corrupted_per_triple]), 'triples that were actually true')
            print()


    def _get_rank(self, net, subject_embeddings, object_embeddings, relations, n, true_triples=None):
        with torch.no_grad():
            scores = net.decoder(subject_embeddings, object_embeddings, relations).data.cpu().numpy()
            # TODO: If memory problems arise, do the line above via utils.predict, similar to:
            # scores = utils.predict(scoring_model, [subject_embeddings, object_embeddings, relations], batch_size=16, move_to_cuda=True, move_to_cpu=True)
            # TODO: Probably unnecessary.
            torch.cuda.empty_cache()

        if self.filtered:  # set the scores for all triples, which are contained in the dataset, to 0
            scores[true_triples] = 0

        # While it is possible to do this directly in pytorch (via torch.sort), sp.stats.rankdata handles equal
        # scores better, and most of the time here is taken up by the calls to the decoder anyways.
        rank = sp.stats.rankdata(-scores, 'average')[n]  # apply negative so highest score is

        # rank_unfiltered = sp.stats.rankdata(-scores, 'ordinal')[n]
        # print(rank, rank_unfiltered, '--> changed', rank_unfiltered-rank, 'ranks')

        return rank
    
    
    def __call__(self, net, verbose=False, show_progress=False, batch_size=32):
        
        print('Running ranking evaluation for', len(self.triples), 'triples and', self.num_nodes, 'nodes with batch_size', batch_size, 'in', 'filtered' if self.filtered else 'raw', 'setting')
        
        # TODO: Maybe refactor this by giving all_node_embeddings as an argument here. Then each model can compute the node embeddings itself (or if it's a simple embedding model, just give the embedding matrix), and this class only does the scoring. Then, add a function get_embedding_matrix() to RGC-layer that yield a tensor with the complete embedding matrix.
        #all_node_embeddings = embedding_model(np.arange(self.num_nodes))
        with torch.no_grad():
            all_nodes = torch.arange(self.num_nodes, dtype=torch.long, device=self.device)
            #all_node_embeddings = net.encoder(all_nodes).data#utils.predict(embedding_model, all_nodes, batch_size=batch_size, to_tensor=True)
            # TODO: Check if it is reasonable to do this via net.encoder(all_nodes), because I am not computing the gradient here, so it shouldn't consume too much memory.
            all_node_embeddings = utils.predict(net.encoder, all_nodes, batch_size=batch_size, to_tensor=True)  # shape: num_nodes, embedding_size
        torch.cuda.empty_cache()
        ranks = []

        for i, triple in enumerate(tqdm(self.triples)) if show_progress else enumerate(self.triples):

            repeated_subject_embedding = all_node_embeddings[triple[0]].expand(self.num_nodes, -1)
            repeated_object_embedding = all_node_embeddings[triple[1]].expand(self.num_nodes, -1)
            # TODO: Handle cuda stuff better here.
            repeated_relation = torch.tensor(triple[2][None], dtype=torch.long, device=self.device).expand(self.num_nodes)

            # TODO: Refactor this.
            rank_subject_corrupted = self._get_rank(net, all_node_embeddings, repeated_object_embedding, repeated_relation, triple[0], self.true_triples_subject_corrupted_per_triple[i] if self.filtered else None)
            ranks.append(rank_subject_corrupted)
            
            rank_object_corrupted = self._get_rank(net, repeated_subject_embedding, all_node_embeddings, repeated_relation, triple[1], self.true_triples_object_corrupted_per_triple[i] if self.filtered else None)
            ranks.append(rank_object_corrupted)
            
            # TODO: Check if these are more or less the same.
            if verbose: print(rank_subject_corrupted, rank_object_corrupted)
            
        ranks = np.asarray(ranks)
        
        def hits_at(n):
            return np.sum(ranks <= n) / len(ranks)
        
        del all_node_embeddings

        return np.mean(ranks), np.mean(1 / ranks), hits_at(1), hits_at(3), hits_at(10)