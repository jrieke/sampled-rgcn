from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm
#from tqdm import tqdm_notebbok as tqdm
    
    
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
            # TODO: Change to .to('cpu')
            scores = net.decoder(subject_embeddings, object_embeddings, relations).data.cpu().numpy()
            # TOOD: If memory problems arise, use this.
            # scores = utils.predict(scoring_model, [subject_embeddings, object_embeddings, relations], batch_size=16, move_to_cuda=True, move_to_cpu=True)
            # TODO: Probably unnecessary.
            torch.cuda.empty_cache()

        score_true_triple = scores[n]
        if self.filtered:
            scores_corrupted_triples = np.delete(scores, true_triples)
            #print('Removed', len(scores) - len(scores_corrupted_triples), 'triples')
        else:
            scores_corrupted_triples = np.delete(scores, [n])

        # This could also be done directly in pytorch, but the speedup is minimal because most time is taken up
        # by calling the decoder above. Also, sp.stats.rankdata is safer on handling equal scores.
        rank = sp.stats.rankdata(-np.hstack([score_true_triple, scores_corrupted_triples]), 'average')[0]  # apply negative so highest score is 
        #print('rank ordinal:', sp.stats.rankdata(-np.hstack([score_true_triple, scores_corrupted_triples]), 'ordinal')[0])
        #print('rank average:', sp.stats.rankdata(-np.hstack([score_true_triple, scores_corrupted_triples]), 'average')[0])
        
        #rank_unfiltered = sp.stats.rankdata(-scores, 'ordinal')[n]
        #print(rank, rank_unfiltered, '--> changed', rank_unfiltered-rank, 'ranks')
        
        return rank
    
    
    def __call__(self, net, verbose=False, show_progress=False, batch_size=32):
        
        print('Running ranking evaluation for', len(self.triples), 'triples and', self.num_nodes, 'nodes with batch_size', batch_size, 'in', 'filtered' if self.filtered else 'raw', 'setting')

        # Step 1: Compute embeddings for all nodes based on the encoder.
        with torch.no_grad():
            all_nodes = torch.arange(self.num_nodes, dtype=torch.long, device=self.device)
            # TODO: Check if it is reasonable to do this via net.encoder(all_nodes), because I am not computing the gradient here, so it shouldn't consume too much memory.
            #all_node_embeddings = net.encoder(all_nodes).data
            all_node_embeddings = utils.predict(net.encoder, all_nodes, batch_size=batch_size, to_tensor=True)  # shape: num_nodes, embedding_size
            torch.cuda.empty_cache()

        # Step 2: Go through all triples and calculate the rank of the correct triple.
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

            if verbose: print(rank_subject_corrupted, rank_object_corrupted)

        del all_node_embeddings
        ranks = np.asarray(ranks)

        def hits_at(n):
            return np.sum(ranks <= n) / len(ranks)

        return np.mean(ranks), np.mean(1 / ranks), hits_at(1), hits_at(3), hits_at(10)