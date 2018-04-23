from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import random
import collections
from block import block


def block_diagonal(blocks):
    """Return a block-diagonal matrix constructed from the matrices in blocks."""
    # TODO: Check if it's faster to construct a list of lists here, or do this in pytorch directly.
    block_matrix = np.zeros((len(blocks), len(blocks)), dtype='object')
    block_matrix[np.diag_indices_from(block_matrix)] = blocks
    return block(block_matrix.tolist())


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, in_features_func, adj_dict, 
                 num_sample_train=10, num_sample_eval=None, activation=F.relu):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_func = in_features_func
        self.adj_dict = adj_dict
        self.num_sample_train = num_sample_train
        self.num_sample_eval = num_sample_eval
        # TODO: Add adj_array (see RelationalGraphConvolution).
        self.activation = activation
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)
        
        
    # TODO: Refactor this.
    # TODO: Profile this with lprun and see how much time each step takes up (especially creating the tensors!).
    def aggregate(self, nodes, adj_dict, include_self=False):
        """Return a vector for each node in `nodes` by mean-aggregating feature vectors from its neighborhood (or a sample thereof)."""
        
        num_sample = self.num_sample_train if self.training else self.num_sample_eval
        
        # Step 1: For each node, sample some neighbors that we aggregate information from. 
        
        # TODO: Move this up to constructor or even outside of this class.
        #adj_array = np.array([adj_dict[i] for i in range(num_nodes)])
        #sampled_neighbors_per_node = adj_array[nodes]
        #print(sampled_neighbors_per_node)
        sampled_neighbors_per_node = [adj_dict[node] for node in nodes]
        
        # TODO: Replace stuff below by this line, check if it works.
        #sampled_neighbors_per_node = [adj_lists[node].update([node]) if include_self else adj_lists[node] for node in nodes]
        
        # TODO: Check if this improves the network. 
        #       Also, in graphsage, this is done after the sampling step (which doesn't make much sense though).
        # TODO: Check that node is not added to adj_lists permanently.
        if include_self:
            for sampled_neighbors, node in zip(sampled_neighbors_per_node, nodes):
                sampled_neighbors.add(node)
        
        if num_sample is not None:
            # TODO: See if local pointers bring speed improvements (see GraphSage code).
            for i in range(len(sampled_neighbors_per_node)):
                if num_sample <= len(sampled_neighbors_per_node[i]):
                    sampled_neighbors_per_node[i] = set(random.sample(sampled_neighbors_per_node[i], num_sample))
                    
                    
        # Step 2: Find the unique neighbors in all sampled neighbors.
        
        unique_neighbors = list(set.union(*sampled_neighbors_per_node))
        unique_neighbors_to_index = {neighbor: i for i, neighbor in enumerate(unique_neighbors)}
        
        if unique_neighbors:
            
            # Step 3: Get embeddings for these unique neighbors from the underlying layer 
            #         (e.g. another GraphConvolution layer or a fixed feature matrix).
            
            unique_neighbors_tensor = torch.LongTensor(unique_neighbors)
            if self.is_cuda():
                unique_neighbors_tensor = unique_neighbors_tensor.cuda()
            unique_neighbors_embeddings = self.in_features_func(unique_neighbors_tensor)
            

            # Step 4: For each input node, sum the embeddings of its (sampled) neighbors, 
            #         using the embeddings obtained above. The algorithm here uses a masking matrix 
            #         to find the neighbor embeddings for each node and add them in one step. 
            
            # TODO: See if this can be implemented in an easier way (i.e. without mask).
            
            # TODO: Check if this really needs to be a variable.
            # TODO: Maybe store mask in self and fill it with zeros here, so it doesn't have to initalized in each forward pass. 
            mask = Variable(torch.zeros(len(nodes), len(unique_neighbors)), requires_grad=False)
            if self.is_cuda():
                mask = mask.cuda()
            # TODO: Understand and rename variables here.
            column_indices = [unique_neighbors_to_index[neighbor] for sampled_neighbors in sampled_neighbors_per_node for neighbor in sampled_neighbors]
            row_indices = [i for i in range(len(sampled_neighbors_per_node)) for j in range(len(sampled_neighbors_per_node[i]))]
            mask[row_indices, column_indices] = 1

            # TODO: Is this actually the number of neighbors, i.e. does it correspond 
            #       to the normalization constant in RGCN paper?
            num_neighbors = mask.sum(1, keepdim=True)
            mask /= num_neighbors + 1e-10  # prevent zero division
            
            #set_trace()
            
            # TODO: Check what the actual output of this term is.
            return mask.mm(unique_neighbors_embeddings)
        else:
            # TODO: If there are no neighbors, this currently returns a zero vector. Is this correct?
            # TODO: Building this variable and moving it to cuda takes up a lot of time (15 % for Link Prediction),
            #       speed this up somehow.
            zeros = Variable(torch.zeros(len(nodes), self.in_features), requires_grad=False)
            if self.is_cuda():
                zeros = zeros.cuda()
            return zeros

        
    def forward(self, nodes):
        neighborhood_features = self.aggregate(nodes, self.adj_dict, include_self=True)
        
        # TODO: Maybe add features of nodes themselves,
        #       or include nodes themselves in sampling process in aggregate function (as in GraphSAGE,
        #       but collides with RGCN use of aggregate function).
        #node_features = self.in_features_func(torch.LongTensor(nodes))
        
        return self.activation(neighborhood_features.mm(self.weight.t()))
    
    def is_cuda(self):
        return self.weight.is_cuda


class RelationalGraphConvolution(GraphConvolution):
    def __init__(self, in_features, out_features, num_nodes, num_relations, in_features_func, relational_adj_dict, 
                 num_sample_train=10, num_sample_eval=None, activation=F.relu, regularization=None, 
                 num_basis_weights=2, num_blocks=10, verbose=False, cuda=True):
        
        GraphConvolution.__init__(self, in_features, out_features, num_nodes, in_features_func, None, 
                                                         num_sample_train, num_sample_eval, activation)
        
        if regularization not in [None, 'basis', 'block']:
            raise ValueError("regularization has to be one of [None, 'basis', 'block']")
        
        # TODO: Maybe create adj_dict from train_triples directly in here.
        self.relational_adj_dict = relational_adj_dict
        self.adj_array = np.array([set(relational_adj_dict[i].keys()) for i in range(num_nodes)], dtype=object)
        
        self.num_relations = num_relations
        self.regularization = regularization
        self.verbose = verbose
        
        if self.regularization is None:
            # Storing the relation weights in a ParameterList instead of a 3D tensor is more memory-efficient,
            # because the gradient is only computed for this specific weight then.
            self.relation_weights = nn.ParameterList()
            for relation in range(num_relations):
                weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
                nn.init.xavier_uniform(weight)
                self.relation_weights.append(weight)
        elif regularization == 'basis':
            self.basis_weights = nn.Parameter(torch.FloatTensor(num_basis_weights, out_features, in_features))
            nn.init.xavier_uniform(self.basis_weights)
            self.coefficients = nn.Parameter(torch.FloatTensor(num_relations, num_basis_weights))
            nn.init.xavier_uniform(self.coefficients)
        elif regularization == 'block':
            block_width = in_features / num_blocks
            block_height = out_features / num_blocks
            if not block_width.is_integer() or not block_height.is_integer():
                raise ValueError('in_features and out_features has to be divisible by num_blocks')
                
            if verbose: print('Constructing blocks of size', block_width, ' - ', block_height)
            self.blocks_per_relation = nn.ParameterList()
            for relation in range(num_relations):
                blocks = nn.Parameter(torch.FloatTensor(num_blocks, int(block_width), int(block_height)))
                nn.init.xavier_uniform(blocks)
                self.blocks_per_relation.append(blocks)
            
            
        self.zero = torch.zeros(1)
        # TODO: Figure out a way to move zero to cuda as soon as cuda is called on the net.
        if cuda:
            self.zero = self.zero.cuda()
            
    def get_relation_weight(self, relation):
        if self.regularization is None:
            return self.relation_weights[relation]
        elif self.regularization == 'basis':
            # TODO: Or could I calculate this once above and reuse it?
            #print(relation)
            return (self.coefficients[relation].view(-1, 1, 1) * self.basis_weights).sum(0)
        elif self.regularization == 'block':
            return block_diagonal(self.blocks_per_relation[relation])
        
    def get_relation_weights(self, relations):
        if self.regularization is None:
            return self.relation_weights[relations]
        else:
            # TODO: Or could I calculate this once above and reuse it?
            #print(relation)
            # TODO: Make this quicker.
            return torch.cat([self.get_relation_weight(relation) for relation in relations])
        
        
    def sample_neighbors(self, nodes):
        sampled_neighbors_per_node = self.adj_array[nodes]
        num_sample = self.num_sample_train if self.training else self.num_sample_eval
        if num_sample is not None:
            # TODO: Check if local pointers bring speed improvements (see GraphSage code).
            for i in range(len(sampled_neighbors_per_node)):
                if num_sample <= len(sampled_neighbors_per_node[i]):
                    # TODO: Check if np.random.choice is faster.
                    sampled_neighbors_per_node[i] = set(random.sample(sampled_neighbors_per_node[i], num_sample))
                    
        if self.verbose: print('Sampled', num_sample, 'neighbors per node:', sampled_neighbors_per_node) 
        return sampled_neighbors_per_node
                    
            
    def forward(self, nodes):
                
        if self.verbose: print('Calculating embeddings for nodes:', nodes)
                
        # Step 1: Get embeddings for all nodes in the mini-batch from the underlying layer.
        #         (e.g. another GraphConvolution layer or a fixed feature matrix).
        input_embeddings = self.in_features_func(nodes)
        if self.verbose: print('Got input_embeddings:', input_embeddings)
        
        # TODO: Maybe add hyperparameter that determines which fraction of the self-embedding to use, 
        #       and which fraction of the aggregated embedding. See if this has an effect 
        #       or if weights find the best way to combine the two embeddings themselves.
        #       Alternatively, include self-embedding in sampling.
        # TODO: Maybe add dropout of the self-connection.
        output_embeddings = input_embeddings.mm(self.weight.t())  # terms for relations will be added to this
        if self.verbose: print('Multiplying them by weight matrix, preliminary output_embeddings are:', output_embeddings)
                
        # Step 2: For each node, sample some neighbors that we aggregate information from. 
        sampled_neighbors_per_node = self.sample_neighbors(nodes)
        
        # TODO: Do all of this in pytorch. 
        #       Store a dict of tensors triples_per_node. Then use 
        #       torch.IntTensor(num_sample).random(len(triples_per_node[node])) to sample some triples.
        #       Then use torch.unique() (only pytorch 0.4!) to find unique values from this matrix.
        
            
        # Step 3: Find the unique neighbors in all sampled neighbors. 
        #         If there are no unique neighbors, return zero embeddings.
        unique_neighbors = list(set.union(*sampled_neighbors_per_node))
        unique_neighbors_to_index = {neighbor: i for i, neighbor in enumerate(unique_neighbors)}
        if self.verbose: print('Unique neighbors are:', unique_neighbors)
        
        if not unique_neighbors:
            # TODO: Maybe create and store this once, and return it each time here.
            if self.verbose: print('No unique neighbors found, returning 0.')
            return Variable(self.zero.repeat(len(nodes), self.out_features))
        
        # Step 3: Get embeddings for these unique neighbors from the underlying layer 
        #         (e.g. another GraphConvolution layer or a fixed feature matrix).
        # TODO: Investigate how much time processing only the unique neighbors actually saves vs the overhead 
        #       done here to reassign unique neighbors to the nodes (for a realistic batch size!).
        unique_neighbors = torch.LongTensor(unique_neighbors)
        if self.is_cuda():
            unique_neighbors = unique_neighbors.cuda()
            
        unique_neighbors_embeddings = self.in_features_func(unique_neighbors)
        if self.verbose: print('Got input embeddings for unique neighbors:', unique_neighbors_embeddings)
            
        #print(np.sum([len(neighbors) for neighbors in sampled_neighbors_per_node]) / len(unique_neighbors_embeddings) )
                
        # Step 4: For each relation, average the embeddings of sampled neighbors, 
        #         using the embeddings obtained above. The algorithm here uses a masking matrix 
        #         to find the neighbor embeddings for each node and add them in one step. 
        # TODO: Try creating torch.zeros(1, 1).cuda() in the constructor, and expanding it here via repeat.
        
        
        #column_indices_per_relation = collections.defaultdict(list)
        #row_indices_per_relation = collections.defaultdict(list)
        
        # This is the solution with a 3D tensor using all relations at once.
        #masks = Variable(self.zero.repeat(self.num_relations, len(nodes), len(unique_neighbors)))
        #for i in range(len(masks)):
            # TODO: Do this for the whole tensor at once.
        #    num_neighbors_per_node = masks[i].sum(1, keepdim=True)
        #    masks[i] /= num_neighbors_per_node + 1e-10  # prevent zero division
            
        # TODO: Check that mask is initialized to 0 each time.
        mask = Variable(self.zero.repeat(len(nodes), len(unique_neighbors)), requires_grad=False)
        masks = collections.defaultdict(lambda: mask.clone())
        #for relation in range(self.num_relations):
        #    masks[relation] = mask.clone()
        for i, (node, sampled_neighbors) in enumerate(zip(nodes, sampled_neighbors_per_node)):
            for neighbor in sampled_neighbors:
                for relation in self.relational_adj_dict[node][neighbor]:
                    masks[relation][i, unique_neighbors_to_index[neighbor]] = 1
                    #column_indices_per_relation[relation].append(unique_neighbors_to_index[neighbor])
                    
        #print([(k, v.sum()) for k, v in masks.items()])
        
        #for node, sampled_neighbors in zip(nodes, sampled_neighbors_per_node):
        #    for neighbor in sampled_neighbors:
        #        for relation in self.relational_adj_dict[node][neighbor]:
        #            column_indices_per_relation[relation].append(unique_neighbors_to_index[neighbor])
              
        # TODO: Integrate this in loop above.
        #for i, node in enumerate(nodes):
        #    for j, neighbor in enumerate(sampled_neighbors_per_node[i]):
        #        for relation in self.relational_adj_dict[node][neighbor]:
        #            row_indices_per_relation[relation].append(i)
                    
        # TODO: Maybe build all masks here and then do bmm with relation weights.
        
        #mask = Variable(torch.zeros(len(nodes), len(unique_neighbors)), requires_grad=False)
        #if self.is_cuda():
        #    mask = mask.cuda()
        
        #masks = Variable(torch.zeros(len(column_indices_per_relation), len(nodes), len(unique_neighbors)), requires_grad=False)
        
            
        
        if self.verbose: print('Iterating over all relations and summing neighbor embeddings for each relation.')

        for relation, mask in masks.iteritems():
            # TODO: Check which of these three alternatives for masking is quickest:
            #       1) Iterate over all nodes and sampled neighbor, and assign column/row indices only 
            #          if edge has relation (current implementation).
            #          Potentially iterates very often over the same arrays.
            #       2) Compile one object sampled_neighbors_per_node_per_relation beforehand, 
            #          and do the usual mask-building for each relation in there.
            #       3) Build one mask for each relation and set values in the mask belonging 
            #          to the relation directly.
            
            
            if self.verbose: print('Relation', relation, 'out of', self.num_relations)

            #column_indices = []
            #for node, sampled_neighbors in zip(nodes, sampled_neighbors_per_node):
            #    for neighbor in sampled_neighbors:
            #        if relation in self.relational_adj_dict[node][neighbor]:
            #            column_indices.append(unique_neighbors_to_index[neighbor])
                        
            #if column_indices:  # if column_indices is empty, there are no sampled neighbors with this relation, so skip it
            #    row_indices = []
            #    for i, node in enumerate(nodes):
            #        for j, neighbor in enumerate(sampled_neighbors_per_node[i]):
            #            if relation in self.relational_adj_dict[node][neighbor]:
            #                row_indices.append(i)
                            
            if self.verbose: print('Mask:', mask)
            #if self.verbose: print('Column indices:', column_indices_per_relation[relation])
            #if self.verbose: print('Row indices:', row_indices_per_relation[relation])

            #mask.zero_()
            #mask[row_indices_per_relation[relation], column_indices_per_relation[relation]] = 1

            # TODO: Here, I divide by the number of neighbors with this relation.
            #       In R-GCN paper, they divide by the number of all neighbors (across relations).
            #       Implement this by summing up these values over all relations.
            #       Also ask Michael, how the results with other normalization constants were.
            num_neighbors_per_node = mask.sum(1, keepdim=True)
            mask /= num_neighbors_per_node + 1e-10  # prevent zero division

            # Non-zero value at (i, j) in mask means that embedding of unique neighbor j contributes 
            # to aggregated embedding for node i (weighted by the value in mask). Formally:
            # aggregated_node_embeddings[i] = \sum_j {mask[i, j] * unique_neighbors_embeddings[j]}
            aggregated_embeddings = mask.mm(unique_neighbors_embeddings)
            if self.verbose: print('Aggregated neighbor embeddings for each node:', aggregated_embeddings)

            # Step 5: Multiply the aggregated feature vectors with the relation-specific weight matrix.
            output_embeddings += aggregated_embeddings.mm(self.get_relation_weight(relation).t())
            if self.verbose: print('Multiplying them by relation weight matrix, preliminary output_embeddings are:', output_embeddings)

            mask = mask.clone()
            
        output_embeddings = self.activation(output_embeddings)
        if self.verbose: print('Applied non-linearity, final output embeddings are:', output_embeddings)
        if self.verbose: print('-'*80)
        return output_embeddings
        
    
        
        
# TODO: Implement this new version that uses triples directly. The current implementation still uses one relation per edge!
# TODO: Check runtime and performance against old version.
class NewRelationalGraphConvolution(GraphConvolution):
    def __init__(self, in_features, out_features, num_nodes, num_relations, in_features_func, train_triples, 
                 num_sample=10, activation=F.relu):
        
        super(RelationalGraphConvolution, self).__init__(in_features, out_features, num_nodes, in_features_func, None, num_sample, activation)
        
        # TODO: Maybe create adj_dict from train_triples directly in here.
        #self.relational_adj_dict = relational_adj_dict
        #self.adj_array = np.array([set(relational_adj_dict[i].keys()) for i in range(num_nodes)], dtype=object)
        self.train_triples = train_triples.copy()
        
        self.num_relations = num_relations
        
        # TODO: Maybe store the relation weights in a 3D tensor instead of a ParameterList. 
        #       Uses about the same amount of memory. Check if it brings speed improvements.
        self.relation_weights = nn.ParameterList()
        for relation in range(num_relations):
            self.relation_weights.append(nn.Parameter(torch.FloatTensor(out_features, in_features)))
            nn.init.xavier_uniform(self.relation_weights[relation])
        #self.weights_per_relation = nn.Parameter(torch.FloatTensor(num_relations, out_features, in_features))
            
    def forward(self, nodes):
        # TODO: Refactor this by passing nodes as tensor in the first place.
        if type(nodes) == torch.LongTensor or type(nodes) == torch.cuda.LongTensor:
            nodes_tensor = nodes
        else:
            nodes_tensor = torch.LongTensor(nodes)
            if self.is_cuda():
                nodes_tensor = nodes_tensor.cuda()
                
        # Step 1: Get embeddings for all nodes in the mini-batch from the underlying layer.
        #         (e.g. another GraphConvolution layer or a fixed feature matrix).
        input_embeddings = self.in_features_func(nodes_tensor)
        output_embeddings = input_embeddings.mm(self.weight.t())  # terms for relations will be added to this
                
                
        # Step 2: For each node, sample some neighbors that we aggregate information from. 
        #sampled_neighbors_per_node = [set(self.relational_adj_dict[node]) for node in nodes]
        # TODO: Refactor this to method sample_neighbors(nodes), which uses self.num_sample and self.relational_adj_dict.
#         sampled_neighbors_per_node = self.adj_array[nodes]
#         if self.num_sample is not None:
#             # TODO: Check if local pointers bring speed improvements (see GraphSage code).
#             for i in range(len(sampled_neighbors_per_node)):
#                 if self.num_sample <= len(sampled_neighbors_per_node[i]):
#                     # TODO: Check if np.random.choice is faster.
#                     sampled_neighbors_per_node[i] = set(random.sample(sampled_neighbors_per_node[i], self.num_sample))
        # TODO: Check how long this takes. Maybe create triples_per_node for all nodes in constructor.
        sampled_triples_per_node = [train_triples[train_triples[:, 0] == node] for node in nodes]
        if self.num_sample is not None:
            for i in range(len(sampled_triples_per_node)):
                if self.num_sample <= len(sampled_triples_per_node[i]):
                    sampled_triples_per_node[i] = sampled_triples_per_node[i][np.random.choice(len(sampled_triples_per_node[i], size=self.num_sample))]

            
        # Step 3: Find the unique neighbors in all sampled neighbors. 
        #         If there are no unique neighbors, return zero embeddings.
        unique_neighbors = np.unique(np.hstack([sampled_triples[:, 1] for sampled_triples in sampled_triples_per_neighbor]))#list(set.union(*sampled_neighbors_per_node))
        unique_neighbors_to_index = {neighbor: i for i, neighbor in enumerate(unique_neighbors)}
        
        # TODO: Check if this works with array.
        if not unique_neighbors:
            # TODO: Maybe create and store this once, and return it each time here.
            zeros = Variable(torch.zeros(len(nodes), self.out_features), requires_grad=False)
            if self.is_cuda():
                zeros = zeros.cuda()
            print('No unique neighbors, returning zero. Nodes:', nodes)
            return zeros
        
        # Step 3: Get embeddings for these unique neighbors from the underlying layer 
        #         (e.g. another GraphConvolution layer or a fixed feature matrix).
        # TODO: Investigate how much time processing only the unique neighbors actually saves vs the overhead 
        #       done here to reassign unique neighbors to the nodes (for a realistiv batch size!)
        unique_neighbors_tensor = torch.LongTensor(unique_neighbors)
        if self.is_cuda():
            unique_neighbors_tensor = unique_neighbors_tensor.cuda()
        unique_neighbors_embeddings = self.in_features_func(unique_neighbors_tensor)
                
        # Step 4: For each relation, average the embeddings of sampled neighbors, 
        #         using the embeddings obtained above. The algorithm here uses a masking matrix 
        #         to find the neighbor embeddings for each node and add them in one step. 
        mask = Variable(torch.zeros(len(nodes), len(unique_neighbors)), requires_grad=False)
        if self.is_cuda():
            mask = mask.cuda()

        for relation in range(self.num_relations):
            # TODO: Check which of these three alternatives for masking is quickest:
            #       1) Iterate over all nodes and sampled neighbor, and assign column/row indices only 
            #          if edge has relation (current method).
            #          Potentially iterates very often over the same arrays.
            #       2) Compile one object sampled_neighbors_per_node_per_relation beforehand, 
            #          and do the usual mask-building for each relation in there.
            #       3) Build one mask for each relation and set values in the mask belonging 
            #          to the relation directly.

            column_indices = []
            for node, sampled_neighbors in zip(nodes, sampled_neighbors_per_node):
                for neighbor in sampled_neighbors:
                    if relation in self.relational_adj_dict[node][neighbor]:
                        column_indices.append(unique_neighbors_to_index[neighbor])
                        
            if column_indices:  # if column_indices is empty, there are no sampled neighbors with this relation, so skip it
                row_indices = []
                for i, node in enumerate(nodes):
                    for j, neighbor in enumerate(sampled_neighbors_per_node[i]):
                        if relation in self.relational_adj_dict[node][neighbor]:
                            row_indices.append(i)

                mask[row_indices, column_indices] = 1
                
                # TODO: Here, I divide by the number of neighbors with this relation.
                #       In R-GCN paper, they divide by the number of all neighbors (across relations).
                #       Implement this by summing up these values over all relations.
                #       Also ask Michael, how the results with other normalization constants were.
                num_neighbors_per_node = mask.sum(1, keepdim=True)
                mask /= num_neighbors_per_node + 1e-10  # prevent zero division

                # Non-zero value at (i, j) in mask means that embedding of unique neighbor j contributes 
                # to aggregated embedding for node i (weighted by the value in mask). Formally:
                # aggregated_node_embeddings[i] = \sum_j {mask[i, j] * unique_neighbors_embeddings[j]}
                aggregated_embeddings = mask.mm(unique_neighbors_embeddings)

                # Step 5: Multiply the aggregated feature vectors with the relation-specific weight matrix.
                output_embeddings += aggregated_embeddings.mm(self.relation_weights[relation].t())

            return self.activation(output_embeddings)
        
        
class OneHotEmbedding(nn.Module):
    
    def __init__(self, num_embeddings, cuda=False):
        #super(OneHotEmbedding, self).__init__()
        nn.Module.__init__(self)
        self.num_embeddings = num_embeddings
        self.row = torch.zeros(num_embeddings)
        # TODO: Figure out a way to move row to cuda as soon as cuda is called on the net.
        if cuda:
            self.row = self.row.cuda()
        
    def forward(self, indices):
        self.row.zero_()
        embeddings = self.row.repeat(len(indices), 1)
        embeddings[range(len(indices)), indices] = 1
        return Variable(embeddings, requires_grad=False)