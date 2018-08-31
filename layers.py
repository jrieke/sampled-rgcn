from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from block import block


def block_diagonal(blocks):
    """Return a block-diagonal matrix constructed from the matrices in blocks."""
    # TODO: Check if it's faster to construct a list of lists here, or do this in pytorch directly.
    #block_matrix = np.zeros((len(blocks), len(blocks)), dtype='object')
    #block_matrix[np.diag_indices_from(block_matrix)] = blocks
    #return block(block_matrix.tolist())

    # Inspired by scipy.linalg.block_diag
    shapes = [a.shape for a in blocks]
    # TODO: Check if backward goes through this and blocks actually change.
    out = Variable(torch.zeros(np.sum(shapes, axis=0).tolist()))

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = blocks[i]
        r += rr
        c += cc
    return out


# TODO: Everything from here until AdditiveRelationalGraphConvolution is the (broken) old model. Replace it as soon as posible.

# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features, num_nodes, in_features_func, adj_dict,
#                  num_sample_train=10, num_sample_eval=None, activation=F.relu):
#         nn.Module.__init__(self)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.in_features_func = in_features_func
#         self.adj_dict = adj_dict
#         self.num_sample_train = num_sample_train
#         self.num_sample_eval = num_sample_eval
#         # TODO: Add adj_array (see RelationalGraphConvolution).
#         self.activation = activation
#
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform(self.weight)
#
#
#     # TODO: Refactor this.
#     # TODO: Profile this with lprun and see how much time each step takes up (especially creating the tensors!).
#     def aggregate(self, nodes, adj_dict, include_self=False):
#         """Return a vector for each node in `nodes` by mean-aggregating feature vectors from its neighborhood (or a sample thereof)."""
#
#         num_sample = self.num_sample_train if self.training else self.num_sample_eval
#
#         # Step 1: For each node, sample some neighbors that we aggregate information from.
#
#         # TODO: Move this up to constructor or even outside of this class.
#         #adj_array = np.array([adj_dict[i] for i in range(num_nodes)])
#         #sampled_neighbors_per_node = adj_array[nodes]
#         #print(sampled_neighbors_per_node)
#         sampled_neighbors_per_node = [adj_dict[node] for node in nodes]
#
#         # TODO: Replace stuff below by this line, check if it works.
#         #sampled_neighbors_per_node = [adj_lists[node].update([node]) if include_self else adj_lists[node] for node in nodes]
#
#         # TODO: Check if this improves the network.
#         #       Also, in graphsage, this is done after the sampling step (which doesn't make much sense though).
#         # TODO: Check that node is not added to adj_lists permanently.
#         if include_self:
#             for sampled_neighbors, node in zip(sampled_neighbors_per_node, nodes):
#                 sampled_neighbors.add(node)
#
#         if num_sample is not None:
#             # TODO: See if local pointers bring speed improvements (see GraphSage code).
#             for i in range(len(sampled_neighbors_per_node)):
#                 if num_sample <= len(sampled_neighbors_per_node[i]):
#                     sampled_neighbors_per_node[i] = set(random.sample(sampled_neighbors_per_node[i], num_sample))
#
#
#         # Step 2: Find the unique neighbors in all sampled neighbors.
#
#         unique_neighbors = list(set.union(*sampled_neighbors_per_node))
#         unique_neighbors_to_index = {neighbor: i for i, neighbor in enumerate(unique_neighbors)}
#
#         if unique_neighbors:
#
#             # Step 3: Get embeddings for these unique neighbors from the underlying layer
#             #         (e.g. another GraphConvolution layer or a fixed feature matrix).
#
#             unique_neighbors_tensor = torch.LongTensor(unique_neighbors)
#             if self.is_cuda():
#                 unique_neighbors_tensor = unique_neighbors_tensor.cuda()
#             unique_neighbors_embeddings = self.in_features_func(unique_neighbors_tensor)
#
#
#             # Step 4: For each input node, sum the embeddings of its (sampled) neighbors,
#             #         using the embeddings obtained above. The algorithm here uses a masking matrix
#             #         to find the neighbor embeddings for each node and add them in one step.
#
#             # TODO: See if this can be implemented in an easier way (i.e. without mask).
#
#             # TODO: Check if this really needs to be a variable.
#             # TODO: Maybe store mask in self and fill it with zeros here, so it doesn't have to initalized in each forward pass.
#             mask = Variable(torch.zeros(len(nodes), len(unique_neighbors)), requires_grad=False)
#             if self.is_cuda():
#                 mask = mask.cuda()
#             # TODO: Understand and rename variables here.
#             column_indices = [unique_neighbors_to_index[neighbor] for sampled_neighbors in sampled_neighbors_per_node for neighbor in sampled_neighbors]
#             row_indices = [i for i in range(len(sampled_neighbors_per_node)) for j in range(len(sampled_neighbors_per_node[i]))]
#             mask[row_indices, column_indices] = 1
#
#             # TODO: Is this actually the number of neighbors, i.e. does it correspond
#             #       to the normalization constant in RGCN paper?
#             num_neighbors = mask.sum(1, keepdim=True)
#             mask /= num_neighbors + 1e-10  # prevent zero division
#
#             #set_trace()
#
#             # TODO: Check what the actual output of this term is.
#             return mask.mm(unique_neighbors_embeddings)
#         else:
#             # TODO: If there are no neighbors, this currently returns a zero vector. Is this correct?
#             # TODO: Building this variable and moving it to cuda takes up a lot of time (15 % for Link Prediction),
#             #       speed this up somehow.
#             zeros = Variable(torch.zeros(len(nodes), self.in_features), requires_grad=False)
#             if self.is_cuda():
#                 zeros = zeros.cuda()
#             return zeros
#
#
#     def forward(self, nodes):
#         neighborhood_features = self.aggregate(nodes, self.adj_dict, include_self=True)
#
#         # TODO: Maybe add features of nodes themselves,
#         #       or include nodes themselves in sampling process in aggregate function (as in GraphSAGE,
#         #       but collides with RGCN use of aggregate function).
#         #node_features = self.in_features_func(torch.LongTensor(nodes))
#
#         return self.activation(neighborhood_features.mm(self.weight.t()))
#
#     def is_cuda(self):
#         return self.weight.is_cuda
#
#
# class AbstractRelationalGraphConvolution(GraphConvolution):
#     def __init__(self, size_in, size_out, num_nodes, num_relations, in_features_func, relational_adj_dict,
#                  num_sample_train=10, num_sample_eval=None, activation=F.relu, verbose=False, device='cuda'):
#
#         # TODO: Maybe remove inheritance from GraphConvolution.
#         GraphConvolution.__init__(self, size_in, size_out, num_nodes, in_features_func, None,
#                                   num_sample_train, num_sample_eval, activation)
#
#         # TODO: Maybe create adj_dict from train_triples directly in here.
#         self.relational_adj_dict = relational_adj_dict
#         self.adj_array = np.array([set(relational_adj_dict[i].keys()) for i in range(num_nodes)], dtype=object)
#         self.num_relations = num_relations
#         self.verbose = verbose
#         self.zero = torch.zeros(1).to(device)
#         # TODO: Figure out a way to move zero to cuda as soon as cuda is called on the net.
#
#     def get_relation_weight(self, relation):
#         raise NotImplementedError()
#
#     def get_relation_weights(self, relations=None):
#         # Default implementation, can be overwritten by subclasses for efficiency.
#         if relations is None:
#             relations = range(self.num_relations)
#         return torch.cat([self.get_relation_weight(relation)[None] for relation in relations])
#
#     def sample_neighbors(self, nodes):
#         sampled_neighbors_per_node = self.adj_array[nodes]
#         num_sample = self.num_sample_train if self.training else self.num_sample_eval
#         if num_sample is not None:
#             # TODO: Check if local pointers bring speed improvements (see GraphSage code).
#             for i in range(len(sampled_neighbors_per_node)):
#                 if num_sample <= len(sampled_neighbors_per_node[i]):
#                     # TODO: Check if np.random.choice is faster.
#                     sampled_neighbors_per_node[i] = set(random.sample(sampled_neighbors_per_node[i], num_sample))
#
#         if self.verbose: print('Sampled', num_sample, 'neighbors per node:', sampled_neighbors_per_node)
#         return sampled_neighbors_per_node
#
#
#
#     def forward_bmm(self, nodes):
#
#         # Step 1: Get embeddings for all nodes in the mini-batch from the underlying layer.
#         #         (e.g. another GraphConvolution layer or a fixed feature matrix).
#         input_embeddings = self.in_features_func(nodes)
#         if self.verbose: print('Got input_embeddings:', input_embeddings)
#
#         # TODO: Maybe add hyperparameter that determines which fraction of the self-embedding to use,
#         #       and which fraction of the aggregated embedding. See if this has an effect
#         #       or if weights find the best way to combine the two embeddings themselves.
#         #       Alternatively, include self-embedding in sampling.
#         # TODO: Maybe add dropout of the self-connection.
#         output_embeddings = input_embeddings.mm(self.weight.t())  # terms for relations will be added to this
#         if self.verbose: print('Multiplying them by weight matrix, preliminary output_embeddings are:', output_embeddings)
#
#         # Step 2: For each node, sample some neighbors that we aggregate information from.
#         sampled_neighbors_per_node = self.sample_neighbors(nodes)
#
#         # TODO: Do all of this in pytorch.
#         #       Store a dict of tensors triples_per_node. Then use
#         #       torch.IntTensor(num_sample).random(len(triples_per_node[node])) to sample some triples.
#         #       Then use torch.unique() (only pytorch 0.4!) to find unique values from this matrix.
#
#
#         # Step 3: Find the unique neighbors in all sampled neighbors.
#         #         If there are no unique neighbors, return zero embeddings.
#         unique_neighbors = list(set.union(*sampled_neighbors_per_node))
#         unique_neighbors_to_index = {neighbor: i for i, neighbor in enumerate(unique_neighbors)}
#         if self.verbose: print('Unique neighbors are:', unique_neighbors)
#
#         if not unique_neighbors:
#             # TODO: Maybe create and store this once, and return it each time here.
#             if self.verbose: print('No unique neighbors found, returning 0.')
#             return self.zero.repeat(len(nodes), self.out_features)
#
#         # Step 3: Get embeddings for these unique neighbors from the underlying layer
#         #         (e.g. another GraphConvolution layer or a fixed feature matrix).
#         unique_neighbors = nodes.new_tensor(unique_neighbors)
#         unique_neighbors_embeddings = self.in_features_func(unique_neighbors)
#         if self.verbose: print('Got input embeddings for unique neighbors:', unique_neighbors_embeddings)
#
#         #print(np.sum([len(neighbors) for neighbors in sampled_neighbors_per_node]) / len(unique_neighbors_embeddings) )
#
#         # Step 4: For each relation, average the embeddings of sampled neighbors,
#         #         using the embeddings obtained above. The algorithm here uses a masking matrix
#         #         to find the neighbor embeddings for each node and add them in one step.
#
#
#         # This is the solution with a 3D tensor using all relations at once.
#         masks = self.zero.repeat(self.num_relations, len(nodes), len(unique_neighbors))
#         #masks = masks.clone()
#         #masks = Variable(torch.zeros(self.num_relations, len(nodes), len(unique_neighbors)))
#         #if self.is_cuda():
#         #    masks = masks.cuda()
#
#
#         # TODO: Check that mask is initialized to 0 each time.
#         #mask = Variable(self.zero.repeat(len(nodes), len(unique_neighbors)), requires_grad=False)
#         #masks = collections.defaultdict(mask.clone)
#         for i, (node, sampled_neighbors) in enumerate(zip(nodes, sampled_neighbors_per_node)):
#             for neighbor in sampled_neighbors:
#                 for relation in self.relational_adj_dict[node][neighbor]:
#                     masks[relation, i, unique_neighbors_to_index[neighbor]] = 1
#
#         for i in range(len(masks)):
#             # TODO: Do this for the whole tensor at once.
#             num_neighbors_per_node = masks[i].sum(1, keepdim=True)
#             masks[i] /= num_neighbors_per_node + 1e-10  # prevent zero division
#
#         #print([(k, v.sum()) for k, v in masks.items()])
#         #print(masks.items())
#         aggregated_embeddings_per_relation = masks.bmm(unique_neighbors_embeddings.expand(len(masks), -1, -1))
#         # TODO: Check if this works with mat.
#         # TODO: Speed up with addbmm.
#         output_embeddings += aggregated_embeddings_per_relation.bmm(self.get_relation_weights().transpose(1, 2)).sum(0)
#
#         output_embeddings = self.activation(output_embeddings)
#         if self.verbose: print('Applied non-linearity, final output embeddings are:', output_embeddings)
#         if self.verbose: print('-'*80)
#         return output_embeddings
#
#
#     # loop
#     def forward(self, nodes):
#
#         # Step 1: Get embeddings for all nodes in the mini-batch from the underlying layer.
#         #         (e.g. another GraphConvolution layer or a fixed feature matrix).
#         input_embeddings = self.in_features_func(nodes)
#
#         # TODO: Maybe add hyperparameter that determines which fraction of the self-embedding to use,
#         #       and which fraction of the aggregated embedding. See if this has an effect
#         #       or if weights find the best way to combine the two embeddings themselves.
#         #       Alternatively, include self-embedding in sampling.
#         # TODO: Maybe add dropout of the self-connection.
#         output_embeddings = input_embeddings.mm(self.weight.t())  # terms for relations will be added to this
#
#         # Step 2: For each node, sample some neighbors that we aggregate information from.
#         sampled_neighbors_per_node = self.sample_neighbors(nodes)
#
#         # TODO: Do all of this in pytorch.
#         #       Store a dict of tensors triples_per_node. Then use
#         #       torch.IntTensor(num_sample).random(len(triples_per_node[node])) to sample some triples.
#         #       Then use torch.unique() (only pytorch 0.4!) to find unique values from this matrix.
#
#         # Step 3: Find the unique neighbors in all sampled neighbors.
#         #         If there are no unique neighbors, return zero embeddings.
#         unique_neighbors = list(set.union(*sampled_neighbors_per_node))
#         unique_neighbors_to_index = {neighbor: i for i, neighbor in enumerate(unique_neighbors)}
#         if self.verbose: print('Unique neighbors are:', unique_neighbors)
#
#         if not unique_neighbors:
#             # TODO: Maybe create and store this once, and return it each time here.
#             if self.verbose: print('No unique neighbors found, returning 0.')
#             return self.zero.repeat(len(nodes), self.out_features)
#
#         # Step 3: Get embeddings for these unique neighbors from the underlying layer
#         #         (e.g. another GraphConvolution layer or a fixed feature matrix).
#         unique_neighbors = nodes.new_tensor(unique_neighbors)
#         unique_neighbors_embeddings = self.in_features_func(unique_neighbors)
#         if self.verbose: print('Got input embeddings for unique neighbors:', unique_neighbors_embeddings)
#
#
#         # Step 4: For each relation, average the embeddings of sampled neighbors,
#         #         using the embeddings obtained above. The algorithm here uses a masking matrix
#         #         to find the neighbor embeddings for each node and add them in one step.
#         mask = self.zero.repeat(len(nodes), len(unique_neighbors))
#         masks = collections.defaultdict(mask.clone)
#
#         for i, (node, sampled_neighbors) in enumerate(zip(nodes, sampled_neighbors_per_node)):
#             for neighbor in sampled_neighbors:
#                 for relation in self.relational_adj_dict[node][neighbor]:
#                     masks[relation][i, unique_neighbors_to_index[neighbor]] = 1
#
#
#         if self.verbose: print('Iterating over all relations and summing neighbor embeddings for each relation.')
#
#         for relation, mask in masks.iteritems():
#             # TODO: Check which of these three alternatives for masking is quickest:
#             #       1) Iterate over all nodes and sampled neighbor, and assign column/row indices only
#             #          if edge has relation (current implementation).
#             #          Potentially iterates very often over the same arrays.
#             #       2) Compile one object sampled_neighbors_per_node_per_relation beforehand,
#             #          and do the usual mask-building for each relation in there.
#             #       3) Build one mask for each relation and set values in the mask belonging
#             #          to the relation directly.
#
#
#             if self.verbose: print('Relation', relation, 'out of', self.num_relations)
#
#             if self.verbose: print('Mask:', mask)
#
#             # TODO: Here, I divide by the number of neighbors with this relation.
#             #       In R-GCN paper, they divide by the number of all neighbors (across relations).
#             #       Implement this by summing up these values over all relations.
#             #       Also ask Michael, how the results with other normalization constants were.
#             num_neighbors_per_node = mask.sum(1, keepdim=True)
#             mask /= num_neighbors_per_node + 1e-10  # prevent zero division
#
#             # Non-zero value at (i, j) in mask means that embedding of unique neighbor j contributes
#             # to aggregated embedding for node i (weighted by the value in mask). Formally:
#             # aggregated_node_embeddings[i] = \sum_j {mask[i, j] * unique_neighbors_embeddings[j]}
#             aggregated_embeddings = mask.mm(unique_neighbors_embeddings)
#             if self.verbose: print('Aggregated neighbor embeddings for each node:', aggregated_embeddings)
#
#             # Step 5: Multiply the aggregated feature vectors with the relation-specific weight matrix.
#             output_embeddings += aggregated_embeddings.mm(self.get_relation_weight(relation).t())
#             if self.verbose: print('Multiplying them by relation weight matrix, preliminary output_embeddings are:', output_embeddings)
#
#         output_embeddings = self.activation(output_embeddings)
#         if self.verbose: print('Applied non-linearity, final output embeddings are:', output_embeddings)
#         if self.verbose: print('-'*80)
#         return output_embeddings
#
#
# class RelationalGraphConvolution(AbstractRelationalGraphConvolution):
#
#     def __init__(self, size_in, size_out, num_nodes, num_relations, in_features_func, relational_adj_dict,
#                  num_sample_train=10, num_sample_eval=None, activation=F.relu, verbose=False, cuda=True):
#         AbstractRelationalGraphConvolution.__init__(self, size_in, size_out, num_nodes, num_relations,
#                                                     in_features_func, relational_adj_dict,
#                                                     num_sample_train, num_sample_eval, activation, verbose, cuda)
#
#         # Storing the relation weights in a ParameterList instead of a 3D tensor is more memory-efficient,
#         # because the gradient is only computed for this specific weight then.
#         # self.relation_weights = nn.ParameterList()
#         # for relation in range(num_relations):
#         #    weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         #    nn.init.xavier_uniform(weight)
#         #    self.relation_weights.append(weight)
#         self.relation_weights = nn.Parameter(torch.FloatTensor(num_relations, size_out, size_in))
#         nn.init.xavier_uniform_(self.relation_weights)
#
#     def get_relation_weight(self, relation):
#         return self.relation_weights[relation]
#
#     def get_relation_weights(self, relations=None):
#         if relations is None:
#             return self.relation_weights
#         else:
#             return self.relation_weights[relations]
#
#
# class BasisRelationalGraphConvolution(AbstractRelationalGraphConvolution):
#
#     def __init__(self, size_in, size_out, num_nodes, num_relations, in_features_func, relational_adj_dict,
#                  num_sample_train=10, num_sample_eval=None, activation=F.relu, verbose=False, cuda=True,
#                  num_basis_weights=2):
#         AbstractRelationalGraphConvolution.__init__(self, size_in, size_out, num_nodes, num_relations,
#                                                     in_features_func, relational_adj_dict,
#                                                     num_sample_train, num_sample_eval, activation, verbose, cuda)
#
#         self.basis_weights = nn.Parameter(torch.FloatTensor(num_basis_weights, size_out, size_in))
#         nn.init.xavier_uniform_(self.basis_weights)
#         self.coefficients = nn.Parameter(torch.FloatTensor(num_relations, num_basis_weights))
#         nn.init.xavier_uniform_(self.coefficients)
#
#     def get_relation_weight(self, relation):
#         # TODO: Or could I calculate this once above and reuse it?
#         # print(relation)
#         return (self.coefficients[relation][:, None, None] * self.basis_weights).sum(0)
#
#     def get_relation_weights(self, relations=None):
#         if relations is None:
#             return (self.coefficients[:, :, None, None] * self.basis_weights[None]).sum(1)
#         else:
#             return (self.coefficients[relations][:, :, None, None] * self.basis_weights[None]).sum(1)
#
#
# class BlockRelationalGraphConvolution(AbstractRelationalGraphConvolution):
#
#     def __init__(self, size_in, size_out, num_nodes, num_relations, in_features_func, relational_adj_dict,
#                  num_sample_train=10, num_sample_eval=None, activation=F.relu, verbose=False,
#                  cuda=True,
#                  num_blocks=10):
#         AbstractRelationalGraphConvolution.__init__(self, size_in, size_out, num_nodes, num_relations,
#                                                     in_features_func, relational_adj_dict,
#                                                     num_sample_train, num_sample_eval, activation, verbose, cuda)
#
#         block_width = size_in / num_blocks
#         block_height = size_out / num_blocks
#         if not block_width.is_integer() or not block_height.is_integer():
#             raise ValueError('in_features and out_features has to be divisible by num_blocks')
#
#         if verbose: print('Constructing blocks of size', block_width, ' - ', block_height)
#         self.blocks_per_relation = nn.ParameterList()
#         for relation in range(num_relations):
#             blocks = nn.Parameter(torch.FloatTensor(num_blocks, int(block_width), int(block_height)))
#             nn.init.xavier_uniform(blocks)
#             self.blocks_per_relation.append(blocks)
#
#         self.zero_weight = nn.Parameter(torch.zeros(size_in, size_out), requires_grad=False)
#
#     def get_relation_weight(self, relation):
#         # Inspired by scipy.linalg.block_diag
#         blocks = self.blocks_per_relation[relation]
#         shapes = [a.shape for a in blocks]
#         out = self.zero.repeat(*np.sum(shapes, axis=0).tolist())
#         r, c = 0, 0
#         for i, (rr, cc) in enumerate(shapes):
#             out[r:r + rr, c:c + cc] += blocks[i]
#             r += rr
#             c += cc
#         return self.zero_weight


# New model.




class AdditiveRelationalGraphConvolution(nn.Module):
    def __init__(self, size_in, size_out, num_nodes, num_relations, in_features_func, relational_adj_dict, train_triples,
                 num_sample_train=10, num_sample_eval=None, activation=F.relu, verbose=False):
        nn.Module.__init__(self)

        self.activation = activation
        self.num_sample_eval = num_sample_eval
        self.num_sample_train = num_sample_train
        #self.relational_adj_dict = relational_adj_dict
        #self.adj_array = np.array([relational_adj_dict[i].keys() + [i] for i in range(num_nodes)], dtype=object)
        self.num_relations = num_relations
        self.verbose = verbose
        # TODO: Rename this probably.
        self.in_features_func = in_features_func

        self.weight = nn.Parameter(torch.FloatTensor(size_out, size_in))
        nn.init.xavier_uniform_(self.weight)
        self.relation_embedding = nn.Embedding(num_relations + 1, size_out)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        # TODO: If I use triples_per_node like this, adj_array and relational_adj_dict could actually be removed.
        # TODO: Implement self-connections as triples here or via sampling below?
        # Add triples for self-connections (these will be used in the sampling process as well).
        # Their "relation" index is equal to self.num_relations.
        self_connection_triples = np.zeros((num_nodes, 3))
        self_connection_triples[:, 0] = np.arange(num_nodes)
        self_connection_triples[:, 1] = np.arange(num_nodes)
        self_connection_triples[:, 2] = self.num_relations
        triples = np.concatenate([train_triples, self_connection_triples])

        # For each node, get the triples that contain this node as a subject (as a numpy array).
        # Store these triple arrays in an array of objects, so we can retrieve them faster during sampling.
        self.triples_per_node = np.array([triples[triples[:, 0] == i] for i in range(num_nodes)], dtype=object)


    def get_device(self):
        return self.weight.device


    def sample_edges(self, nodes):
        """
        Sample a fixed number of edges for each node in `nodes`. Return the neighboring nodes and the relations.

        This is based on the training triples, so it doesn't see the full graph. If a node has fewer edges than
        the number of edges to sample, repeat them.
        """
        num_sample = self.num_sample_train if self.training else self.num_sample_eval

        sampled_neighbors_per_node = np.zeros((len(nodes), num_sample), dtype=int)
        sampled_relations_per_node = np.zeros((len(nodes), num_sample), dtype=int)

        for i, node in enumerate(nodes):
            sampled_triples = self.triples_per_node[node][np.random.randint(len(self.triples_per_node[node]), size=num_sample)]
            sampled_neighbors_per_node[i] = sampled_triples[:, 1]
            sampled_relations_per_node[i] = sampled_triples[:, 2]

        # TODO: Maybe do the sampling step directly in torch/on the GPU.
        #       If it's done on the GPU, self.triples_per_node would need to be on the GPU as well,
        #       which might be memory-intensive.
        sampled_neighbors_per_node, sampled_relations_per_node = torch.from_numpy(sampled_neighbors_per_node).to(self.get_device()), torch.from_numpy(sampled_relations_per_node).to(self.get_device())
        return sampled_neighbors_per_node, sampled_relations_per_node


    def forward(self, nodes):

        # Step 1: Sample a fixed number of edges for each node.
        sampled_neighbors_per_node, sampled_relations_per_node = self.sample_edges(nodes)

        # TODO: Investigate which effect the explicit self-embedding has.
        #input_embeddings = self.in_features_func(nodes).mm(self.weight.t())

        # TODO: Investigate again if selecting the embeddings from the weight matrix directly gives a performance advantage.
        #       Also look in my notes!
        #if self.in_features_func is None:
        #    neighbor_embeddings = self.weight[:, sampled_neighbors_per_node.flatten()]
        #else:
        #    neighbor_embeddings = self.in_features_func(torch.from_numpy(sampled_neighbors_per_node.flatten()))

        # Step 2: For each node, get the node embeddings of its sampled neighbors (from the previous layer)
        #         and average over them.
        neighbor_embeddings = self.in_features_func(sampled_neighbors_per_node.view(-1))
        neighbor_embeddings = neighbor_embeddings.view(sampled_neighbors_per_node.shape[0], sampled_neighbors_per_node.shape[1], -1)  # shape: batch_size, num_sample, size_in
        aggregated_neighbor_embeddings = neighbor_embeddings.mean(1)  # shape: batch_size, size_in

        # Step 3: For each node, get the relation embeddings of its sampled relations (explicit relation embedding in this layer)
        #         and average over them.
        relation_embeddings = self.relation_embedding(sampled_relations_per_node.view(-1))
        relation_embeddings = relation_embeddings.view(sampled_relations_per_node.shape[0], sampled_relations_per_node.shape[1], -1)  # shape: batch_size, num_sample, size_out
        aggregated_relation_embeddings = relation_embeddings.mean(1)  # shape: batch_size, size_out

        # TODO: Could do the last step more efficiently by using an EmbeddingBag layer like below.
        #       Decreases runtime from 5:30 min to 5 min on CPU. Investigate thoroughly if this works, then maybe use it.
        # self.relation_embedding_mean = nn.EmbeddingBag(num_relations + 1, size_out, mode='mean')
        # nn.init.xavier_uniform_(self.relation_embedding_mean.weight)
        # aggregated_relation_embeddings = self.relation_embedding_mean(sampled_relations_per_node)

        # Step 4: Multiply the (aggregated) neighbor embeddings with the weight matrix (this converts
        # the embedding size of the previous layer to the embedding size of this layer), add the relation embeddings
        # and apply the activation function.
        output_embeddings = aggregated_neighbor_embeddings.mm(self.weight.t()) + aggregated_relation_embeddings# + input_embeddings  # shape: batch_size, size_out
        return self.activation(output_embeddings)



class LoopRelationalGraphConvolution(nn.Module):
    def __init__(self, size_in, size_out, num_nodes, num_relations, in_features_func, relational_adj_dict, train_triples,
                 num_sample_train=10, num_sample_eval=None, activation=F.relu, verbose=False):
        nn.Module.__init__(self)

        self.activation = activation
        self.num_sample_eval = num_sample_eval
        self.num_sample_train = num_sample_train
        #self.relational_adj_dict = relational_adj_dict
        #self.adj_array = np.array([relational_adj_dict[i].keys() + [i] for i in range(num_nodes)], dtype=object)
        self.num_relations = num_relations
        self.verbose = verbose
        # TODO: Rename this probably.
        self.in_features_func = in_features_func

        self.relation_weights = nn.ParameterList()
        for relation in range(num_relations+1):
            weight = nn.Parameter(torch.FloatTensor(size_out, size_in))
            nn.init.xavier_uniform_(weight)
            self.relation_weights.append(weight)
        # TODO: Is it more memory-efficient to use parameter list here or a 3D tensor like below.
        #self.relation_weights = nn.Parameter(torch.FloatTensor(num_relations+1, size_out, size_in))  # last row is for self-connections, will be accessed via -1
        #nn.init.xavier_uniform_(self.relation_weights)

        # TODO: If I use triples_per_node like this, adj_array and relational_adj_dict could actually be removed.
        # TODO: Implement self-connections as triples here or via sampling below?
        # Add triples for self-connections (these will be sampled just like normal triples below).
        self_connection_triples = np.zeros((num_nodes, 3))
        self_connection_triples[:, 0] = np.arange(num_nodes)
        self_connection_triples[:, 1] = np.arange(num_nodes)
        self_connection_triples[:, 2] = self.num_relations
        triples = np.concatenate([train_triples, self_connection_triples])

        # For each node, get the triples that contain this node as a subject (as a numpy array).
        # Store these triple arrays in an array of objects, so we can retrieve them faster during sampling.
        self.triples_per_node = np.array([triples[triples[:, 0] == i] for i in range(num_nodes)], dtype=object)


    def get_device(self):
        return self.relation_weights[0].device


    def sample_edges(self, nodes):
        """
        Sample a fixed number of edges for each node in `nodes`. Return the neighboring nodes and the relations.

        This is based on the training triples, so it doesn't see the full graph. If a node has fewer edges than
        the number of edges to sample, repeat them.
        """
        num_sample = self.num_sample_train if self.training else self.num_sample_eval

        sampled_neighbors_per_node = np.zeros((len(nodes), num_sample), dtype=int)
        sampled_relations_per_node = np.zeros((len(nodes), num_sample), dtype=int)

        for i, node in enumerate(nodes):
            sampled_triples = self.triples_per_node[node][np.random.randint(len(self.triples_per_node[node]), size=num_sample)]
            sampled_neighbors_per_node[i] = sampled_triples[:, 1]
            sampled_relations_per_node[i] = sampled_triples[:, 2]

        # TODO: Maybe do the sampling step directly in torch/on the GPU.
        #       If it's done on the GPU, self.triples_per_node would need to be on the GPU as well,
        #       which might be memory-intensive.
        sampled_neighbors_per_node, sampled_relations_per_node = torch.from_numpy(sampled_neighbors_per_node).to(self.get_device()), torch.from_numpy(sampled_relations_per_node).to(self.get_device())
        return sampled_neighbors_per_node, sampled_relations_per_node

    def forward(self, nodes):

        # Step 1: Sample a fixed number of edges for each node in the batch.
        sampled_neighbors_per_node, sampled_relations_per_node = self.sample_edges(nodes)

        # Step 2: Get the node embeddings for all sampled neighbors.
        neighbor_embeddings = self.in_features_func(sampled_neighbors_per_node.view(-1))
        neighbor_embeddings = neighbor_embeddings.view(sampled_neighbors_per_node.shape[0],
                                                       sampled_neighbors_per_node.shape[1], -1)  # shape: batch_size, num_sample, size_in

        for relation in range(self.num_relations+1):  # +1 is for self-connections

            # Step 3: For each relation, take the mean of all neighbor embeddings that are connected via this relation.
            # TODO: Should this handle triples where the node is subject OR object?
            mask = (sampled_relations_per_node == relation).float().unsqueeze(-1)  # shape: batch_size, num_sample, 1
            aggregated_neighbor_embeddings = (mask * neighbor_embeddings).mean(1)  # shape: batch_size, size_in

            # TODO: Maybe find a more elegant solution to add to the output_embeddings.
            # Step 4: Multiply the aggregated neighbor embeddings with the relation-specific weight.
            if relation == 0:
                output_embeddings = aggregated_neighbor_embeddings.mm(self.relation_weights[relation].t())  # shape: batch_size, size_out
            else:
                output_embeddings += aggregated_neighbor_embeddings.mm(self.relation_weights[relation].t())

        # Step 5: Divide by the number of relations (+1 for self-connections) and apply activation function.
        # TODO: Quick fix, see tensor model.
        return self.activation(output_embeddings / (self.num_relations+1) * 1e3)



class TensorRelationalGraphConvolution(nn.Module):
    def __init__(self, size_in, size_out, num_nodes, num_relations, in_features_func, relational_adj_dict, train_triples,
                 num_sample_train=10, num_sample_eval=None, activation=F.relu, verbose=False):
        nn.Module.__init__(self)

        self.activation = activation
        self.num_sample_eval = num_sample_eval
        self.num_sample_train = num_sample_train
        #self.relational_adj_dict = relational_adj_dict
        #self.adj_array = np.array([relational_adj_dict[i].keys() + [i] for i in range(num_nodes)], dtype=object)
        self.num_relations = num_relations
        self.verbose = verbose
        # TODO: Rename this probably.
        self.in_features_func = in_features_func

        self.relation_weights = nn.Parameter(torch.FloatTensor(num_relations+1, size_out, size_in))  # +1 for self-connections
        # TODO: Is there any difference between these two?
        #nn.init.xavier_uniform_(self.relation_weights)
        for i in range(len(self.relation_weights)):
            nn.init.xavier_uniform_(self.relation_weights[i])


        # TODO: If I use triples_per_node like this, adj_array and relational_adj_dict could actually be removed.
        # TODO: Implement self-connections as triples here or via sampling below?
        # Add triples for self-connections (these will be sampled just like normal triples below).
        self_connection_triples = np.zeros((num_nodes, 3))
        self_connection_triples[:, 0] = np.arange(num_nodes)
        self_connection_triples[:, 1] = np.arange(num_nodes)
        self_connection_triples[:, 2] = self.num_relations
        triples = np.concatenate([train_triples, self_connection_triples])

        # For each node, get the triples that contain this node as a subject (as a numpy array).
        # Store these triple arrays in an array of objects, so we can retrieve them faster during sampling.
        self.triples_per_node = np.array([triples[triples[:, 0] == i] for i in range(num_nodes)], dtype=object)


    def get_device(self):
        return self.relation_weights.device


    def sample_edges(self, nodes):
        """
        Sample a fixed number of edges for each node in `nodes`. Return the neighboring nodes and the relations.

        This is based on the training triples, so it doesn't see the full graph. If a node has fewer edges than
        the number of edges to sample, repeat them.
        """
        num_sample = self.num_sample_train if self.training else self.num_sample_eval

        # TODO: Investigate how much time the sampling step takes and if there is any way to decrease this.
        # TODO: Maybe create torch tensor on cuda here directly. Check how this affects runtime.
        sampled_neighbors_per_node = np.zeros((len(nodes), num_sample), dtype=int)
        sampled_relations_per_node = np.zeros((len(nodes), num_sample), dtype=int)

        for i, node in enumerate(nodes):
            sampled_triples = self.triples_per_node[node][np.random.randint(len(self.triples_per_node[node]), size=num_sample)]
            sampled_neighbors_per_node[i] = sampled_triples[:, 1]
            sampled_relations_per_node[i] = sampled_triples[:, 2]

        # TODO: Maybe do the sampling step directly in torch/on the GPU.
        #       If it's done on the GPU, self.triples_per_node would need to be on the GPU as well,
        #       which might be memory-intensive.
        sampled_neighbors_per_node, sampled_relations_per_node = torch.from_numpy(sampled_neighbors_per_node).to(self.get_device()), torch.from_numpy(sampled_relations_per_node).to(self.get_device())
        return sampled_neighbors_per_node, sampled_relations_per_node

    def forward(self, nodes):

        # Step 1: Sample a fixed number of edges for each node.
        sampled_neighbors_per_node, sampled_relations_per_node = self.sample_edges(nodes)

        # Step 2: Get the node embeddings for all sampled neighbors.
        neighbor_embeddings = self.in_features_func(sampled_neighbors_per_node.view(-1))
        neighbor_embeddings = neighbor_embeddings.view(sampled_neighbors_per_node.shape[0],
                                                       sampled_neighbors_per_node.shape[1], -1)  # shape: batch_size, num_sample, size_in

        # Step 3: Convert relations to a one-hot matrix.
        # TODO: Maybe find a way to create this tensor only once and reuse it here.
        sampled_relations_per_node_onehot = torch.zeros(sampled_relations_per_node.shape[0],
                                                        sampled_relations_per_node.shape[1], self.num_relations+1,
                                                        device=sampled_relations_per_node.device)  # shape: batch_size, num_sample, num_relations+1
        # TODO: Maybe make this sparse.
        sampled_relations_per_node_onehot.scatter_(2, sampled_relations_per_node.unsqueeze(-1), 1)

        # Step 4: Repeat the neighbor embeddings for each relation and elementwise-multiply with the onehot matrix.
        expanded_neighbor_embeddings = neighbor_embeddings.unsqueeze(2).expand(-1, -1, self.num_relations+1, -1)  # shape: batch_size, num_sample, num_relations+1, size_in
        aggregated_neighbor_embeddings = (expanded_neighbor_embeddings
                                          * sampled_relations_per_node_onehot.unsqueeze(-1)).mean(1)  # shape: batch_size, num_relations+1, size_in

        transposed_aggregated_neighbor_embeddings = aggregated_neighbor_embeddings.transpose(0, 1)  # shape: num_relations+1, batch_size, size_in
        # TODO: Maybe use addbmm instead of bmm(...).mean(0)
        output_embeddings = torch.bmm(transposed_aggregated_neighbor_embeddings, self.relation_weights.transpose(1, 2)).mean(0)  # shape: batch_size, size_out

        # TODO: Dirty hack, think about a good solution to make the output values higher here.
        #       Maybe using relu instead of elu could actually fix this already.
        return self.activation(output_embeddings*1e3)



class OneHotEmbedding(nn.Module):
    
    def __init__(self, num_embeddings, device='cuda'):
        nn.Module.__init__(self)
        self.num_embeddings = num_embeddings
        self.row = torch.zeros(num_embeddings).to(device)
        # TODO: Figure out a way to move row to cuda as soon as cuda is called on the net.
        
    def forward(self, indices):
        self.row.zero_()
        embeddings = self.row.repeat(len(indices), 1)
        embeddings[range(len(indices)), indices] = 1
        return Variable(embeddings)


# TODO: Is this still used?
class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def __init__(self, sparse):
        super(SparseMM, self).__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input