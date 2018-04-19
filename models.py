from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import random

from layers import GraphConvolution, RelationalGraphConvolution, OneHotEmbedding


class BaseEmbeddingModel(nn.Module):
    
    def __init__(self, embedding_size, num_nodes, num_relations, decoder=None):
        super(BaseEmbeddingModel, self).__init__()
        self.entity_embedding = nn.Embedding(num_nodes, embedding_size)
        self.decoder = decoder
        # TODO: Do not initalize embeddings via xavier normalization, that doesn't really make sense 
        #       because it depends on the number of embeddings.
        nn.init.xavier_normal(self.entity_embedding.weight)
        
    def forward(self, triples):
        # TODO: Give triples as variables here in the first place.
        subject_tensor = Variable(torch.LongTensor(triples[:, 0]), requires_grad=False)
        if self.entity_embedding.weight.is_cuda:
            subject_tensor = subject_tensor.cuda()
        subject_embeddings = self.entity_embedding(subject_tensor)
            
        object_tensor = Variable(torch.LongTensor(triples[:, 1]), requires_grad=False)
        if self.entity_embedding.weight.is_cuda:
            object_tensor = object_tensor.cuda()
        object_embeddings = self.entity_embedding(object_tensor)
            
        return self.decoder(subject_embeddings, object_embeddings, triples[:, 2])
    
    # TODO: Dirty workaround for ranking evaluation. Handle triples/entities as tensors everywhere, 
    #       than net.get_embeddings can be substituted by net.entitiy_embedding in the ranking objects.
    def get_embeddings(self, entities):
        entities_tensor = Variable(torch.LongTensor(entities), requires_grad=False)
        if self.entity_embedding.weight.is_cuda:
            entities_tensor = entities_tensor.cuda()
            
        return self.entity_embedding(entities_tensor)
    
    
class DistMultDecoder(nn.Module):
    
    def __init__(self, embedding_size, num_relations, dropout=0):
        super(DistMultDecoder, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)
        nn.init.xavier_normal(self.relation_embedding.weight)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, subject_embeddings, object_embeddings, relations):
        # TODO: Give relations as tensor in the first place.
        if type(relations) == torch.LongTensor or type(relations) == torch.cuda.LongTensor:
            relations_tensor = relations
        else:
            relations_tensor = Variable(torch.LongTensor(relations), requires_grad=False)
            if self.relation_embedding.weight.is_cuda:
                relations_tensor = relations_tensor.cuda()
            
        subject_embeddings = self.dropout(subject_embeddings)
        object_embeddings = self.dropout(object_embeddings)
            
        relation_embeddings = self.relation_embedding(relations_tensor)
        scores = (subject_embeddings * relation_embeddings * object_embeddings).sum(1, keepdim=True)
        
        del subject_embeddings, object_embeddings, relation_embeddings
        
        return scores
    

class DistMult(BaseEmbeddingModel):
    
    def __init__(self, embedding_size, num_nodes, num_relations, dropout=0):
        decoder = DistMultDecoder(embedding_size, num_relations, dropout)
        super(DistMult, self).__init__(embedding_size, num_nodes, num_relations, decoder)
        
        
class TransEDecoder(nn.Module):
    
    def __init__(self, embedding_size, num_relations, dropout=0, p_norm=2):
        super(TransEDecoder, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)
        # TODO: Does xavier make sense here?
        nn.init.xavier_normal(self.relation_embedding.weight)
        self.dropout = torch.nn.Dropout(dropout)
        self.dissimilarity = nn.PairwiseDistance(p=p_norm)
        
    def forward(self, subject_embeddings, object_embeddings, relations):
        # TODO: Give relations as tensor in the first place.
        relations_tensor = Variable(torch.LongTensor(relations), requires_grad=False)
        if self.relation_embedding.weight.is_cuda:
            relations_tensor = relations_tensor.cuda()
            
        subject_embeddings = self.dropout(subject_embeddings)
        object_embeddings = self.dropout(object_embeddings)
            
        relation_embeddings = self.relation_embedding(relations_tensor)
        scores = self.dissimilarity(subject_embeddings + relation_embeddings, object_embeddings)
        
        del subject_embeddings, object_embeddings, relation_embeddings
        
        return scores
    
    
class TransE(BaseEmbeddingModel):
    
    def __init__(self, embedding_size, num_nodes, num_relations, dropout=0):
        decoder = TransEDecoder(embedding_size, num_relations, dropout)
        super(TransE, self).__init__(embedding_size, num_nodes, num_relations, decoder)
        
        
class UnsupervisedRGCN(nn.Module):

    # TODO: Add Dropout.
    def __init__(self, num_nodes, num_relations, relational_adj_dict, node_features=None, embedding_size=128, 
                 dist_mult_dropout=0, num_sample_train=10, num_sample_eval=None, activation=F.relu, regularization=None):
        nn.Module.__init__(self)
        
        if node_features is not None:
            # Use a dense embedding matrix initialized from node_features.
            node_features_size = node_features.shape[1]
            node_features_embedding = nn.Embedding(num_nodes, node_features_size)
            node_features_embedding.weight = nn.Parameter(torch.FloatTensor(node_features), requires_grad=False)
            print('Initialized from node_features with', node_features_size, 'features')
        else:
            # Use a one-hot embedding that is generated on the fly during training.
            # Saves 0.8 GB GPU memory on FB15k-237 without increasing the runtime 
            # (vs using a node_features matrix with one-hot embeddings).
            node_features_size = num_nodes
            node_features_embedding = OneHotEmbedding(num_nodes, cuda=True)
            print('Initialized with OneHotEmbedding')
            
        if regularization == 'block':
            # Stack one more linear layer between node features and R-GCN, like in original paper.
            # TODO: Use bias here or not?
            self.additional_layer = nn.Linear(node_features_size, embedding_size)
            node_features_embedding_and_additional_layer = lambda nodes: self.additional_layer(node_features_embedding(nodes))
            
            self.graph_conv1 = RelationalGraphConvolution(embedding_size, embedding_size, num_nodes, num_relations, 
                                                          node_features_embedding_and_additional_layer, relational_adj_dict, 
                                                          num_sample_train, num_sample_eval, activation, regularization)
        else:
            self.graph_conv1 = RelationalGraphConvolution(node_features_size, embedding_size, num_nodes, num_relations, 
                                                          node_features_embedding, relational_adj_dict, 
                                                          num_sample_train, num_sample_eval, activation, regularization)
        self.graph_conv1.name='conv1'
        self.graph_conv2 = RelationalGraphConvolution(embedding_size, embedding_size, num_nodes, num_relations, 
                                                      self.graph_conv1, relational_adj_dict, 
                                                      num_sample_train, num_sample_eval, activation, regularization)
        self.graph_conv2.name='conv2'
        # TODO: Rename to decoder to make it more general.
        self.dist_mult = DistMultDecoder(embedding_size, num_relations, dist_mult_dropout)
        
    def forward(self, triples):
        # TODO: This computes lots of duplicates if nodes appear as subject and object.
        #       As a quick solution, stack subjects and objects, run them through the network together, 
        #       and then separate the embeddings. Check how this changes memory requirements.
#         subjects = triples[:, 0]
#         objects = triples[:, 1]
#         all_embeddings = self.graph_conv2(np.hstack([subjects, objects]), num_sample=num_sample)
#         subject_embeddings = all_embeddings[:len(subjects)]
#         object_embeddings = all_embeddings[len(subjects):]

        
        subject_embeddings = self.graph_conv2(triples[:, 0])  # implicitly calls underlying conv layers
        object_embeddings = self.graph_conv2(triples[:, 1])  # implicitly calls underlying conv layers
        
        scores = self.dist_mult(subject_embeddings, object_embeddings, triples[:, 2])
        del subject_embeddings, object_embeddings
        return scores