from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn

from layers import RelationalGraphConvolution, BasisRelationalGraphConvolution, BlockRelationalGraphConvolution, OneHotEmbedding, AdditiveRelationalGraphConvolution


class BaseEmbeddingModel(nn.Module):
    
    def __init__(self, embedding_size, num_nodes, decoder=None):
        super(BaseEmbeddingModel, self).__init__()
        self.entity_embedding = nn.Embedding(num_nodes, embedding_size)
        self.decoder = decoder
        nn.init.xavier_normal(self.entity_embedding.weight)
        
    def forward(self, triples):
        subject_embeddings = self.entity_embedding(triples[:, 0])
        object_embeddings = self.entity_embedding(triples[:, 1])
        return self.decoder(subject_embeddings, object_embeddings, triples[:, 2])
    
    
class DistMultDecoder(nn.Module):
    
    def __init__(self, embedding_size, num_relations, dropout=0):
        super(DistMultDecoder, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)
        nn.init.xavier_normal(self.relation_embedding.weight)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, subject_embeddings, object_embeddings, relations):
        subject_embeddings = self.dropout(subject_embeddings)
        object_embeddings = self.dropout(object_embeddings)
        relation_embeddings = self.relation_embedding(relations)
        scores = (subject_embeddings * relation_embeddings * object_embeddings).sum(1, keepdim=True)
        return scores
    

class DistMult(BaseEmbeddingModel):
    
    def __init__(self, embedding_size, num_nodes, num_relations, dropout=0):
        decoder = DistMultDecoder(embedding_size, num_relations, dropout)
        super(DistMult, self).__init__(embedding_size, num_nodes, num_relations, decoder)
        
        
class TransEDecoder(nn.Module):
    
    def __init__(self, embedding_size, num_relations, dropout=0, p_norm=2):
        super(TransEDecoder, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)
        nn.init.xavier_normal(self.relation_embedding.weight)
        self.dropout = torch.nn.Dropout(dropout)
        self.dissimilarity = nn.PairwiseDistance(p=p_norm)
        
    def forward(self, subject_embeddings, object_embeddings, relations):
        subject_embeddings = self.dropout(subject_embeddings)
        object_embeddings = self.dropout(object_embeddings)
        relation_embeddings = self.relation_embedding(relations)#_tensor)
        scores = self.dissimilarity(subject_embeddings + relation_embeddings, object_embeddings)
        return scores
    
    
class TransE(BaseEmbeddingModel):
    
    def __init__(self, embedding_size, num_nodes, num_relations, dropout=0):
        decoder = TransEDecoder(embedding_size, num_relations, dropout)
        super(TransE, self).__init__(embedding_size, num_nodes, num_relations, decoder)
        
        
class UnsupervisedRGCN(nn.Module):
    """
    **kwargs: Any additional keyword args, that should be passed on to the graph conv layers.
    """

    def __init__(self, num_nodes, num_relations, relational_adj_dict, train_triples, node_features=None, embedding_size=128,
                 dist_mult_dropout=0, regularization=None, **kwargs):
        nn.Module.__init__(self)
        
        if node_features is not None:
            # Use a dense embedding matrix initialized from node_features.
            node_features_size = node_features.shape[1]
            node_features_embedding = nn.Embedding(num_nodes, node_features_size)
            node_features_embedding.weight = nn.Parameter(torch.FloatTensor(node_features), requires_grad=False)
            print('Initialized model from node_features with', node_features_size, 'features')
        else:
            # Use a one-hot embedding that is generated on the fly during training.
            # Saves 0.8 GB GPU memory on FB15k-237 without increasing the runtime 
            # (vs using a node_features matrix with one-hot embeddings).
            node_features_size = num_nodes
            node_features_embedding = OneHotEmbedding(num_nodes, cuda=True)
            print('Initialized model with OneHotEmbedding')

        # if regularization is None:
        #     RGC = RelationalGraphConvolution
        # elif regularization == 'basis':
        #     RGC = BasisRelationalGraphConvolution
        # elif regularization == 'block':
        #     RGC = BlockRelationalGraphConvolution
        # else:
        #     raise ValueError("regularization must be one of None, 'basis' or 'block'")
        #
        # if regularization == 'block':
        #     # Stack one more linear layer between node features and R-GCN, like in original paper.
        #     # TODO: Use bias here or not?
        #     self.additional_layer = nn.Linear(node_features_size, embedding_size)
        #     node_features_embedding_and_additional_layer = lambda nodes: self.additional_layer(node_features_embedding(nodes))
        #
        #
        #     self.graph_conv1 = RGC(embedding_size, embedding_size, num_nodes, num_relations,
        #                                                   node_features_embedding_and_additional_layer, relational_adj_dict,
        #                                                   **kwargs)
        # else:
        #     self.graph_conv1 = RGC(node_features_size, embedding_size, num_nodes, num_relations,
        #                                                   node_features_embedding, relational_adj_dict,
        #                                                   **kwargs)
        # self.graph_conv1.name='conv1'
        self.graph_conv1 = AdditiveRelationalGraphConvolution(node_features_size, embedding_size, num_nodes, num_relations,
                                                              node_features_embedding, relational_adj_dict, train_triples,
                                                              **kwargs)
        # self.graph_conv2 = RGC(embedding_size, embedding_size, num_nodes, num_relations,
        #                                               self.graph_conv1, relational_adj_dict,
        #                                               **kwargs)
        # self.graph_conv2.name='conv2'
        self.decoder = DistMultDecoder(embedding_size, num_relations, dist_mult_dropout)
        
    def forward(self, triples):
        # TODO: This computes lots of duplicates if nodes appear as subject and object.
        #       As a quick solution, stack subjects and objects, run them through the network together, 
        #       and then separate the embeddings. Check how this changes memory requirements.
        subject_embeddings = self.graph_conv1(triples[:, 0])  # implicitly calls underlying conv layers
        object_embeddings = self.graph_conv1(triples[:, 1])  # implicitly calls underlying conv layers
        scores = self.decoder(subject_embeddings, object_embeddings, triples[:, 2])
        return scores