from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import OneHotEmbedding, AdditiveRelationalGraphConvolution, LoopRelationalGraphConvolution, TensorRelationalGraphConvolution


class AbstractGraphAutoEncoder(nn.Module):
    """
    Abstract model for a graph autoencoder for link prediction in a relational graph.

    Uses an encoder to turn nodes into feature vectors (e.g. R-GCN, embeddings)
    and a decoder to turn a triple into a probability score (e.g. DistMult).
    """
    def __init__(self, encoder, decoder, dropout=0):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = nn.Dropout(dropout)

    def forward(self, triples):
        subject_embeddings = self.dropout(self.encoder(triples[:, 0]))
        object_embeddings = self.dropout(self.encoder(triples[:, 1]))
        return self.decoder(subject_embeddings, object_embeddings, triples[:, 2])
    
    
class DistMultDecoder(nn.Module):
    
    def __init__(self, embedding_size, num_relations):
        super(DistMultDecoder, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)
        nn.init.xavier_normal_(self.relation_embedding.weight)
        
    def forward(self, subject_embeddings, object_embeddings, relations):
        relation_embeddings = self.relation_embedding(relations)
        scores = (subject_embeddings * relation_embeddings * object_embeddings).sum(1, keepdim=True)
        return scores
    

class DistMult(AbstractGraphAutoEncoder):
    
    def __init__(self, embedding_size, num_nodes, num_relations, dropout=0):
        entity_embedding = nn.Embedding(num_nodes, embedding_size)
        nn.init.xavier_normal_(entity_embedding.weight)
        decoder = DistMultDecoder(embedding_size, num_relations)
        AbstractGraphAutoEncoder.__init__(self, entity_embedding, decoder, dropout)
        
        
class TransEDecoder(nn.Module):
    
    def __init__(self, embedding_size, num_relations, p_norm=2):
        super(TransEDecoder, self).__init__()
        self.relation_embedding = nn.Embedding(num_relations, embedding_size)
        nn.init.xavier_normal(self.relation_embedding.weight)
        self.dissimilarity = nn.PairwiseDistance(p=p_norm)
        
    def forward(self, subject_embeddings, object_embeddings, relations):
        relation_embeddings = self.relation_embedding(relations)
        scores = self.dissimilarity(subject_embeddings + relation_embeddings, object_embeddings)
        return scores
    
    
class TransE(AbstractGraphAutoEncoder):
    
    def __init__(self, embedding_size, num_nodes, num_relations, dropout=0):
        entity_embedding = nn.Embedding(num_nodes, embedding_size)
        decoder = TransEDecoder(embedding_size, num_relations)
        AbstractGraphAutoEncoder.__init__(self, entity_embedding, decoder, dropout)
        
        
class UnsupervisedRGCN(AbstractGraphAutoEncoder):
    """
    **kwargs: Any additional keyword args, that should be passed on to the graph conv layers.
    """

    def __init__(self, num_nodes, num_relations, train_triples, node_features=None, embedding_size=128,
                 dropout=0, regularization=None, device='cuda', **kwargs):
        
        if node_features is not None:
            # Use a dense embedding matrix initialized from node_features.
            node_features_size = node_features.shape[1]
            node_features_embedding = nn.Embedding(num_nodes, node_features_size)
            node_features_embedding.weight = nn.Parameter(torch.FloatTensor(node_features), requires_grad=False)
            print('Creating model from node_features with', node_features_size, 'features on device', device)
        else:
            # Use a one-hot embedding that is generated on the fly during training.
            # Saves 0.8 GB GPU memory on FB15k-237 without increasing the runtime 
            # (vs using a node_features matrix with one-hot embeddings).
            node_features_size = embedding_size
            node_features_embedding = nn.Embedding(num_nodes, embedding_size)
            nn.init.xavier_uniform_(node_features_embedding.weight)
            #node_features_size = num_nodes
            #node_features_embedding = OneHotEmbedding(num_nodes, device=device)
            print('Creating model with OneHotEmbedding on device', device)

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
        # TODO: Check that weights of graph_conv1 still change, as it is referenced only indirectly here now.
        graph_conv1 = TensorRelationalGraphConvolution(node_features_size, embedding_size, num_nodes, num_relations,
                                                         node_features_embedding, train_triples,
                                                         **kwargs)
        # graph_conv2 = TensorRelationalGraphConvolution(embedding_size, embedding_size, num_nodes, num_relations,
        #                                                graph_conv1, train_triples,
        #                                                **kwargs)

        # self.graph_conv2.name='conv2'
        #self.encoder = self.graph_conv1
        #self.dropout = nn.Dropout(dropout)
        super(UnsupervisedRGCN, self).__init__(graph_conv1, DistMultDecoder(embedding_size, num_relations), dropout)
