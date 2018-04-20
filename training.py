from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook

import utils


# TODO: Maybe make sure that corrupted triples are truly negative, ie they do not appear anywhere in the dataset.
#       This is not done in TransE etc, but may improve training.
def sample_negatives(triples, num_nodes, num_negatives=1):
    """Return a copy of triples where either the subject or the object in each triple is replaced with a random entity."""
    corrupted_triples = []
    for i in range(num_negatives):
        for s, o, r in triples:
            if np.random.rand() < 0.5:
                corrupted_triples.append((np.random.randint(num_nodes), o, r))
            else:
                corrupted_triples.append((s, np.random.randint(num_nodes), r))
    return np.asarray(corrupted_triples)


class SimplifiedMarginRankingLoss(nn.MarginRankingLoss):
    """Same as torch.nn.MarginRankingLoss, but input1 is always higher than input2."""
    
    def __call__(self, input1, input2):
        target = Variable(torch.ones(input1.shape), requires_grad=False)
        if input1.is_cuda:
            target = target.cuda()
        return super(SimplifiedMarginRankingLoss, self).__call__(input1, input2, target)
    
    
# TODO: Replace train_forward_kwargs and val_forward_kwargs by num_sample_train and num_sample_eval.
# TODO: Refactor to use TensorDataset. 
# TODO: Add train_ranker/val_ranker.
def train_via_ranking(net, train_triples, val_triples, optimizer, num_nodes, train_ranker, val_ranker, embedding_func, scoring_func, num_epochs, batch_size, margin, history=None, save_best_to=None):
    
    if history is None:
        history = utils.History()
    loss_function = SimplifiedMarginRankingLoss(margin)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        net.train()

        # TODO: Maybe use tensor for batch_triples and wrap everything here in pytorch DataSet/DataLoader.
        batches = utils.split_into_batches(train_triples, batch_size=batch_size, shuffle=True)#[:3]
        batches = tqdm_notebook(batches)
        batches_history = utils.History()

    #     running_metrics = collections.defaultdict(lambda: 0)

        for batch, batch_triples in enumerate(batches):

            optimizer.zero_grad()
            output = net(batch_triples)#, **train_forward_kwargs)
            output_negative = net(sample_negatives(batch_triples, num_nodes))#, **train_forward_kwargs)

            loss = loss_function(output, output_negative)
            loss.backward()
            optimizer.step()

            # TODO: Especially getting the loss takes quite some time (as much as a single prediction for dist mult), maybe replace it by a running metric directly in torch.
            batches_history.log_metric('loss', loss.data[0])
            batches_history.log_metric('acc', (output > output_negative).float().mean().data[0])
            batches_history.log_metric('mean_diff', (output - output_negative).mean().data[0])
            batches_history.log_metric('median_diff', (output - output_negative).median().data[0])
    #         running_metrics['loss'] += loss
    #         running_metrics['acc'] += (output > output_negative).float().mean()
    #         running_metrics['mean_diff'] += (output - output_negative).mean()
    #         running_metrics['median_diff'] += (output - output_negative).median()

            if batch % 10 == 0:
                batches.set_postfix(batches_history.latest())

            # TODO: It seems like this does not free up the GPU memory. Check it again and try to fix.
            del output, output_negative, loss

    #     for key in running_metrics:
    #         running_metrics[key] /= len(batches)


        net.eval()
        # TODO: Set variables to volatile=True when evaluating on the validation set to save GPU memory (instead of or in combination with to_tensor).
        val_output = utils.predict(net, val_triples, batch_size=32, to_tensor=True)
        val_output_negative = utils.predict(net, sample_negatives(val_triples, num_nodes), batch_size=32, to_tensor=True)
        val_loss = loss_function(val_output, val_output_negative)

        # TODO: Maybe implement these metrics in a batched fashion.
        history.log_metric('loss', batches_history.mean('loss'), 
                           val_loss.data[0], 'Loss', print_=True)
        history.log_metric('acc', batches_history.mean('acc'), 
                           (val_output > val_output_negative).float().mean(), 'Accuracy', print_=True)
        history.log_metric('mean_diff', batches_history.mean('mean_diff'), 
                           (val_output - val_output_negative).mean(), 'Mean Difference', print_=True)
        history.log_metric('median_diff', batches_history.mean('median_diff'), 
                           (val_output - val_output_negative).median(), 'Median Difference', print_=True)

        del val_output, val_output_negative, val_loss


        print('Running rank evaluation for train in {} setting...'.format('filtered' if train_ranker.filtered else 'raw'))
        mean_rank, mean_rec_rank, hits_1, hits_3, hits_10 = train_ranker(
            embedding_func, scoring_func)
        print('Running rank evaluation for val in {} setting...'.format('filtered' if val_ranker.filtered else 'raw'))
        val_mean_rank, val_mean_rec_rank, val_hits_1, val_hits_3, val_hits_10 = val_ranker(
            embedding_func, scoring_func)

        history.log_metric('mean_rank', mean_rank, val_mean_rank, 'Mean Rank', print_=True)
        history.log_metric('mean_rec_rank', mean_rec_rank, val_mean_rec_rank, 'Mean Rec Rank', print_=True)
        history.log_metric('hits_1', hits_1, val_hits_1, 'Hits@1', print_=True)
        history.log_metric('hits_3', hits_3, val_hits_3, 'Hits@3', print_=True)
        history.log_metric('hits_10', hits_10, val_hits_10, 'Hits@10', print_=True)
        
        if save_best_to is not None and (epoch == 0 or 
                                         history['val_mean_rec_rank'][-1] >= np.max(history['val_mean_rec_rank'][:-1])):
            # TODO: Using save on the model here directly gives an error. 
            torch.save(net.state_dict(), save_best_to)
            print()
            print('Saving model after epoch {} to {}'.format(epoch+1, save_best_to))
            
        
        print('-'*80)
        print()
        
    return history



class TriplesDataset(TensorDataset):
    # TODO: Extend this to create negative triples on the fly. 
    
    def __init__(self, triples, num_nodes, num_negatives=1):
        triples_and_negatives = np.vstack([triples, sample_negatives(triples, num_nodes, num_negatives)])
        labels = torch.zeros(len(triples_and_negatives), 1)
        labels[:len(triples), 0] = 1
        TensorDataset.__init__(self, torch.from_numpy(triples_and_negatives), labels)
       
    

# TODO: Make validation code work.
# TODO: Add train_ranker/val_ranker.
def train_via_classification(net, train_triples, val_triples, optimizer, num_nodes, train_ranker, val_ranker, embedding_func, scoring_func, num_epochs, batch_size, num_negatives, history=None, save_best_to=None):
    
    if history is None:
        history = utils.History()
    loss_function = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        net.train()

        train_dataset = TriplesDataset(train_triples, num_nodes, num_negatives)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_tqdm = tqdm_notebook(train_loader)
        batches_history = utils.History()

        #running_metrics = collections.defaultdict(lambda: 0)

        for batch, (batch_triples, batch_labels) in enumerate(train_loader_tqdm):

            batch_labels = Variable(batch_labels)

            # TODO: Maybe do force_cpu or cuda as parameter here.
            if torch.cuda.is_available():# and not force_cpu:
                batch_triples = batch_triples.cuda()
                batch_labels = batch_labels.cuda()

            optimizer.zero_grad()
            output = net(batch_triples)
            loss = loss_function(output, batch_labels)
            loss.backward()
            optimizer.step()

            # TODO: Especially getting the loss takes quite some time (as much as a single prediction for dist mult), maybe replace it by a running metric directly in torch.
            batches_history.log_metric('loss', loss.data[0])
            batches_history.log_metric('acc', (F.sigmoid(output).round() == batch_labels).float().mean().data[0])
            #running_metrics['loss'] += loss.data[0]
            #running_metrics['acc'] += (F.sigmoid(output).round() == batch_labels).float().mean().data[0]

            if batch % 10 == 0:
                train_loader_tqdm.set_postfix(batches_history.latest())

        #for key in running_metrics:
        #    running_metrics[key] /= len(batches)


        
        net.eval()
        
        val_dataset = TriplesDataset(val_triples, num_nodes, num_negatives)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        val_batches_history = utils.History()
        
        for batch, (batch_triples, batch_labels) in enumerate(val_loader):

            # TODO: Set all variables to volatile here. This doesn't have an effect yet, because the triples 
            #       are disconnected from the rest of the computation graph.
            batch_labels = Variable(batch_labels, volatile=True)

            # TODO: Maybe do force_cpu or cuda as parameter here.
            if torch.cuda.is_available():# and not force_cpu:
                batch_triples = batch_triples.cuda()
                batch_labels = batch_labels.cuda()

            output = net(batch_triples)
            loss = loss_function(output, batch_labels)

            # TODO: Especially getting the loss takes quite some time (as much as a single prediction for dist mult), maybe replace it by a running metric directly in torch.
            val_batches_history.log_metric('loss', loss.data[0])
            val_batches_history.log_metric('acc', (F.sigmoid(output).round() == batch_labels).float().mean().data[0])
            #running_metrics['loss'] += loss.data[0]
            #running_metrics['acc'] += (F.sigmoid(output).round() == batch_labels).float().mean().data[0]


        #for key in running_metrics:
        #    running_metrics[key] /= len(batches)

        # TODO: Maybe implement these metrics in a batched fashion.
        history.log_metric('loss', batches_history.mean('loss'), 
                           val_batches_history.mean('loss'), 'Loss', print_=True)
        history.log_metric('acc', batches_history.mean('acc'), 
                           val_batches_history.mean('acc'), 'Accuracy', print_=True)



        print('Running rank evaluation for train in {} setting...'.format('filtered' if train_ranker.filtered else 'raw'))
        mean_rank, mean_rec_rank, hits_1, hits_3, hits_10 = train_ranker(embedding_func, scoring_func)
        print('Running rank evaluation for val in {} setting...'.format('filtered' if val_ranker.filtered else 'raw'))
        val_mean_rank, val_mean_rec_rank, val_hits_1, val_hits_3, val_hits_10 = val_ranker(embedding_func, scoring_func)

        history.log_metric('mean_rank', mean_rank, val_mean_rank, 'Mean Rank', print_=True)
        history.log_metric('mean_rec_rank', mean_rec_rank, val_mean_rec_rank, 'Mean Rec Rank', print_=True)
        history.log_metric('hits_1', hits_1, val_hits_1, 'Hits@1', print_=True)
        history.log_metric('hits_3', hits_3, val_hits_3, 'Hits@3', print_=True)
        history.log_metric('hits_10', hits_10, val_hits_10, 'Hits@10', print_=True)
        
        if save_best_to is not None and (epoch == 0 or history['val_mean_rec_rank'][-1] >= np.max(history['val_mean_rec_rank'][:-1])):
            # TODO: Using save on the model here directly gives an error. 
            torch.save(net.state_dict(), save_best_to)
            print()
            print('Saving model after epoch {} to {}'.format(epoch+1, save_best_to))
            
        print('-'*80)
        print()
        
    return history