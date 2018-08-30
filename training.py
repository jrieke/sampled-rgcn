from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
#from tensorboardX import SummaryWriter

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
        target = torch.ones_like(input1)
        # TODO: Refactored to pytorch v0.4, test and remove old code.
        #target = Variable(torch.ones(input1.shape), requires_grad=False)
        #if input1.is_cuda:
        #    target = target.cuda()
        return super(SimplifiedMarginRankingLoss, self).__call__(input1, input2, target)
    

class TriplesDatasetClassification(TensorDataset):
    # TODO: Extend this to create negative triples on the fly.
    # TODO: Maybe create negative samples in loader instead, then the batch size is known and the labels can be fixed.
    #       However, this would mean that each batch contains a balanced number of positive and negative samples.
    #       Maybe this could even increase accuracy, because the gradient is alway calculated over a positive and the corresponding negative sample.
    #       Also, this would have the advantage that the dataset does not have to be reinitialized each time.
    
    def __init__(self, triples, num_nodes, num_negatives=1):
        triples_and_negatives = np.vstack([triples, sample_negatives(triples, num_nodes, num_negatives)])
        labels = torch.zeros(len(triples_and_negatives), 1)
        labels[:len(triples), 0] = 1
        TensorDataset.__init__(self, torch.from_numpy(triples_and_negatives), labels)
        
        
class TriplesDatasetRanking(TensorDataset):
    # TODO: Extend this to create negative triples on the fly. 
    
    def __init__(self, triples, num_nodes):
        TensorDataset.__init__(self, torch.from_numpy(triples), torch.from_numpy(sample_negatives(triples, num_nodes)))
        
    
    
def train_via_ranking(net, train_triples, val_triples, optimizer, num_nodes, train_ranker, val_ranker, num_epochs, batch_size, batch_size_eval, device, margin=1, history=None, save_best_to=None, dry_run=False, ranking_eval=True):

    #writer = SummaryWriter()

    if history is None:
        history = utils.History()
    loss_function = SimplifiedMarginRankingLoss(margin)
    
    if dry_run:  # use first batch only
        train_triples = train_triples[:batch_size]
        val_triples = val_triples[:batch_size_eval]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # -------------------- Training --------------------
        net.train()
        train_dataset = TriplesDatasetRanking(train_triples, num_nodes)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_tqdm = tqdm(train_loader)
        batches_history = utils.History()

        #running_metrics = collections.defaultdict(lambda: 0)

        for batch, (batch_triples, batch_negative_triples) in enumerate(train_loader_tqdm):

            batch_triples = batch_triples.to(device)
            batch_negative_triples = batch_negative_triples.to(device)

            # Sanity check: Train on 0 inputs.
            #print('WARNING: Sanity check enabled')
            #batch_triples = torch.zeros_like(batch_triples)
            #batch_negative_triples = torch.zeros_like(batch_negative_triples)

            optimizer.zero_grad()
            output = net(batch_triples)
            output_negative = net(batch_negative_triples)
            loss = loss_function(output, output_negative)
            loss.backward()
            optimizer.step()

            batches_history.log_metric('loss', loss.item())
            batches_history.log_metric('acc', (output > output_negative).float().mean().item())
            batches_history.log_metric('mean_diff', (output - output_negative).mean().item())
            batches_history.log_metric('median_diff', (output - output_negative).median().item())

            if batch % 10 == 0:
                train_loader_tqdm.set_postfix(batches_history.latest())

        #for key in running_metrics:
        #    running_metrics[key] /= len(batches)
        
        del batch_triples, batch_negative_triples, output, output_negative, loss
        torch.cuda.empty_cache()


        # -------------------- Testing --------------------
        net.eval()
        with torch.no_grad():
            val_dataset = TriplesDatasetRanking(val_triples, num_nodes)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False)
            val_batches_history = utils.History()

            for batch, (batch_triples, batch_negative_triples) in enumerate(val_loader):

                # TODO: Does it actually make sense to move these to CUDA? They are just used as indices.
                batch_triples = batch_triples.to(device)
                batch_negative_triples = batch_negative_triples.to(device)
                output = net(batch_triples)
                output_negative = net(batch_negative_triples)
                loss = loss_function(output, output_negative)

                # TODO: Especially getting the loss takes quite some time (as much as a single prediction for dist mult), maybe replace it by a running metric directly in torch.
                val_batches_history.log_metric('loss', loss.item())
                val_batches_history.log_metric('acc', (output > output_negative).float().mean().item())
                val_batches_history.log_metric('mean_diff', (output - output_negative).mean().item())
                val_batches_history.log_metric('median_diff', (output - output_negative).median().item())

            del batch_triples, batch_negative_triples, output, output_negative, loss
            torch.cuda.empty_cache()

        #for key in running_metrics:
        #    running_metrics[key] /= len(batches)

        # TODO: Maybe implement these metrics in a batched fashion.
        history.log_metric('loss', batches_history.mean('loss'), 
                           val_batches_history.mean('loss'), 'Loss', print_=True)
        #writer.add_scalar('test/loss', batches_history.mean('loss'), epoch)
        #writer.add_scalar('test/val_loss', val_batches_history.mean('loss'), epoch)
        history.log_metric('acc', batches_history.mean('acc'), 
                           val_batches_history.mean('acc'), 'Accuracy', print_=True)
        history.log_metric('mean_diff', batches_history.mean('mean_diff'), 
                           val_batches_history.mean('mean_diff'), 'Mean Difference', print_=True)
        history.log_metric('median_diff', batches_history.mean('median_diff'), 
                           val_batches_history.mean('median_diff'), 'Median Difference', print_=True)


        # -------------------- Ranking --------------------
        if ranking_eval:
            mean_rank, mean_rec_rank, hits_1, hits_3, hits_10 = train_ranker(net, batch_size=batch_size_eval)
            val_mean_rank, val_mean_rec_rank, val_hits_1, val_hits_3, val_hits_10 = val_ranker(net, batch_size=batch_size_eval)

            history.log_metric('mean_rank', mean_rank, val_mean_rank, 'Mean Rank', print_=True)
            history.log_metric('mean_rec_rank', mean_rec_rank, val_mean_rec_rank, 'Mean Rec Rank', print_=True)
            history.log_metric('hits_1', hits_1, val_hits_1, 'Hits@1', print_=True)
            history.log_metric('hits_3', hits_3, val_hits_3, 'Hits@3', print_=True)
            history.log_metric('hits_10', hits_10, val_hits_10, 'Hits@10', print_=True)


        # -------------------- Saving --------------------
        if save_best_to is not None and (epoch == 0 or history['val_mean_rec_rank'][-1] >= np.max(history['val_mean_rec_rank'][:-1])):
            # TODO: Using save on the model here directly gives an error. 
            torch.save(net.state_dict(), save_best_to)
            print()
            print('Saving model after epoch {} to {}'.format(epoch+1, save_best_to))
            
        print('-'*80)
        print()
        
    return history



def train_via_classification(net, train_triples, val_triples, optimizer, num_nodes, train_ranker, val_ranker, num_epochs, batch_size, batch_size_eval, device, history=None, save_best_to=None, dry_run=False, ranking_eval=True):
    
    if history is None:
        history = utils.History()
    loss_function = nn.BCEWithLogitsLoss()
    
    if dry_run:
        train_triples = train_triples[:batch_size]
        val_triples = val_triples[:batch_size_eval]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # -------------------- Training --------------------
        net.train()
        train_dataset = TriplesDatasetClassification(train_triples, num_nodes)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_tqdm = tqdm(train_loader)
        batches_history = utils.History()

        # running_metrics = collections.defaultdict(lambda: 0)

        for batch, (batch_triples, batch_labels) in enumerate(train_loader_tqdm):

            batch_triples = batch_triples.to(device)
            batch_labels = batch_labels.to(device)

            # Sanity check 1: Train on 0 inputs.
            #train_loader_tqdm.set_description('WARNING: Sanity check enabled')
            #batch_triples = torch.zeros_like(batch_triples)

            # Sanity check 2: Train on 0 targets.
            #train_loader_tqdm.set_description('WARNING: Sanity check enabled')
            #batch_labels = torch.zeros_like(batch_labels)

            # Sanity check 3: Overfit on a single batch.
            #train_loader_tqdm.set_description('WARNING: Sanity check enabled')
            #if epoch == 0 and batch == 0:
            #    fixed_batch_values = batch_triples, batch_labels
            #else:
            #    batch_triples, batch_labels = fixed_batch_values

            # Sanity check 4: Overfit on a few batches.
            # train_loader_tqdm.set_description('WARNING: Sanity check enabled')
            # if epoch == 0:
            #     if batch == 0:
            #         fixed_batch_values = []
            #     if batch < 100:
            #         fixed_batch_values.append((batch_triples, batch_labels))
            #     else:
            #         break
            # else:
            #     if batch < len(fixed_batch_values):
            #         batch_triples, batch_labels = fixed_batch_values[batch]
            #     else:
            #         break


            optimizer.zero_grad()
            output = net(batch_triples)
            loss = loss_function(output, batch_labels)
            loss.backward()
            optimizer.step()

            batches_history.log_metric('loss', loss.item())
            batches_history.log_metric('acc', (torch.sigmoid(output).round() == batch_labels).float().mean().item())
            #batches_history.log_metric('mean_diff', (output - output_negative).mean().item())
            #batches_history.log_metric('median_diff', (output - output_negative).median().item())

            if batch % 10 == 0:
                train_loader_tqdm.set_postfix(batches_history.latest())

        # for key in running_metrics:
        #    running_metrics[key] /= len(batches)

        del batch_triples, batch_labels, output, loss
        torch.cuda.empty_cache()

        # -------------------- Testing --------------------
        net.eval()
        with torch.no_grad():
            val_dataset = TriplesDatasetClassification(val_triples, num_nodes)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False)
            val_batches_history = utils.History()

            for batch, (batch_triples, batch_labels) in enumerate(val_loader):
                # TODO: Does it actually make sense to move these to CUDA? They are just used as indices.
                batch_triples = batch_triples.to(device)
                batch_labels = batch_labels.to(device)
                output = net(batch_triples)
                loss = loss_function(output, batch_labels)

                # TODO: Especially getting the loss takes quite some time (as much as a single prediction for dist mult), maybe replace it by a running metric directly in torch.
                val_batches_history.log_metric('loss', loss.item())
                val_batches_history.log_metric('acc', (torch.sigmoid(output).round() == batch_labels).float().mean().item())
                #val_batches_history.log_metric('mean_diff', (output - output_negative).mean().item())
                #val_batches_history.log_metric('median_diff', (output - output_negative).median().item())

            del batch_triples, batch_labels, output, loss
            torch.cuda.empty_cache()

        # for key in running_metrics:
        #    running_metrics[key] /= len(batches)

        # TODO: Maybe implement these metrics in a batched fashion.
        history.log_metric('loss', batches_history.mean('loss'),
                           val_batches_history.mean('loss'), 'Loss', print_=True)
        # writer.add_scalar('test/loss', batches_history.mean('loss'), epoch)
        # writer.add_scalar('test/val_loss', val_batches_history.mean('loss'), epoch)
        history.log_metric('acc', batches_history.mean('acc'),
                           val_batches_history.mean('acc'), 'Accuracy', print_=True)
        #history.log_metric('mean_diff', batches_history.mean('mean_diff'),
        #                   val_batches_history.mean('mean_diff'), 'Mean Difference', print_=True)
        #history.log_metric('median_diff', batches_history.mean('median_diff'),
        #                   val_batches_history.mean('median_diff'), 'Median Difference', print_=True)

        # -------------------- Ranking --------------------
        if ranking_eval:
            mean_rank, mean_rec_rank, hits_1, hits_3, hits_10 = train_ranker(net, batch_size=batch_size_eval)
            val_mean_rank, val_mean_rec_rank, val_hits_1, val_hits_3, val_hits_10 = val_ranker(net,
                                                                                               batch_size=batch_size_eval)

            history.log_metric('mean_rank', mean_rank, val_mean_rank, 'Mean Rank', print_=True)
            history.log_metric('mean_rec_rank', mean_rec_rank, val_mean_rec_rank, 'Mean Rec Rank', print_=True)
            history.log_metric('hits_1', hits_1, val_hits_1, 'Hits@1', print_=True)
            history.log_metric('hits_3', hits_3, val_hits_3, 'Hits@3', print_=True)
            history.log_metric('hits_10', hits_10, val_hits_10, 'Hits@10', print_=True)

        # -------------------- Saving --------------------
        if save_best_to is not None and (
                epoch == 0 or history['val_mean_rec_rank'][-1] >= np.max(history['val_mean_rec_rank'][:-1])):
            # TODO: Using save on the model here directly gives an error.
            torch.save(net.state_dict(), save_best_to)
            print()
            print('Saving model after epoch {} to {}'.format(epoch + 1, save_best_to))

        print('-' * 80)
        print()

    return history
