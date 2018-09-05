from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import itertools
from collections import OrderedDict, Callable
import collections
import os
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

import torch
import random
import datetime
import json


# -------------------- Data Utils --------------------

def invert_dict(d):
    return {value: key for key, value in d.iteritems()}


class IndexMap():
    """
    Map objects to unique integer indices and back.
    
    Args:
        arrays: One or more arrays that contain the objects (in any order and with duplicates).
        start_index (int): Start the integer indices here.
    """
    
    def __init__(self, *arrays, **kwargs):
        self.start_index = kwargs.get('start_index', 0)
        unique_values = np.unique(np.concatenate([np.unique(array) for array in arrays]))
        self._index_to_value = dict(enumerate(unique_values, self.start_index))
        self._value_to_index = invert_dict(self._index_to_value)
        
    def from_index(self, index):
        return self._index_to_value[index]
    
    def to_index(self, value):
        return self._value_to_index[value]
    
    def __len__(self):
        return len(self._index_to_value)
  
    def __repr__(self):
        return 'IndexMap ({} unique objects, starting at {})'.format(len(self), self.start_index)
    
    
def train_val_test_split(*arrays, **kwargs):
    """Like sklearn.model_selection.train_test_split, but split into three subsets."""
    if 'train_size' in kwargs:
        raise ValueError('Please specify val_size and test_size, do not use train_size')
    val_size = kwargs.pop('val_size', None)
    test_size = kwargs.pop('test_size', None)
    shuffle = kwargs.pop('shuffle', True)
    random_state = kwargs.pop('random_state', None)
    
    train_val, test = train_test_split(*arrays, test_size=test_size, random_state=random_state, shuffle=shuffle)
    train, val = train_test_split(train_val, test_size=val_size, shuffle=False)
    return train, val, test
    
    
# -------------------- Visualization Utils --------------------
    
def plot_matrix(m):
    plt.imshow(m, cmap='Greys', interpolation=None)
    
    
# -------------------- Training Utils --------------------
    
def shuffle_together(arrays, seed=None):
    """Shuffle multiple arrays together so values at the same index stay at the same index."""
    #arrays = [np.asanyarray(arr) for arr in arrays]
    
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(arrays[0]))
    
    return [arr[indices] for arr in arrays]


def split_into_batches(*arrays, **kwargs):
    """
    Split multiple arrays into batches.
    
    Args:
        arrays: The arrays which to split. One sample consists of one value from each array.
        batch_size (int): The number of samples in each batch (default: 128).
        shuffle (boolean): Whether to shuffle the samples before returning (default: False).
        seed (int): The seed for shuffling (default: None).
        
    Returns:
        A generator over batches, where each batch consists of subsets of all arrays.
    """
    batch_size = kwargs.get('batch_size', 128)
    shuffle = kwargs.get('shuffle', False)
    seed = kwargs.get('seed', None)
    #arrays = [np.asanyarray(arr) for arr in arrays]
    
    if shuffle:
        arrays = shuffle_together(arrays, seed=seed)
        
    # TODO: Make sure that all arrays are same size.
    batch_indices_list = np.array_split(np.arange(len(arrays[0])), np.arange(batch_size, len(arrays[0]), batch_size))
    
    if len(arrays) == 1:
        return [arrays[0][batch_indices] for batch_indices in batch_indices_list]
    else:
        return [[arr[batch_indices] for arr in arrays] for batch_indices in batch_indices_list]


#def predict_loader(net, loader, show_progress=False, ):

    

def predict(net, *arrays, **kwargs):
    """Run `net` on `samples` in batches of size `batch_size` (batched version of a normal forward pass)."""
    batch_size = kwargs.get('batch_size', 128)
    show_progress = kwargs.get('show_progress', False)
    # TODO: Maybe change to argument 'keep_variable' or 'retain_variable', which is False by default.
    to_tensor = kwargs.get('to_tensor', False)
    #move_to_cuda = kwargs.get('move_to_cuda', False)
    forward_kwargs = kwargs.get('forward_kwargs', None)
    
    if forward_kwargs is None:
        forward_kwargs = {}
    
    #was_training = net.training
    #net.eval()
    
    num_batches = int(np.ceil(len(arrays[0]) / batch_size))
    batches = split_into_batches(*arrays, batch_size=batch_size, shuffle=False)
    if show_progress:
        batches = tqdm_notebook(batches, total=num_batches)
        
    # TODO: Only works for single output. Make it work for multiple outputs as well.
    outputs = []
    for batch_arrays in batches:
        if len(arrays) == 1:
            output = net(batch_arrays, **forward_kwargs)
        else:
            output = net(*batch_arrays, **forward_kwargs)
        if to_tensor:
            #output_cuda = output
            output = output.data
            #del output.cuda
        outputs.append(output)
        
    #if was_training:
    #    net.train()
        
    return torch.cat(outputs)
    
    
def seed_all(seed=None):
    """Set seed for random, numpy.random, torch and torch.cuda."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        torch.manual_seed(np.random.randint(1e6))
        torch.cuda.manual_seed(np.random.randint(1e6))
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    
# -------------------- Logging Utils --------------------

# class OrderedDefaultDict(OrderedDict):
#     """
#     Combination of collections.OrderedDict and collections.defaultdict.
#
#     From: https://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python
#     """
#     # Source: http://stackoverflow.com/a/6190500/562769
#     def __init__(self, default_factory=None, *a, **kw):
#         if (default_factory is not None and
#            not isinstance(default_factory, Callable)):
#             raise TypeError('first argument must be callable')
#         OrderedDict.__init__(self, *a, **kw)
#         self.default_factory = default_factory
#
#     def __getitem__(self, key):
#         try:
#             return OrderedDict.__getitem__(self, key)
#         except KeyError:
#             return self.__missing__(key)
#
#     def __missing__(self, key):
#         if self.default_factory is None:
#             raise KeyError(key)
#         self[key] = value = self.default_factory()
#         return value
#
#     def __reduce__(self):
#         if self.default_factory is None:
#             args = tuple()
#         else:
#             args = self.default_factory,
#         return type(self), args, None, None, self.items()
#
#     def copy(self):
#         return self.__copy__()
#
#     def __copy__(self):
#         return type(self)(self.default_factory, self)
#
#     def __deepcopy__(self, memo):
#         import copy
#         return type(self)(self.default_factory,
#                           copy.deepcopy(self.items()))
#
#     def __repr__(self):
#         return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
#                                                OrderedDict.__repr__(self))
            
        
# class OldHistory(OrderedDefaultDict):
#     """Class which stores metrics for each batch or epoch."""
#
#     def __init__(self, description=''):
#         super(History, self).__init__(list)
#         self.long_names = {}
#         self.timestamp = str(datetime.datetime.now())
#         self.description = description
#         # TODO: Add parameters field.
#
#     def log_metric(self, name, value, val_value=None, long_name=None, print_=False):
#         self[name].append(value)
#         self.long_names[name] = long_name if long_name is not None else name
#         if val_value is not None:
#             val_name = 'val_' + name
#             self[val_name].append(val_value)
#             self.long_names[val_name] = 'Val ' + long_name if long_name is not None else val_name
#             if print_:
#                 self.summary(name, val_name)
#         elif print_:
#                 self.summary(name)
#
#     def latest(self, name=None):
#         if name is not None:
#             return self[name][-1]
#         else:
#             return {name: values[-1] for name, values in self.items()}
#
#     def mean(self, name=None):
#         if name is not None:
#             return np.mean(self[name])
#         else:
#             return {name: np.mean(values) for name, values in self.items()}
#
#     # TODO: Make this method both a class method and static (to compare multiple logs).
#     def plot(self, *names, **kwargs):
#         plot_val = kwargs.get('plot_val', True)
#         figsize = kwargs.get('figsize', None)
#         compare = kwargs.get('compare', None)
#         xlim = kwargs.get('xlim')
#
#         if compare is None:
#             compare = []
#         else:
#             for i in range(len(compare)):
#                 try:
#                     compare[i] = History.load(compare[i])
#                 except:
#                     pass
#
#         fig, axes = plt.subplots(len(names), sharex=True, figsize=figsize)
#
#         for name, ax in zip(names, axes):
#             plt.sca(ax)
#             plt.grid()
#             plt.ylabel(self.long_names[name])
#             if xlim:
#                 plt.xlim(*xlim)
#
#             color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])  # default mpl colors
#             for h in [self] + compare:
#                 color = next(color_cycle)
#                 if name in h:
#                     plt.plot(h[name], color=color)
#                 if plot_val and 'val_' + name in h:
#                     plt.plot(h['val_' + name], color=color, linestyle='--')
#
#         axes[-1].set_xlabel('Epoch')
#
#     # TODO: Implement a new summary method, which prints the metric, its val value, and a new line.
#     #       Or use tabulate to print a table with the metrics and one column train and one column val.
#     #def new_summary(self):
#     #    max_length_names = np.max([len(self.long_names[name]) for name in names])
#     #    names_to_print = self.keys()
#     #    for name in names_to_print:
#     #        print(('{:' + str(max_length_names+4) + '}{}').format(self.long_names[name] + ':', self[name][-1]))
#
#
#     # TODO: Maybe print timestamp, description, parameters here.
#     def summary(self, *names):
#         if not names:
#             names = self.keys()
#
#         max_length_names = np.max([len(self.long_names[name]) for name in names])
#         for name in names:
#             print(('{:' + str(max_length_names+4) + '}{}').format(self.long_names[name] + ':', self[name][-1]))
#         print()
#
#     # TODO: Save and load as json (via json.dumps(self.__dict__)).
#     def save(self, filename):
#         with open(filename, 'w') as outfile:
#             json.dump(self.__dict__, outfile)
#         #df = pd.DataFrame.from_dict(self)
#         #df = df.rename(columns=lambda name: '{} [{}]'.format(name, self.long_names[name]))
#         #df.to_csv(filename, index=False, sep=str('\t'))
#
#     @staticmethod
#     def load(filename):
#         history = History()
#         df = pd.read_csv(filename, sep='\t')
#         for col in df:
#             name, long_name = col.split(' ', 1)
#             long_name = long_name[1:-1]
#             history[name] = list(df[col])
#             history.long_names[name] = long_name
#         return history


class History(object):
    """Class which stores metrics for each batch or epoch."""

    def __init__(self, description=''):
        super(History, self).__init__()
        self.values = collections.defaultdict(list)
        self.timestamp = str(datetime.datetime.now())
        self.description = description
        # TODO: Add parameters field.

    def __repr__(self):
        return 'History from {} with metrics {}'.format(self.timestamp, self.values.keys())

    def log(self, name, value, val_value=None, print_=False):
        self.values[name].append(value)
        if val_value is not None:
            val_name = 'val_' + name
            self.values[val_name].append(val_value)
            if print_:
                print(name + '    :', value)
                print(val_name + ':', val_value)
        elif print_:
            print(name + ':', value)

    def last(self, name=None):
        if name is not None:
            return self.values[name][-1]
        else:
            return {name: values[-1] for name, values in self.values.items()}

    def mean(self, name=None):
        if name is not None:
            return np.mean(self.values[name])
        else:
            return {name: np.mean(values) for name, values in self.values.items()}

    def plot(self, names=None, plot_val=True, figsize=None, xlim=None, compare_to=None):
        if names is None:
            names = self.values.keys()

        if compare_to is None:
            compare_to = []
        else:
            # If one of the elements in compare_to is a filename, load it as a History object.
            for i in range(len(compare_to)):
                try:
                    compare_to[i] = History.load(compare_to[i])
                except:
                    pass

        fig, axes = plt.subplots(len(names), sharex=True, figsize=figsize)

        for name, ax in zip(names, axes):
            plt.sca(ax)
            plt.grid()
            plt.ylabel(name)
            if xlim:
                plt.xlim(*xlim)

            color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])  # default mpl colors
            for h in [self] + compare_to:
                color = next(color_cycle)
                if name in h.values:
                    plt.plot(h.values[name], color=color)
                if plot_val and 'val_' + name in h.values:
                    plt.plot(h.values['val_' + name], color=color, linestyle='--')

        axes[-1].set_xlabel('Epoch')

    @staticmethod
    def plot_from_file(filenames, names=None, plot_val=True, figsize=None, xlim=None):
        History.load(filenames[0]).plot(names=names, plot_val=plot_val, figsize=figsize, xlim=xlim,
                                        compare_to=filenames[1:])

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def load(filename):
        h = History()
        with open('test.json') as f:
            contents = json.load(f)
        h.timestamp = contents['timestamp']
        h.description = contents['description']
        h.values.update(contents['values'])
        return h
