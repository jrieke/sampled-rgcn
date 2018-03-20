from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import itertools
from collections import OrderedDict, Callable
import os
from tqdm import tqdm_notebook


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
    
    
# -------------------- Visualization Utils --------------------
    
def plot_matrix(m):
    plt.imshow(m, cmap='Greys', interpolation=None)
    
    
# -------------------- Training Utils --------------------
    
def shuffle_together(arrays, seed=None):
    """Shuffle multiple arrays together so values at the same index stay at the same index."""
    arrays = [np.asanyarray(arr) for arr in arrays]
    
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
    arrays = [np.asanyarray(arr) for arr in arrays]
    
    if shuffle:
        arrays = shuffle_together(arrays, seed=seed)
        
    # TODO: Make sure that all arrays are same size.
    batch_indices_list = np.array_split(np.arange(len(arrays[0])), np.arange(batch_size, len(arrays[0]), batch_size))
    
    if len(arrays) == 1:
        return (arrays[0][batch_indices] for batch_indices in batch_indices_list)
    else:
        return ([arr[batch_indices] for arr in arrays] for batch_indices in batch_indices_list)
    

def predict(net, samples, batch_size=256):
    """Run `net` on `samples` in batches of size `batch_size` (batched version of a normal forward pass)."""
    was_training = net.training
    net.eval()
    
    outputs = []
    for batch_samples in split_into_batches(samples, batch_size):
        outputs.append(net(batch_samples))
        
    if was_training:
        net.train()
        
    return torch.cat(outputs)
    
    
# -------------------- Logging Utils --------------------

class OrderedDefaultDict(OrderedDict):
    """
    Combination of collections.OrderedDict and collections.defaultdict.
    
    From: https://stackoverflow.com/questions/6190331/can-i-do-an-ordered-default-dict-in-python
    """
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))
            
        
class History(OrderedDefaultDict):
    """Class which stores metrics for each batch or epoch."""
    
    def __init__(self):
        super(History, self).__init__(list)
        self.long_names = {}
        
    def log_metric(self, name, value, val_value=None, long_name=None, print_=False):
        self[name].append(value)
        self.long_names[name] = long_name if long_name is not None else name
        if val_value is not None:
            val_name = 'val_' + name
            self[val_name].append(val_value)
            self.long_names[val_name] = 'Val ' + long_name if long_name is not None else val_name
            if print_:
                self.summary(name, val_name)
        elif print_:
                self.summary(name)
                
    def latest(self, name=None):
        if name is not None:
            return self[name][-1]
        else:
            return {name: values[-1] for name, values in self.items()}
    
    def mean(self, name=None):
        if name is not None:
            return np.mean(self[name])
        else:
            return {name: np.mean(values) for name, values in self.items()}
        
    def plot(self, *names, **kwargs):
        plot_val = kwargs.get('plot_val', True)
        figsize = kwargs.get('figsize', None)
        
        default_color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        fig, axes = plt.subplots(len(names), sharex=True, figsize=figsize)

        for name, ax in zip(names, axes):
            plt.sca(ax)
            line, = plt.plot(self[name], next(default_color_cycle))
            plt.ylabel(self.long_names[name])
            if plot_val and 'val_' + name in self:
                plt.plot(self['val_' + name], color=line.get_color(), linestyle='--')

        axes[-1].set_xlabel('Epoch')
        
    # TODO: Implement a new summary method, which prints the metric, its val value, and a new line.
    #       Or use tabulate to print a table with the metrics and one column train and one column val.
    #def new_summary(self):
    #    max_length_names = np.max([len(self.long_names[name]) for name in names])
    #    names_to_print = self.keys()
    #    for name in names_to_print:
    #        print(('{:' + str(max_length_names+4) + '}{}').format(self.long_names[name] + ':', self[name][-1]))
        
        
    def summary(self, *names):
        if not names:
            names = self.keys()
        
        max_length_names = np.max([len(self.long_names[name]) for name in names])
        for name in names:
            print(('{:' + str(max_length_names+4) + '}{}').format(self.long_names[name] + ':', self[name][-1]))
        print()
        
    def save(self, filename):
        df = pd.DataFrame.from_dict(self)
        df = df.rename(columns=lambda name: '{} [{}]'.format(name, self.long_names[name]))
        df.to_csv(filename, index=False, sep=str('\t'))
        
    @staticmethod
    def load(filename):
        history = History()
        df = pd.read_csv(filename, sep='\t')
        for col in df:
            name, long_name = col.split(' ', 1)
            long_name = long_name[1:-1]
            history[name] = list(df[col])
            history.long_names[name] = long_name
        return history
    
    