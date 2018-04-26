from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import pandas as pd
import collections

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from PIL import Image  # install pillow-simd instead of the default pillow version for a ~10 % speedup
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

import utils


def load_graph(root_dir):
    print('Loading graph triples from:', root_dir)
    train_df = pd.read_csv(os.path.join(root_dir, 'train.txt'), sep='\t', names=['subject', 'relation', 'object'])
    val_df = pd.read_csv(os.path.join(root_dir, 'valid.txt'), sep='\t', names=['subject', 'relation', 'object'])
    test_df = pd.read_csv(os.path.join(root_dir, 'test.txt'), sep='\t', names=['subject', 'relation', 'object'])
    print('Found', len(train_df), 'triples for train,', len(val_df), 'triples for val,', len(test_df),
          'triples for test')

    entity_map = utils.IndexMap(train_df[['subject', 'object']],
                                val_df[['subject', 'object']],
                                test_df[['subject', 'object']])
    relation_map = utils.IndexMap(train_df['relation'],
                                  val_df['relation'],
                                  test_df['relation'])
    num_nodes = len(entity_map)
    num_relations = len(relation_map)
    print('Found', num_nodes, 'nodes and', num_relations, 'relation types')

    def to_triples_array(df):
        subjects = df['subject'].map(entity_map.to_index)
        objects = df['object'].map(entity_map.to_index)
        relations = df['relation'].map(relation_map.to_index)
        return np.vstack([subjects, objects, relations]).T

    train_triples = to_triples_array(train_df)
    val_triples = to_triples_array(val_df)
    test_triples = to_triples_array(test_df)

    # Required for filtered ranking evaluation to check if corrupted triples appear anywhere in the dataset.
    all_triples = np.vstack([train_triples, val_triples, test_triples])

    print()
    return num_nodes, num_relations, train_triples, val_triples, test_triples, all_triples, entity_map, relation_map


def load_image_features(num_nodes, entity_map):
    # TODO: Only supported for Freebase, implement this for other graphs as well.
    print('Loading image features from: ../data/fb15k-237-onoro-rubio/feature-vectors-vgg19_bn.npz')
    feature_tensors_per_mid = np.load('../data/fb15k-237-onoro-rubio/feature-vectors-vgg19_bn.npz')
    node_features = np.zeros((num_nodes, len(feature_tensors_per_mid['m.0gcrg'][0])))

    for i in range(num_nodes):
        mid = entity_map.from_index(i)[1:].replace('/', '.')
        if mid in feature_tensors_per_mid:
            # TODO: Use image 0 here?
            node_features[i] = feature_tensors_per_mid[mid][0]
        else:
            # TODO: Exclude mids without images from the graph (i.e. from entity_map and triples).
            print('No image for mid:', mid)

    print()
    return node_features


def get_adj_dict(triples, undirected=False):
    adj_dict = collections.defaultdict(set)
    for s, o, r in triples:  # only use training set, so the network does not implicilty see the whole graph
        adj_dict[s].add(o)
        adj_dict[o].add(s)  # use non-directed edges
    return adj_dict


def get_relational_adj_dict(triples, undirected=False):
    relational_adj_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, o, r in triples:  # only use training set, so the network does not implicilty see the whole graph
        relational_adj_dict[s][o].append(r)
        # TODO: Try using undirected edges here as well and see if it improves the results.
    return relational_adj_dict


# TODO: Test this.
def compute_image_features(root_dir, image_model=None, show_progress=True):
    """
    Compute image features for all images in subdirs of root_dir`, using a convolutional neural network.`

    Arguments:
        root_dir (string): Images should be contained in subdirs of this dir.
        image_model (torch.nn.Module): The model used to compute image features (e.g. one of the models in
                                       torchvision.models). Last layer will be cut off. If None (default),
                                       torchvision.models.vgg19_bn will be used.
        show_progress (bool): Show a progress bar while computing the features.

    Returns:
        Dictionary where keys are the subdirs of `root_dir` and items are numpy arrays with features of all images
        in that subdir.
    """
    if image_model is None:
        image_model = torchvision.models.vgg19_bn(pretrained=True)

    # Remove last fully-connected layer, move to GPU and set to eval.
    image_model.classifier = nn.Sequential(*list(image_model.classifier.children())[:-1])
    if torch.cuda.is_available():
        image_model.cuda()
    image_model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feature_tensors_per_dir = {}
    dirs = filter(lambda dir_: os.path.isdir(os.path.join(root_dir, dir_)), os.listdir(root_dir))
    if show_progress:
        dirs = tqdm(dirs)

    for dir_ in dirs:
        img_tensors = []
        for filename in os.listdir(os.path.join(root_dir, dir_)):
            try:
                img = Image.open(os.path.join(root_dir, dir_, filename))
            except IOError:
                print('Could not read image:', os.path.join(root_dir, dir_, filename))
            else:
                img_tensors.append(transform(img)[None])
        img_tensors = Variable(torch.cat(img_tensors), volatile=True)
        if torch.cuda.is_available():
            img_tensors = img_tensors.cuda()
        feature_tensors = image_model(img_tensors)
        feature_tensors_per_dir[dir_] = feature_tensors.data.cpu().numpy()

    return feature_tensors_per_dir
