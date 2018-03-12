from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


def invert_dict(d):
    return {value: key for key, value in d.iteritems()}
