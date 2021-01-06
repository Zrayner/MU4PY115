# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:14:51 2021

@author: ewenf
"""

from __future__ import print_function
import tensorflow as tf
import keras,sklearn
import matplotlib.pyplot as plt
import pickle
import numpy as np


# suppress tensorflow compilation warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed=0
np.random.seed(seed) # fix random seed

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

from dscribe.descriptors import SOAP
from ase.build import molecule
from ase import Atoms
from ase.io import write
from keras.models import load_model

Zundel_NN = load_model('Fitted_Zundel_NN.h5')

tf.keras.utils.plot_model(
    Zundel_NN,
    to_file="Zundel_model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)