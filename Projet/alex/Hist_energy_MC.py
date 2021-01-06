# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:19:38 2021

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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

from dscribe.descriptors import SOAP
from ase.build import molecule
from ase import Atoms
from keras.models import load_model

datapath='../../../'
#positions and corresponding energies of a zundel molecule importation
all_positions = pickle.load(open(os.path.join(datapath,'zundel_100K_pos'),'rb'))
all_energies = pickle.load(open(os.path.join(datapath,'zundel_100K_energy'),'rb'))[1:]

Guess_energy=np.load('guess_energy_overtime.npy')
print(np.shape(Guess_energy))
DFT_energy=all_energies[:100000]

plt.hist(DFT_energy,label='DFT')
plt.hist(Guess_energy,label='MC')
plt.savefig('hist_energy')
