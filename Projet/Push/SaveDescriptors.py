# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 17:23:32 2020

@author: ewenf
"""

import numpy as np
from dscribe.descriptors import SOAP
from ase.build import molecule
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import keras

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Dropout


print("Import done")

#parameters settings
species = ["H", "O"]
sigma_SOAP = 0.01
periodic = False #pour le Zundel, mais True pour CO2
nmax = 3
lmax = 2
rcut = 6.0
N_atoms = 7

#soap settings
soap = SOAP(
    species=species,
    sigma=sigma_SOAP,
    periodic=periodic,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse=False,
    #rbf='polynomial'
)

print('Soap done')

N_features = soap.get_number_of_features()


#positions and corresponding energies of a zundel molecule importation
positions = pickle.load(open('zundel_100K_pos', 'rb'))[::100]
energies = pickle.load(open('zundel_100K_energy', 'rb'))[::100]

print('Positions shape:', np.shape(positions))

print('Energies shape:', np.shape(energies))


print('Data loaded')

#zundel molecule creation
from ase import Atoms
zundels = np.empty(np.shape(positions)[0], dtype=object) # il est souvent préférable d'utiliser des array
for i_time in range(np.shape(positions)[0]):  # une boucle sur toutes les config possibles
      zundels[i_time] = Atoms(numbers=[8,8,1,1,1,1,1], positions=positions[i_time])

print("Computing descriptors")
# computing descriptors for each positions
descriptors=np.empty([np.shape(positions)[0],N_atoms,N_features])
for i_time in range(np.shape(positions)[0]):
    descriptors[i_time,:,:] = soap.create(zundels[i_time],positions=np.arange(N_atoms))

print("Soap descriptors created")
print(np.shape(descriptors))
