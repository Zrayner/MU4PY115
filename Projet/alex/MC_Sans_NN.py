# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 02:07:32 2021

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
Zundel_NN = load_model('Fitted_Zundel_NN.h5')

datapath='../../../'
#positions and corresponding energies of a zundel molecule importation
all_positions = pickle.load(open(os.path.join(datapath,'zundel_100K_pos'),'rb'))
all_energies = pickle.load(open(os.path.join(datapath,'zundel_100K_energy'),'rb'))

positions = all_positions[::10]
energies = all_energies[1::10]

#parameters settings
species = ["H","O"]
sigma_SOAP = 0.7
periodic = False
nmax = 3
lmax = 4
rcut = 11.0

#soap settings
soap = SOAP(
    species=species,
    sigma=sigma_SOAP,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse=False,
)

n_configs = np.shape(positions)[0]
n_features = soap.get_number_of_features()
n_elements = 2
n_oxygens = 2
n_hydrogens = 5
n_atoms = n_hydrogens + n_oxygens



def get_energy(positions):

    zundel = Atoms(numbers=[8,8,1,1,1,1,1], positions=positions)
    descriptors = soap.create(zundel,positions=np.arange(n_atoms),n_jobs=4)   
    descriptors[n_oxygens:,:] = scaler_H_1.transform(descriptors[n_oxygens:,:])
    for i_hydrogens in range(n_hydrogens):
        descriptors[n_oxygens+i_hydrogens,:] = pca_hydrogens.transform(descriptors[n_oxygens+i_hydrogens,:].reshape(1,-1))
    descriptors[n_oxygens:,:pca_treshold] = scaler_H_2.transform(descriptors[n_oxygens:,:pca_treshold])
    
    descriptors[:n_oxygens,:] = scaler_O_1.transform(descriptors[:n_oxygens,:])
    for i_oxygens in range(n_oxygens):
        descriptors[i_oxygens,:] = pca_oxygens.transform(descriptors[i_oxygens,:].reshape(1,-1))
    descriptors[:n_oxygens,:pca_treshold] =scaler_O_2.transform(descriptors[:n_oxygens,:pca_treshold])
   
    desc = np.ones([1,pca_treshold])
    descriptors_nn =[]
    for i_atom in range(n_atoms):
        desc[:,:] = descriptors[i_atom,:pca_treshold]
        descriptors_nn.append(np.int_(desc))
    
    return energies_scaler.inverse_transform(Zundel_NN.predict(descriptors_nn))[0][0]


T = 100
k = 1.380649e-23
beta = 1/(T*k)
dist = np.empty([n_configs-1,3])
for i_configs in range(n_configs-1):
    for j_pos in range(3):
        dist[i_configs,j_pos] = np.absolute(all_positions[i_configs,2,j_pos]-all_positions[i_configs+1,2,j_pos])
delta = max(np.mean(dist,axis=0)) 
print("delta=",delta)



t = 0
acceptation = []
mc_positions = all_positions[:100,:,:]
mc_energies = all_energies[1:100+1]
guess_energy_overtime=np.empty(100)
guess_positions_overtime=np.empty([100,50,n_atoms,3])
guess_positions_overtime[0] = mc_positions[0,:,:]
for i_time in range(1,100):
    accepted_try_positions = np.empty([50,n_atoms,3])
    accepted_try_energys = np.empty(50)

    
    while t<50:
        increment_aleatoire=np.random.random((n_atoms,3))*2*delta
        print("increment aleatoire",increment_aleatoire)
        try_position = guess_positions_overtime[i_time-1,:,:] + increment_aleatoire - delta  
        print(np.shape(try_position))
        try_energy = get_energy(try_position)

    
        diff_E = mc_energies[i_time] - try_energy
        print("diff_E=",diff_E)
        if diff_E < 0 : 
            accepted_try_energys[t] = try_energy
            accepted_try_positions[t,:,:] = try_position
            t = t + 1
            acceptation.append(1)
        elif np.exp(-beta * diff_E) >= np.random.random():
            accepted_try_energy[t] = try_energy
            accepteded_try_positions[t,:,:] = try_position
            t = t + 1
            acceptation.append(1)
        else:
            acceptation.append(0)
            pass
    guess_positions_overtime[i_time] = accepted_try_positions[np.argmin(accepted_try_energy)]
    guess_energy_overtime[i_time]=accepted_try_energy[np.argmin(accepted_try_energy)]
     
print("taux d'acceptation=",np.mean(acceptation))   

