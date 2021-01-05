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
from ase.io import iread, write
from keras.models import load_model
from vmd import measure

Zundel_NN = load_model('Fitted_Zundel_NN.h5')

datapath='../../../'
#positions and corresponding energies of a zundel molecule importation
all_positions = pickle.load(open(os.path.join(datapath,'zundel_100K_pos'),'rb'))
all_energies = pickle.load(open(os.path.join(datapath,'zundel_100K_energy'),'rb'))[1:]

positions = all_positions[::10]
energies = all_energies[::10]

#parameters settings
species = ["H","O"]
sigma_SOAP = 0.7
periodic = False
nmax = 4
lmax = 5
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
n_dims = n_features
n_elements = 2
n_oxygens = 2
n_hydrogens = 5
n_atoms = n_hydrogens + n_oxygens


#zundel molecule creation
zundels = np.empty(n_configs,dtype=object )
for i_configs in range(n_configs):
      zundels[i_configs] = Atoms(numbers=[8,8,1,1,1,1,1], positions=positions[i_configs])


# computing descriptors for each positions
descriptors=np.empty([n_configs,n_atoms,n_features])
for i_configs in range(n_configs):
    descriptors[i_configs,:,:] = soap.create(zundels[i_configs],positions=np.arange(n_atoms),n_jobs=4)
print('soap ok')

#scaling inputs and outputs
energies_scaler = StandardScaler().fit(energies.reshape((-1,1))) 
scaled_energies = energies_scaler.transform(energies.reshape((-1,1)))



n_features_oxygens = n_configs*n_oxygens
n_features_hydrogens = n_configs*n_hydrogens


scaled_descriptors = np.empty([n_features_hydrogens+n_features_oxygens,n_dims])


scaler_O_1 = StandardScaler()
scaler_H_1 = StandardScaler()
scaled_descriptors[n_features_oxygens:,:] = scaler_H_1.fit_transform(descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[n_features_oxygens:,:])
scaled_descriptors[:n_features_oxygens,:] = scaler_O_1.fit_transform(descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[:n_features_oxygens,:])



#PCA
var_ratio_pca_oxygens = np.empty(n_features_oxygens)
var_ratio_pca_hydrogens = np.empty(n_features_hydrogens)   

pca_oxygens = PCA(n_dims)
pca_hydrogens = PCA(n_dims)
pca_oxygens.fit(scaled_descriptors[:n_features_oxygens,:])
pca_hydrogens.fit(scaled_descriptors[n_features_oxygens:,:])
var_ratio_pca_hydrogens = pca_hydrogens.explained_variance_ratio_
var_ratio_pca_oxygens = pca_oxygens.explained_variance_ratio_

var_ratio_oxygens = 0
var_ratio_hydrogens = 0
pca_treshold_hydrogens = 0
pca_treshold_oxygens = 0

while var_ratio_hydrogens<0.999999:
    var_ratio_hydrogens +=  var_ratio_pca_hydrogens[pca_treshold_hydrogens]
    pca_treshold_hydrogens += 1
    
while var_ratio_oxygens<0.999999:
    var_ratio_oxygens += var_ratio_pca_oxygens[pca_treshold_oxygens]
    pca_treshold_oxygens += 1
        

pca_treshold = max(pca_treshold_hydrogens,pca_treshold_oxygens)
print("dimension desc post pca=", pca_treshold, "\n"
      "dimennsion desc pre pca=",n_dims)

scaled_pca_descriptors = np.empty([n_configs,n_atoms,n_dims])
for i_hydrogens in range(n_hydrogens):
    scaled_pca_descriptors[:,i_hydrogens+n_oxygens,:] = pca_hydrogens.transform(scaled_descriptors.reshape(n_configs,n_atoms,n_dims)[:,i_hydrogens+n_oxygens,:])
for i_oxygens in range(n_oxygens):
    scaled_pca_descriptors[:,i_oxygens,:] = pca_oxygens.transform(scaled_descriptors.reshape(n_configs,n_atoms,n_dims)[:,i_oxygens,:])
    
#scaling post pca
scaler_O_2 = MinMaxScaler()
scaler_H_2 = MinMaxScaler()

scaled_pca_descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[n_features_oxygens:,:pca_treshold] = scaler_H_2.fit_transform(scaled_descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[n_features_oxygens:,:pca_treshold])
scaled_pca_descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[:n_features_oxygens,:pca_treshold] = scaler_O_2.fit_transform(scaled_descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[:n_features_oxygens,:pca_treshold])


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


dist = np.empty([n_configs-1,n_atoms,3])
for i_configs in range(n_configs-1):
    for i_atom in range(n_atoms):
        for j_pos in range(3):
            dist[i_configs,i_atom,j_pos] = np.absolute(all_positions[i_configs,i_atom,j_pos]-all_positions[i_configs+1,i_atom,j_pos])
delta = (max(np.max(np.max(dist,axis=0),axis=1))- min(np.min(np.min(dist,axis=0),axis=1)))  #facteur tq taux acceptation = 0.4
print("delta=",delta)


mc_time = 10
mc_iterations = 50
acceptation = []
e = 1.602176e-19


guess_energy_overtime = np.empty(mc_time)
guess_positions_overtime = np.empty([mc_time,n_atoms,3])
guess_positions_overtime[0] = all_positions[0,:,:]


for i_time in range(1,mc_time):
    accepted_try_positions = np.empty([mc_iterations,n_atoms,3])
    accepted_try_energies = np.empty(mc_iterations)
    n_iterations = 0
    
    while n_iterations < mc_iterations:
        increment_aleatoire = np.random.random((n_atoms,3))*2*delta - delta 
        try_position = guess_positions_overtime[i_time-1,:,:] + increment_aleatoire 
        try_energy = get_energy(try_position)

    
        diff_E = (guess_energy_overtime[i_time-1] - try_energy) * 27.211396641308*e  #1 hartree = 27,211396641308eV
        if diff_E < 0 : 
            accepted_try_energies[n_iterations] = try_energy
            accepted_try_positions[n_iterations,:,:] = try_position
            n_iterations = n_iterations + 1
            acceptation.append(1)
        elif np.exp(-beta * diff_E) >= np.random.random():
            accepted_try_energies[n_iterations] = try_energy
            accepted_try_positions[n_iterations,:,:] = try_position
            n_iterations = n_iterations + 1
            acceptation.append(1)
        else:
            acceptation.append(0)
            pass
    guess_positions_overtime[i_time] = accepted_try_positions[np.argmin(accepted_try_energies)]
    guess_energy_overtime[i_time] = min(accepted_try_energies)
    print("mc_time=",i_time)
    
 
print("taux d'acceptation=",np.mean(acceptation))  

zundel_MC = np.empty(mc_time,dtype=object )
zundel_DFT = np.empty(mc_time,dtype=object )
 
for i_time_mc in range(mc_time):
      zundel_MC[i_time_mc] = Atoms(numbers=[8,8,1,1,1,1,1], positions=guess_positions_overtime[i_time_mc,:,:])

print(np.shape(zundel_MC))
for i_time_mc in range(mc_time):
      zundel_DFT[i_time_mc] = Atoms(numbers=[8,8,1,1,1,1,1], positions=all_positions[i_time_mc,:,:])

write("trajectoire_MC.xyz",zundel_MC,append=True)
write("trajectoire_DFT.xyz",zundel_DFT,append=True)

distance00_MC = measure.bond(0,1,iread("trajectoire_MC.xyz"))
distance00_DFT = measure.bond(0,1,iread("trajectoire_DFT.xyz"))

plt.clf()
plt.hist(distance00_MC,color="red",alpha=0.5)
plt.hist(distance00_DFT,color="blue",alpha=0.5)
plt.legend(['MC','DFT'],loc='best')
plt.savefig('g(r)oxygens.jpg')





