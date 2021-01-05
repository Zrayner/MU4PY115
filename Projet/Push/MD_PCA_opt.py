# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 04:06:55 2020

@author: ewenf
"""



from __future__ import print_function
import keras,sklearn
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
# suppress tensorflow compilation warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
seed=0
np.random.seed(seed) # fix random seed
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import pickle
from sklearn.decomposition import PCA

from dscribe.descriptors import SOAP
from ase.build import molecule
import pickle
from ase import Atoms
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
import math 


datapath='../../../'
#positions and corresponding energies of a zundel molecule importation
positions = pickle.load(open(os.path.join(datapath,'zundel_100K_pos'),'rb'))[::5]
energies = pickle.load(open(os.path.join(datapath,'zundel_100K_energy'),'rb'))[1::5]




"""Tested_Space= {'nmax': hp.choice('nmax', [2,3, 4, 5]),
        'lmax': hp.choice('lmax', [2,3, 4, 5]),
        'rcut': hp.choice('rcut', [5.0,6.0,7.0, 8.0,9.0,10.0]),
        'sigma_SOAP': hp.choice('sigma_SOAP', [0.01,0.1,1,0.001]),
        'layers_units': hp.choice('layers_units', [20,30,40,50]),
        'layers_number': hp.choice('layers_number', [2,3,4]),
        'kernel_initializer': hp.choice('kernel_initializer', [None, 'GlorotUniform']),
                
    
    
    
    
    }"""

best_params_yet={'nmax': 4,
        'lmax': 5,
        'rcut': 11.0,
        'sigma_SOAP': 1,
        'layers_units': 30,
        'layers_number': 2,
        'kernel_initializer': None,
 
       
    
    }


"""To_test= { 'batch_size'
                rcut >10
                
          
                
    
 
    
    } """

Space= {'Scaler_2': hp.choice('Scaler_2', [StandardScaler(), MaxAbsScaler(), MinMaxScaler()]),

                
    
    
    
    
    }


def objective(space_params):
    
    print ('Params testing: ', space_params)
    #parameters settings
    species = ["H","O"]
    sigma_SOAP = best_params_yet['sigma_SOAP']
    periodic = False
    nmax = best_params_yet['nmax']
    lmax = best_params_yet['lmax']
    rcut = best_params_yet['rcut']
    
 
    
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
    
    
    
    scaler_O_2 = space_params['Scaler_2']
    scaler_H_2 = space_params['Scaler_2']
    
    scaled_pca_descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[n_features_oxygens:,:pca_treshold] = scaler_H_2.fit_transform(scaled_descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[n_features_oxygens:,:pca_treshold])
    scaled_pca_descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[:n_features_oxygens,:pca_treshold] = scaler_O_2.fit_transform(scaled_descriptors.reshape(n_features_hydrogens+n_features_oxygens,n_dims)[:n_features_oxygens,:pca_treshold])
    
    #swaping axes for NN purpose
    descriptors_swap = np.swapaxes(scaled_pca_descriptors.reshape(n_configs,n_atoms,n_dims)[:,:,:pca_treshold],0,1)
    
    
    #setting the train and test and validation set
    descriptors_train = descriptors_swap[:,:85000*2,:]
    descriptors_val = descriptors_swap[:,85000*2:95000*2,:]
    descriptors_test = descriptors_swap[:,95000*2:,:]
    energies_train = scaled_energies[:85000*2]
    energies_val = scaled_energies[85000*2:95000*2]
    energies_test = scaled_energies[95000*2:]
    
    
    #creating a list of array to fit in the NN
    descriptors_train_nn = []
    descriptors_test_nn = []
    descriptors_val_nn = []
    for i_atom in range(n_atoms):
        descriptors_train_nn.append(descriptors_train[i_atom,:,:])
        descriptors_test_nn.append(descriptors_test[i_atom,:,:])
        descriptors_val_nn.append(descriptors_val[i_atom,:,:])
    
    
    
    
    def model(params):
            
        model = Sequential()
        for i in range(params['layers_number']):
            model.add(Dense(params['layers_units'], activation='tanh',kernel_initializer=params['kernel_initializer']))
        model.add(Dense(1,))
      
            
        return model
    
    
    
    model0 = model(best_params_yet)
    modelH = model(best_params_yet)
    
    inputs = []
    for i_atoms in range(n_atoms):
        inputs.append(keras.layers.Input(shape=(pca_treshold,)))
    
    subnets = []
    for i_oxygens in range(n_oxygens):
        subnets.append(model0(inputs[i_oxygens]))
    for j_hydrogens in range(n_hydrogens):
        subnets.append(modelH(inputs[i_hydrogens+n_oxygens]))
        
    
    added = keras.layers.Add()(subnets)
    zundel_model = keras.models.Model(inputs, outputs=added)
    
    
    
    def compile_model(model):
        
        model.compile(loss=keras.losses.mse,
                      optimizer=keras.optimizers.Adam())
        return model
    
    
    
    Zundel_NN = compile_model(zundel_model)
    
    batchsize = 200
    epochs= 1000
    
    #callbacks
    lr_reduce = keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=4, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-10
    )
    
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001, patience=10)
    
    #training the NN
    history = Zundel_NN.fit(descriptors_train_nn,energies_train,
                                          batch_size=batchsize,
                                          epochs=epochs,
                                          verbose=2,
                                          callbacks=[early_stopping,lr_reduce],
                                          validation_data=(descriptors_val_nn,energies_val))
    

    
    last_loss=history.history['val_loss'][-1]
    return {'loss': last_loss, 'status': STATUS_OK }
    
trials=Trials()
best = fmin(objective, Space, algo=tpe.suggest, trials=trials, max_evals=10)
print (best)
print (trials.best_trial)



