# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:52:25 2020

@author: ewenf
"""

import numpy as np
from dscribe.descriptors import SOAP
from ase.build import molecule
import pickle
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import keras

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Dropout

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials

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
positions = pickle.load(open('zundel_100K_pos', 'rb'))[::10]
energies = pickle.load(open('zundel_100K_energy', 'rb'))[::10]

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


space = {'lr': hp.choice('lr', [0.00001, 0.0001, 0.001, 0.1]),
    
    
    
    }

space



def objective(params):
    print ('Params testing: ', params)
    def create_model():
        # instantiate model
        model = Sequential()
        # add a dense all-to-all tanh layer
        model.add(Dense(30, activation='tanh'))
        # add a dense all-to-all tanh layer
        model.add(Dense(30, activation='tanh'))
        # apply dropout with rate 0.5
      
        model.add(Dense(1, activation='linear'))
        
        return model
    
    model_H=create_model()
    model_O=create_model()
    
    
    inputO1 = keras.layers.Input(shape=(63))
    inputO2 = keras.layers.Input(shape=(63,))
    inputH1 = keras.layers.Input(shape=(63,))
    inputH2 = keras.layers.Input(shape=(63,))
    inputH3 = keras.layers.Input(shape=(63,))
    inputH4 = keras.layers.Input(shape=(63,))
    inputH5 = keras.layers.Input(shape=(63,))
    
    EO1=model_O(inputO1)
    EO2=model_O(inputO2)
    EH1=model_H(inputH1)
    EH2=model_H(inputH2)
    EH3=model_H(inputH3)
    EH4=model_H(inputH4)
    EH5=model_H(inputH5)
    
    added= keras.layers.Add()([EO1, EO2,EH1,EH2,EH3,EH4,EH5])
    model_TOT=keras.models.Model(inputs=[inputO1, inputO2, inputH1 , inputH2 , inputH3 , inputH4 , inputH5], outputs=added)
    
    print('model created')
    
    
    def compile_model(model):
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                      metrics=['mean_squared_error'])
        return model
    
    model_TOT=compile_model(model_TOT)   
    
    print('model compiled')
    
    print('Model Summary:')
    model_TOT.summary()   
    
    model_TOT.save("model_TOT")
    
    descriptors_train=np.swapaxes(descriptors[:80000],0,1)
    descriptors_test=np.swapaxes(descriptors[80000:],0,1)
    
    
    energies_train=energies[:80000]
    energies_train = scale(energies_train[:,np.newaxis])
    energies_test=energies[80000:]
    energies_test =  scale(energies_test[:,np.newaxis])
    
    list_descriptor_train=[]
    for i in range(np.size(descriptors_train[:,1,1])):
        list_descriptor_train.append(scale(descriptors_train[i]))
       
    list_descriptor_test=[]
    for i in range(np.size(descriptors_test[:,1,1])):
        list_descriptor_test.append(scale(descriptors_test[i]))
    
    epochs=8
    batch_size=30
    
    print('Fitting')
    callback=keras.callbacks.EarlyStopping(monitor='loss',patience=5)
    
    
    
    
    
    
    
    
    history_TOT=model_TOT.fit(list_descriptor_train, energies_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callback,
              verbose=1,
              validation_data=(list_descriptor_test, energies_test))
    
    last_loss=history_TOT.history['val_loss'][-1]

    return {'loss': last_loss, 'status': STATUS_OK }



trials=Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=4)
print (best)
print (trials.best_trial)