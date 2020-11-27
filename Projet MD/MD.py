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




def create_model():
    # instantiate model
    model = Sequential()
    # add a dense all-to-all tanh layer
    model.add(Dense(30, activation='tanh'))
    # add a dense all-to-all tanh layer
    model.add(Dense(30, activation='tanh'))
    # apply dropout with rate 0.5
  
    model.add(Dense(1, activation='softmax'))
    
    return model

model_H=create_model()
model_O=create_model()


inputO1 = tf.keras.layers.Input(shape=(63))
inputO2 = tf.keras.layers.Input(shape=(63,))
inputH1 = tf.keras.layers.Input(shape=(63,))
inputH2 = tf.keras.layers.Input(shape=(63,))
inputH3 = tf.keras.layers.Input(shape=(63,))
inputH4 = tf.keras.layers.Input(shape=(63,))
inputH5 = tf.keras.layers.Input(shape=(63,))

EO1=model_O(inputO1)
EO2=model_O(inputO2)
EH1=model_H(inputH1)
EH2=model_H(inputH2)
EH3=model_H(inputH3)
EH4=model_H(inputH4)
EH5=model_H(inputH5)

added= tf.keras.layers.Add()([EO1, EO2,EH1,EH2,EH3,EH4,EH5])
model_TOT=tf.keras.models.Model(inputs=[inputO1, inputO2, inputH1 , inputH2 , inputH3 , inputH4 , inputH5], outputs=added)

print('model created')

def compile_model(model):
    # create the mode
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

model_TOT=compile_model(model_TOT)   

print('model compiled')

print('Model Summary:')
model_TOT.summary()   


descriptors_train=np.swapaxes(descriptors[:8000],0,1).tolist()
descriptors_test=np.swapaxes(descriptors[8000:],0,1).tolist()
energies_train=energies[:8000].tolist()
energy_test=energies[8000:].tolist()


epochs=10
batch_size=64



history=model_TOT.fit(descriptors_train, energies_train[1],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(descriptors_test, energy_test))

