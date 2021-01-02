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


#positions and corresponding energies of a zundel molecule importation
positions = pickle.load(open('zundel_100K_pos','rb'))[::10]
energies = pickle.load(open('zundel_100K_energy','rb'))[::10]


#parameters settings
species = ["H","O"]
sigma_SOAP = 0.01
periodic = False
nmax = 5
lmax = 3
rcut = 8.0

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


#scaling inputs and outputs
energies_scaler = StandardScaler().fit(energies.reshape((-1,1))) 
scaled_energies = energies_scaler.transform(energies.reshape((-1,1)))
scaled_descriptors = np.empty(np.shape(descriptors))

scaler_O_1 = [StandardScaler()]*2
scaler_H_1 = [StandardScaler()]*5

for i_hydrogens in range(n_hydrogens):
    scaled_descriptors[:,i_hydrogens+n_oxygens,:]=scaler_H_1[i_hydrogens].fit_transform(descriptors[:,i_hydrogens+n_oxygens,:])
for i_oxygens in range(n_oxygens):
    scaled_descriptors[:,i_oxygens,:]=scaler_O_1[i_oxygens].fit_transform(descriptors[:,i_oxygens,:])



#PCA
n_features_oxygens = n_configs*n_oxygens
n_features_hydrogens = n_configs*n_hydrogens
descriptors_pca = np.reshape(scaled_descriptors,(n_features_hydrogens+n_features_oxygens,n_dims))

var_ratio_pca_oxygens = np.empty(n_features_oxygens)
var_ratio_pca_hydrogens = np.empty(n_features_hydrogens)   

pca_oxygens = PCA(n_dims)
pca_hydrogens = PCA(n_dims)
pca_oxygens.fit(descriptors_pca[:n_features_oxygens,:])
pca_hydrogens.fit(descriptors_pca[n_features_oxygens:,:])
var_ratio_pca_hydrogens = pca_hydrogens.explained_variance_ratio_
var_ratio_pca_oxygens = pca_oxygens.explained_variance_ratio_

var_ratio_oxygens = 0
var_ratio_hydrogens = 0
pca_treshold_hydrogens = 0
pca_treshold_oxygens = 0

while var_ratio_hydrogens<0.9999:
    var_ratio_hydrogens +=  var_ratio_pca_hydrogens[pca_treshold_hydrogens]
    pca_treshold_hydrogens += 1
    
while var_ratio_oxygens<0.9999:
    var_ratio_oxygens += var_ratio_pca_oxygens[pca_treshold_oxygens]
    pca_treshold_oxygens += 1
        

pca_treshold = max(pca_treshold_hydrogens,pca_treshold_oxygens)
print(pca_treshold)


for i_hydrogens in range(n_hydrogens):
    pca_hydrogens.transform(scaled_descriptors[:,i_hydrogens+n_oxygens,:])
for i_oxygens in range(n_oxygens):
    pca_oxygens.transform(scaled_descriptors[:,i_oxygens,:])
    


scaled_pca_descriptors = np.empty(np.shape(scaled_descriptors[:,:,:pca_treshold]))


scaler_O_2 = [StandardScaler()]*2
scaler_H_2 = [StandardScaler()]*5

for i_hydrogens in range(n_hydrogens):
    scaled_pca_descriptors[:,i_hydrogens+n_oxygens,:] = scaler_H_2[i_hydrogens].fit_transform(scaled_descriptors[:,i_hydrogens+n_oxygens,:pca_treshold])

for i_oxygens in range(n_oxygens):
    scaled_pca_descriptors[:,i_oxygens,:] = scaler_O_2[i_oxygens].fit_transform(scaled_descriptors[:,i_oxygens,:pca_treshold])



descriptors_swap = np.swapaxes(scaled_pca_descriptors,0,1)


#setting the train and test and validation set
descriptors_train = descriptors_swap[:,:85000,:]
descriptors_val = descriptors_swap[:,85000:95000,:]
descriptors_test = descriptors_swap[:,95000:,:]
energies_train = scaled_energies[:85000]
energies_val = scaled_energies[85000:95000]
energies_test = scaled_energies[95000:]


#creating a list of array to fit in the NN
descriptors_train_nn = []
descriptors_test_nn = []
descriptors_val_nn = []
for i_atom in range(n_atoms):
    descriptors_train_nn.append(descriptors_train[i_atom,:,:])
    descriptors_test_nn.append(descriptors_test[i_atom,:,:])
    descriptors_val_nn.append(descriptors_val[i_atom,:,:])



# creating the model
# tf.keras.regularizers.L1L2(
#     l1=0.0001, l2=0.0001
# )
def model():
    
    model = Sequential()
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(1,))
#kernel_regularizer='l1_l2'
    
    return model



model0=model()
modelH=model()

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
    
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['MeanSquaredError'])
    return model



Zundel_NN = compile_model(zundel_model)

batchsize = 32
epochs= 100

#callbacks
lr_reduce = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=2, verbose=0,
    mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-10
)


early_stopping = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001, patience=4)

#training the NN
history = Zundel_NN.fit(descriptors_train_nn,energies_train,
                                      batch_size=batchsize,
                                      epochs=epochs,
                                      verbose=2,
                                      callbacks=[early_stopping,lr_reduce],
                                      validation_data=(descriptors_val_nn,energies_val))






predicted_energies = Zundel_NN.predict(descriptors_test_nn)
descaled_energies = energies_scaler.inverse_transform(scaled_energies)
descaled_predicted_energies = energies_scaler.inverse_transform(predicted_energies)

plt.figure(figsize=[10, 5])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='best')
plt.savefig('loss_over_epoch.jpg')
plt.clf()
plt.figure(figsize=[7,7])
plt.plot(descaled_energies[95000:],descaled_predicted_energies,'.')
plt.plot(descaled_energies[95000:],descaled_energies[95000:],'.',color='red')
plt.savefig('comparaison.jpg')
