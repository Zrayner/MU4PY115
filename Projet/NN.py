from __future__ import print_function
import keras,sklearn
from sklearn.preprocessing import scale
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

n_elements = 2
n_oxygens = 2
n_hydrogens = 5
n_atoms = n_hydrogens + n_oxygens

#loading data
descriptors = pickle.load(open('descriptors(0.01,3,2)','rb'))
energies = pickle.load(open('zundel_100K_energy','rb'))[::10]

#scaling inputs and outputs
scaled_energies = scale(energies)
# scaled_descriptors = np.empty(np.shape(descriptors))

# for i_atom in range(n_atoms):
#     for j_descriptors in range(np.shape(descriptors)[2]):
#         scaled_descriptors[:,i_atom,j_descriptors]=scale(descriptors[:,i_atom,j_descriptors],axis=0)


#swaping axes to fit in the NN
# descriptors_swap = np.swapaxes(descriptors,0,1)

#PCA
n_features_oxygens = np.shape(descriptors)[1]*n_oxygens
n_features_hydrogens = np.shape(descriptors_swap)[1]*n_hydrogens
descriptors_pca = np.reshape(descriptors(np.shape(descriptors_swap)[0],n_features_hydrogens+n_features_oxygens))

var_ratio_pca_oxygens = np.empty(n_features_oxygens)   
var_ratio_pca_hydrogens = np.empty(n_features_hydrogens)   

pca_oxygens = PCA(n_features_oxygens)
pca_hydrogens = PCA(n_features_hydrogens)
pca_oxygens.fit(descriptors_pca[:,:n_features_oxygens-1])
pca_hydrogens.fit(descriptors_pca[:,n_features_oxygens:])
var_ratio_pca_oxygens = pca_oxygens.explained_variance_ratio_
var_ratio_pca_hydrogens = pca_hydrogens.explained_variance_ratio_


#setting the train and test and validation set
descriptors_train = descriptors_swap[:,:80000,:]
descriptors_val = descriptors_swap[:,80000:85000,:]
descriptors_test = descriptors_swap[:,85000:,:]
energies_train = scaled_energies[:80000]
energies_val = scaled_energies[80000:85000]
energies_test = scaled_energies[85000:]


#creating a list of array to fit in the NN
descriptors_train_nn = []
descriptors_test_nn = []
descriptors_val_nn = []
for i_atom in range(n_hydrogens+n_oxygens):
    descriptors_train_nn.append(descriptors_train[i_atom,:,:])
    descriptors_test_nn.append(descriptors_test[i_atom,:,:])
    descriptors_val_nn.append(descriptors_val[i_atom,:,:])



#creating the model
def model():
    
    model = Sequential()
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(30, activation='tanh'))
    model.add(Dense(1,kernel_regularizer='l1_l2'))
    
    return model



model0=model()
modelH=model()

inputs = []
for i_elements in range(n_hydrogens+n_oxygens):
    inputs.append(keras.layers.Input(shape=(63,)))

subnets = []
for i_oxygens in range(n_oxygens):
    subnets.append(model0(inputs[i_oxygens]))
for j_hydrogens in range(n_hydrogens):
    subnets.append(modelH(inputs[j_hydrogens+n_oxygens]))
    

added = keras.layers.Add()(subnets)
model = keras.models.Model(inputs, outputs=added)



def compile_model(model):
    
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['MeanSquaredError'])
    return model



Zundel_NN = compile_model(model)
Zundel_NN.summary()

batchsize = 64
epochs= 100






early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = Zundel_NN.fit(descriptors_train_nn,energies_train,
                                      batch_size=batchsize,
                                      epochs=epochs,
                                      verbose=2,
                                      callbacks=[early_stopping,],
                                      validation_data=(descriptors_val_nn,energies_val))




results = Zundel_NN.evaluate(descriptors_test_nn,
                             energies_test,
                             batchsize)

print(results)

predicted_energies = Zundel_NN.predict(descriptors_test_nn)


plt.figure(figsize=[10, 5])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'test'], loc='best')
plt.savefig('loss_over_epoch.jpg')
plt.clf()
plt.plot(energies_test,predicted_energies,'.')
plt.plot(energies_test,energies_test,'.',color='red')
plt.savefig('comparaison')

