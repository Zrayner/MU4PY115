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

datapath='../../../'
#positions and corresponding energies of a zundel molecule importation
all_positions = pickle.load(open(os.path.join(datapath,'zundel_100K_pos'),'rb'))
all_energies = pickle.load(open(os.path.join(datapath,'zundel_100K_energy'),'rb'))

positions = all_positions[::2]
energies = all_energies[1::2]

#parameters settings
species = ["H","O"]
sigma_SOAP = 0.7
periodic = False
nmax = 4
lmax = 5
rcut = 11.0

model_params={
        'layers_units': 30,
        'layers_number': 2,
        'kernel_initializer': None,
 
       
    
    }

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

#swaping axes for NN purpose
descriptors_swap = np.swapaxes(scaled_pca_descriptors.reshape(n_configs,n_atoms,n_dims)[:,:,:pca_treshold],0,1)


#setting the train and test and validation set
descriptors_train = descriptors_swap[:,:400000,:]
descriptors_val = descriptors_swap[:,400000:450000,:]
descriptors_test = descriptors_swap[:,450000:,:]
energies_train = scaled_energies[:400000]
energies_val = scaled_energies[400000:450000]
energies_test = scaled_energies[450000:]


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



model0 = model(model_params)
modelH = model(model_params)

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

batchsize = 10
epochs= 300

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
                                      verbose=1,
                                      callbacks=[early_stopping,lr_reduce],
                                      validation_data=(descriptors_val_nn,energies_val))




Zundel_NN.save('Fitted_Zundel_NN.h5')

#descaling energies and outputs
predicted_energies = Zundel_NN.predict(descriptors_test_nn)
descaled_energies = energies_scaler.inverse_transform(scaled_energies)
descaled_predicted_energies = energies_scaler.inverse_transform(predicted_energies)
energy = np.linspace(0,0.008,200)
print("max error=",max(np.absolute(descaled_energies[95000:]-descaled_predicted_energies)))
plt.figure(figsize=[10, 5])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'val'], loc='best')
plt.savefig('loss_over_epoch.jpg')
plt.clf()
plt.figure(figsize=[7,7])
plt.plot(descaled_energies[95000:],descaled_predicted_energies,'.',markersize=2)
plt.plot(energy,energy,markersize=2,color='red')
plt.savefig('comparaison.jpg')

"""'''Monte-Carlo'''


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter 


T = 100
k = 1.380649e-23
beta = 1/(T*k)
dist = np.empty([n_configs-1,3])
for i_configs in range(n_configs-1):
    for j_pos in range(3):
        dist[i_configs,j_pos] = np.absolute(all_positions[i_configs,2,j_pos]-all_positions[i_configs+1,2,j_pos])
delta = max(np.mean(dist,axis=0)) 
print("delta=",delta)


    
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


t = 0
acceptation = []
mc_positions = all_positions[:100,:,:]
mc_energies = all_energies[1:100+1]
guess_energy_overtime=np.empty(100)
guess_positions_overtime=np.empty([100,n_atoms,3])
guess_positions_overtime[0] = mc_positions[0,:,:]
for i_time in range(1,100):
    accepted_try_positions = np.empty([50,n_atoms,3])
    accepted_try_energys = np.empty(50)

    
    while t<50:
        increment_aleatoire=np.random.random((n_atoms,3))*2*delta
        print("increment aleatoire",increment_aleatoire)
        try_position = guess_positions_overtime[i_time-1,:,:] + increment_aleatoire - delta  
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

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c, m, atom in [('r', 'o', 0), ('r', 'o', 1),
                   ('black', 'o', 2), ('black', 'o', 3),
                   ('black', 'o', 4), ('black', 'o', 5),
                   ('black', 'o', 6)]:
    
    xs = mc_positions[:,atom,0]
    ys = mc_positions[:,atom,1]
    zs = mc_positions[:,atom,2]
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig('monte-carlo.jpg')
plt.clf()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for c, m, atom in [('r', 'o', 0), ('r', 'o', 1),
                   ('black', 'o', 2), ('black', 'o', 3),
                   ('black', 'o', 4), ('black', 'o', 5),
                   ('black', 'o', 6)]:
    
    xs = positions[:100,atom,0]
    ys = positions[:100,atom,1]
    zs = positions[:100,atom,2]
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig('DFT.jpg')

plt.clf()
plt.plot(all_energies[:100],mc_energies,'.',marksize=2)
plt.plot(all_energies[:100],all_energies[:100],marksize=2)
plt.savefig('mc_energies.jpg')
plt.clf()

        

        

        



