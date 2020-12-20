import numpy as np
from dscribe.descriptors import SOAP
from ase.build import molecule
import pickle




#positions and corresponding energies of a zundel molecule importation
positions = pickle.load(open('zundel_100K_pos','rb'))[::10]
energies = pickle.load(open('zundel_100K_energy','rb'))[::10]
working_size = np.shape(positions)[0]

def descriptor(species,sigma_SOAP,nmax,lmax,rcut,N_atoms):
    
    #soap settings
    soap = SOAP(
        species=species,
        sigma=sigma_SOAP,
        periodic=False,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        sparse=False,
        #rbf='polynomial'
    )
    
    N_features = soap.get_number_of_features()
    
    
    #zundel molecule creation
    from ase import Atoms
    zundels = np.empty(working_size,dtype=object ) # il est souvent préférable d'utiliser des array
    for i_time in range(working_size):  # une boucle sur toutes les config possibles
          zundels[i_time] = Atoms(numbers=[8,8,1,1,1,1,1], positions=positions[i_time])
    
    
    # computing descriptors for each positions
    descriptors=np.empty([working_size,N_atoms,N_features])
    for i_time in range(working_size):
        descriptors[i_time,:,:] = soap.create(zundels[i_time],positions=np.arange(N_atoms),n_jobs=4)
        
    with open('descriptors({},{},{})'.format(sigma_SOAP,nmax,lmax),'wb') as f:
        pickle.dump(descriptors,f)


#parameters settings
species = ["H","O"]
sigma_SOAP = [0.01]
periodic = False #pour le Zundel, mais True pour CO2
nmax = [3]
lmax = [2]
rcut = 6.0
N_atoms = 7

#computing and saving descriptors
for sigma in sigma_SOAP:
    for n in nmax:
        for l in lmax:
            descriptor(species,sigma,n,l,rcut,N_atoms)

