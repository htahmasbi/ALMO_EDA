# DScribe: SOAP descriptor

from dscribe.descriptors import SOAP
from ase.io import read, write
import numpy as np


# First we will have to create the features for atomic environments. Let's
# use SOAP.

# rcut (float) - A cutoff for local region in angstroms. Should be bigger than 1 angstrom.
# nmax (int) - The number of radial basis functions.
# lmax (int) - The maximum degree of spherical harmonics.
desc = SOAP(species=[1, 8], r_cut=5.0, n_max=8, l_max=6, sigma=0.1, periodic=True, sparse=False)

a_snapshot = read("coord.cif", format='cif')
a_features = desc.create(a_snapshot)
n_samples, n_features = a_features.shape
print(n_samples, n_features)

#j = 0
for i in range(90000, 100000, 2):
    o_features = np.zeros((125, n_features))
    a_snapshot = read("../data/0"+str(i)+"/coord.cif", format='cif')
    a_features = desc.create(a_snapshot)
    for k in range(125): 
        o_features[k, :] = np.array([a_features[k*3]])
    #a_features = desc.create(a_snapshot)
    np.save("../data/0"+str(i)+"/coord_soap_nmax8_lmax6_cut5.npy", o_features)
    #j+=1
