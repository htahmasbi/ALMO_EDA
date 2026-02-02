#!/bin/python

import numpy as np
from dscribe.descriptors import SOAP
from ase.io import read, write
import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from torch.utils.data import random_split
import optuna
import torch.optim as optim

from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold

au2ev = 2.72113838565563E+01 # Hartree to eV
au2kjpmol = 2.62549961709828E+03 # Hartree

def data_loader(base_path, n_snapshots, n_samples, n_features):
    
    #----------------------- 
    #features_allox = np.zeros((n_snapshots, n_samples, n_features))
    features_list = []
    
    #j = 0
    for i in range(n_snapshots):
        file_path = f"{base_path}/{system_typ}_coord_{i}_modified_soap_n8l6c5.npy"
        #ox_features = np.load("../../../data_CPO-27/Zn2W7/Zn2W7_coord_"+str(i)+"_modified_soap_n12l12c4.npy")
        ox_features = np.load(file_path)
        #features_allox[j, :, :] = ox_features
        #j+=1
        features_list.append(ox_features)

    features_allox = np.stack(features_list)

    print(features_allox.shape)
    
    features_allo_reshaped = np.reshape(features_allox, (n_snapshots*n_samples, n_features))
    print(features_allo_reshaped.shape)
    #np.save("coord_desc_soap.npy", features_allox)
    
    total_size = n_snapshots*n_samples 
    # Load the dataset
    # select some points randomly for training
    idx_rnd = np.random.randint(total_size, size=total_size)
    D_numpy = features_allo_reshaped[idx_rnd, :] #[:, 1, :] # O center
    n_samples, n_features = D_numpy.shape
    print(D_numpy.shape)

    # Standardize input for improved learning. Fit is done only on training data,
    # scaling is applied to both descriptors and their derivatives on training and
    # test sets.
    scaler = StandardScaler().fit(D_numpy)
    D_numpy = scaler.transform(D_numpy)
    D_numpy = torch.Tensor(D_numpy)
    
    return D_numpy

class FFNet(nn.Module):
    """
    A configurable feed-forward neural network with multiple hidden layers,
    custom activation functions, and weight initialization.
    """
    def __init__(self, input_size, hidden_layers, output_size, activation_func, dropout_prob=0.0):
        """
        Parameters:
        - input_size (int): Number of input features.
        - hidden_layers (list of int): List specifying the size of each hidden layer.
        - output_size (int): Number of output features.
        - activation_func (nn.Module): Activation function class (e.g., nn.ReLU).
        """
        super(FFNet, self).__init__()
        layers = []
        in_features = input_size

        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            #layers.append(nn.BatchNorm1d(hidden_size))  # Batch Normalization
            layers.append(activation_func())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))  # Dropout to prevent overfitting
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))

        # Create the model
        self.model = nn.Sequential(*layers)

        # Initialize weights
#        self._initialize_weights()
#
#    def _initialize_weights(self):
#        """Initialize weights using Xavier/Glorot initialization for better convergence."""
#        for layer in self.model:
#            if isinstance(layer, nn.Linear):
#                nn.init.xavier_uniform_(layer.weight)
#                if layer.bias is not None:
#                    nn.init.zeros_(layer.bias)


    def forward(self, x):
        """
        Forward pass through the network.
        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, input_size).
        
        Returns:
        - Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.model(x)

def main_eval(base_path, Ns):

    n_snapshots = 1000  
    n_samples = Ns #258 #183/174 #72
    n_features = 952
    torch.manual_seed(123)

    input_size = n_features      # Number of input features
    hidden_sizes = [50]*2  # Sizes of the hidden layers
    output_size = 2      # Number of output features

    model = FFNet(input_size=n_features, hidden_layers=hidden_sizes, output_size=output_size, activation_func=nn.Tanh)

    D_whole = data_loader(base_path, n_snapshots, n_samples, n_features)

    # Analysis
    # Way to tell pytorch that we are entering the evaluation phase
    model.load_state_dict(torch.load("best_model_donor.pt"))
    model.eval()
    
    #D_whole.requires_grad = True
    E_whole_pred = model(D_whole)
    #D_numpy_wcp.requires_grad = True
    #E_whole_pred = model(D_numpy_wcp)
    print(E_whole_pred.shape)
    
    # Plot energies for the whole range
    E_whole_pred = E_whole_pred.detach().numpy()
    #E_whole_pred = scaler_e.inverse_transform(E_whole_pred)
    print(E_whole_pred[0])
    return E_whole_pred
    
def main_plot(E_whole_pred, num_bins=100, ylim_top=0.25):
    #num_bins = 100
    colors = ["#d62728", "#1f77b4"]  # Red for 1st, Blue for 2nd
    
    #E_whole_pred = -np.exp(E_whole_pred)
    E_whole_pred = -np.exp(E_whole_pred)
    fig, ax = plt.subplots()
    
    weights1=np.ones(len(E_whole_pred[:, 0]))/len(E_whole_pred[:, 0])
    weights2=np.ones(len(E_whole_pred[:, 1]))/len(E_whole_pred[:, 1])

    # the histogram of the data
    n1, bins1, patches1 = ax.hist(E_whole_pred[:, 0], bins=num_bins, range=(-30,0), weights=weights1, alpha=0.5, label="1st", color=colors[0], histtype='stepfilled', zorder=2)
    n2, bins2, patches2 = ax.hist(E_whole_pred[:, 1], bins=num_bins, range=(-30,0), weights=weights2, alpha=0.5, label="2nd", color=colors[1], histtype='stepfilled', zorder=2)
    #print(np.sum(n), bins)
    print(n1, bins1, patches1)
    
    ax.set_xlabel('Energy (mHartree)', fontsize=15)
    ax.set_ylabel('Normalized Frequency', fontsize=15)
    ax.set_ylim(0.00,ylim_top)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.set_title('Histogram of energy distribution - lowest donor')
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig(f"EDA_{system_typ}.pdf")

if __name__ == "__main__":
    for i in range(5): 
        configs = ["Zn2W2", "Zn2W5", "Zn2W7", "Cu2W2", "Cu2W5"]
        nsamples = [72, 183, 258, 72, 174]
        #ylim = [0.15, 0.25, 0.15, 0.12, 0.12]
        system_typ = configs[i]
        Ns = nsamples[i]
        base_path = f"/home/tahmas41/work/ALMO_nn/Bulk_water_ALMO_karhan/data_CPO-27/{system_typ}"
        E_whole_pred = main_eval(base_path, Ns)
        main_plot(E_whole_pred, num_bins=100) #, ylim_top=ylim[i])
