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

import os
from multiprocessing import Pool

au2ev = 2.72113838565563E+01 # Hartree to eV
au2kjpmol = 2.62549961709828E+03 # Hartree

def load_features(file_path):
    """Loads the feature file from the given path."""
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Error loading file: {file_path}, Error: {e}")
        return None

def load_energy_data(file_path):
    """Loads energy data from a text file, converting to mHartree."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        return np.array([[float(val) * 1000 for val in line.split()[1:6]] for line in lines])
    except Exception as e:
        print(f"Error reading energy file: {file_path}, Error: {e}")
        return None

def plot_energy_histogram(data_out_donor_reshaped, num_bins=100, range_xax=(-30, 0), file_name="inp_data_hist.pdf"):
    """
    Plots a histogram of energy distributions for donor molecules.
    
    Parameters:
    - data_out_donor_reshaped (numpy.ndarray): Reshaped donor energy data, where each column represents different energy levels.
    - num_bins (int): Number of bins for the histogram.
    """
    colors = ["#d62728", "#1f77b4"]  # Red for 1st, Blue for 2nd
    fig, ax = plt.subplots()

    # Calculate weights to normalize histograms
    weights1 = np.ones(len(data_out_donor_reshaped[:, 0])) / len(data_out_donor_reshaped[:, 0])
    weights2 = np.ones(len(data_out_donor_reshaped[:, 1])) / len(data_out_donor_reshaped[:, 1])
    
    # Plot histograms
    ax.hist(data_out_donor_reshaped[:, 0], bins=num_bins, range=range_xax,  
            weights=weights1, edgecolor=colors[0], alpha=0.5, label="1st", color=colors[0])
    ax.hist(data_out_donor_reshaped[:, 1], bins=num_bins, range=range_xax,  
            weights=weights2, edgecolor=colors[1], alpha=0.5, label="2nd", color=colors[1])
    
    # Labels and title
    ax.set_xlabel('Energy (mHartree)', fontsize=15)
    ax.set_ylabel('Normalized Frequency', fontsize=15)
    ax.set_ylim(0.00,0.10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.set_title('Histogram of energy distribution - lowest donor')
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    
    # Adjust layout and display plot
    fig.tight_layout()
    plt.savefig(file_name)

def data_loader(n_snapshot, n_samples, n_features, output_type="donor", num_outputs=1,
                data_dir="/home/tahmas41/work/ALMO_nn/Bulk_water_ALMO_karhan/data/",
                start_index=90000, end_index=100000, step=2,
                num_train_samples=500000, test_size=0.2, random_seed=123,
                use_multiprocessing=True):
    """
    Load molecular features and energy values, preprocess data, and return train/validation tensors.

    Parameters:
    - n_snapshot (int): Number of snapshots (timesteps).
    - n_samples (int): Number of samples per snapshot.
    - n_features (int): Number of feature dimensions.
    - output_type (str): Choose "donor", "acceptor", or "both".
    - num_outputs (int): Number of outputs to return (1 or 2), taken from first or first two energy columns.
    - data_dir (str): Base directory for data files.
    - start_index (int): Start index for file names.
    - end_index (int): End index for file names.
    - step (int): Step size for indexing (default: 2).
    - num_train_samples (int): Number of training samples.
    - test_size (float): Fraction of data to use for validation.
    - random_seed (int): Random seed for reproducibility.
    - use_multiprocessing (bool): Use multiprocessing for faster file loading.

    Returns:
    - n_features (int): Number of input features.
    - D_train (Tensor): Training feature set.
    - D_valid (Tensor): Validation feature set.
    - E_train (Tensor): Training energy values.
    - E_valid (Tensor): Validation energy values.
    """
    
    assert num_outputs in [1, 2], "num_outputs must be 1 or 2 (selects first or first two energy values)."

    # Prepare file indices
    file_indices = range(start_index, end_index, step)

    # ==========================
    # Load Features
    # ==========================
    features_allox = np.zeros((n_snapshot, n_samples, n_features))
    file_paths = [os.path.join(data_dir, f"0{idx}/coord_soap_nmax8_lmax6_cut5.npy") for idx in file_indices]

    if use_multiprocessing:
        with Pool() as pool:
            feature_data = pool.map(load_features, file_paths)
    else:
        feature_data = [load_features(fp) for fp in file_paths]

    valid_count = 0
    for i, features in enumerate(feature_data):
        if features is not None:
            features_allox[valid_count, :, :] = features
            valid_count += 1

    print("Feature shape:", features_allox.shape)
    # Reshape features
    features_allo_reshaped = features_allox.reshape(valid_count * n_samples, n_features)
    print("Feature reshaped:", features_allo_reshaped.shape)

    # ==========================
    # Load Energy Data (Donor/Acceptor/Both)
    # ==========================
    acceptor_paths = [os.path.join(data_dir, f"0{idx}/molecules.lowest.acceptor") for idx in file_indices]
    donor_paths = [os.path.join(data_dir, f"0{idx}/molecules.lowest.donor") for idx in file_indices]

    if use_multiprocessing:
        with Pool() as pool:
            acceptor_data = pool.map(load_energy_data, acceptor_paths)
            donor_data = pool.map(load_energy_data, donor_paths)
    else:
        acceptor_data = [load_energy_data(fp) for fp in acceptor_paths]
        donor_data = [load_energy_data(fp) for fp in donor_paths]

    data_out_accep = np.zeros((valid_count, n_samples, 5))
    data_out_donor = np.zeros((valid_count, n_samples, 5))

    valid_count = 0
    for i, (a_data, d_data) in enumerate(zip(acceptor_data, donor_data)):
        if a_data is not None and d_data is not None:
            data_out_accep[valid_count, :a_data.shape[0], :] = a_data
            data_out_donor[valid_count, :d_data.shape[0], :] = d_data
            valid_count += 1

    print("Energy shape:", data_out_accep.shape, data_out_donor.shape)
    # Reshape energy data
    data_out_accep_reshaped = data_out_accep.reshape(valid_count * n_samples, 5)
    data_out_donor_reshaped = data_out_donor.reshape(valid_count * n_samples, 5)
    print("Energy reshaped:", data_out_accep_reshaped.shape, data_out_donor_reshaped.shape)

    plot_energy_histogram(data_out_donor_reshaped, num_bins=100, range_xax=(-30, 0), file_name="out_data_hist_donor.pdf")
    plot_energy_histogram(data_out_accep_reshaped, num_bins=100, range_xax=(-30, 0), file_name="out_data_hist_accep.pdf")

    # Convert to log scale
    data_out_log_accep = np.log(-data_out_accep_reshaped)
    data_out_log_donor = np.log(-data_out_donor_reshaped)

    plot_energy_histogram(data_out_log_donor, num_bins=100, range_xax=(-4, 4), file_name="out_data_hist_donor_log.pdf")
    plot_energy_histogram(data_out_log_accep, num_bins=100, range_xax=(-4, 4), file_name="out_data_hist_accep_log.pdf")
    # ==========================
    # Select Training Data
    # ==========================
    np.random.seed(random_seed)
    idx_rnd = np.random.choice(len(features_allo_reshaped), num_train_samples, replace=False)

    D_numpy = features_allo_reshaped[idx_rnd, :]

    # Select output type
    if output_type == "donor":
        E_numpy = data_out_log_donor[idx_rnd, :num_outputs]  # First 1 or 2 columns
    elif output_type == "acceptor":
        E_numpy = data_out_log_accep[idx_rnd, :num_outputs]  # First 1 or 2 columns
    elif output_type == "both":
        E_numpy = np.concatenate([
            data_out_log_donor[idx_rnd, :num_outputs],  # First 1 or 2 columns
            data_out_log_accep[idx_rnd, :num_outputs]   # First 1 or 2 columns
        ], axis=1)
    else:
        raise ValueError("Invalid output_type! Choose 'donor', 'acceptor', or 'both'.")

    var_energy_train = E_numpy.var()
    # ==========================
    # Standardize Data (Properly)
    # ==========================
    D_train, D_valid, E_train, E_valid = train_test_split(
        D_numpy, E_numpy, test_size=test_size, random_state=random_seed
    )

    scaler = StandardScaler().fit(D_train)
    D_train = scaler.transform(D_train)
    D_valid = scaler.transform(D_valid)

    print(f"Shapes -> D_train: {D_train.shape}, D_valid: {D_valid.shape}, E_train: {E_train.shape}, E_valid: {E_valid.shape}")

    return n_features, var_energy_train, D_train, D_valid, E_train, E_valid
    

# Custom Dataset
class AtomisticDataset(Dataset):
    def __init__(self, features, energies):
        assert len(features) == len(energies), "Mismatch in dataset size"
        self.features = features  # Input features (e.g., atomic properties)
        self.energies = energies  # Target energies (2 energy outputs per sample)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.energies[idx], dtype=torch.float32)

#Then let us define our model and loss function:

class FFNet(nn.Module):
    """
    A configurable feed-forward neural network with multiple hidden layers,
    custom activation functions, Dropout, BatchNorm, and weight initialization.
    """
    def __init__(self, input_size, hidden_layers, output_size, activation_func=nn.ReLU, dropout_prob=0.0):
        """
        Parameters:
        - input_size (int): Number of input features.
        - hidden_layers (list of int): List specifying the size of each hidden layer.
        - output_size (int): Number of output features.
        - activation_func (nn.Module): Activation function class (e.g., nn.ReLU).
        - dropout_prob (float): Dropout probability (default 0.2).
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

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        mse = torch.mean((output - target) ** 2)
        
        # Compute variation of energy (assuming it's std of target)
        variation_of_energy = torch.var(target) + 1e-8  # Avoid division by zero
        
        loss = mse / variation_of_energy
        return loss

def energy_force_loss(E_pred, E_train, var_energy_train):
    """Custom loss function that targets both energies and forces.
    """
    energy_loss = torch.mean((E_pred - E_train)**2) / var_energy_train
    #force_loss = torch.mean((F_pred - F_train)**2) / var_force_train
    return energy_loss #+ force_loss


def train_model(model, optimizer, train_loader, val_loader, criterion, device, num_epochs, early_stopping=False, patience=20):

    train_losses = []
    valid_losses = []

    #model.to(device)
    best_valid_loss = float('inf')
    i_worse = 0
    # Add scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True, threshold=1e-4)


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training Phase
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation Phase
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        valid_loss /= len(val_loader)
        valid_losses.append(valid_loss)

        # Update learning rate based on validation loss
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_model_donor.pt")
            i_worse = 0 # Reset counter if improvement 
        else: 
            if early_stopping: # Only increment if early stopping is enabled and loss didn't improve
                i_worse += 1
                if i_worse >= patience:
                    print("Early stopping at epoch {}".format(epoch))
                    break

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    return train_losses, valid_losses

def main_train():

    #------------------------------------------------------
    n_snapshot = 3000
    n_samples = 125   # water molecules
    n_features = 952

    # Model parameters
    input_size = n_features      # Number of input features
    hidden_sizes = [50]*2  # Sizes of the hidden layers
    output_size = 2      # Number of output features
    num_epochs = 500
    learning_rate=1.0e-4
    weight_decay_param=1e-3
    batch_size=128
    #------------------------------------------------------

    n_features, var_energy_train, D_train, D_valid, E_train, E_valid = data_loader(n_snapshot, n_samples, n_features, output_type="donor", num_outputs=output_size, start_index=90000, end_index=96000, step=2, num_train_samples=375000)

    # Training on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Shape Check - D_train:", D_train.shape, "E_train:", E_train.shape)
    print("Shape Check - D_valid:", D_valid.shape, "E_valid:", E_valid.shape)


    train_dataset = AtomisticDataset(D_train, E_train)
    valid_dataset = AtomisticDataset(D_valid, E_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print("Train Batches:", len(train_loader), "Validation Batches:", len(valid_loader))
    
    # Define Loss Function
    #criterion = nn.MSELoss()
    criterion = CustomLoss()

    # Initialize model
    model = FFNet(input_size=n_features, hidden_layers=hidden_sizes, output_size=output_size, activation_func=nn.Tanh) 
    model.to(device)

    # The Adam optimizer is used for training the model parameters,  
    # L2 Regularization (weight decay, default=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_param)

    train_losses, valid_losses = train_model(model, optimizer, train_loader, valid_loader, criterion, device, num_epochs, early_stopping=True, patience=20) 

    # Plot Losses
    fig, ax = plt.subplots()
    x = range(len(train_losses))
    plt.yscale("log")
    plt.plot(x, train_losses, label="train")
    plt.plot(x, valid_losses, label="valid")

    plt.legend()
    plt.savefig("tv_loss.png")

if __name__ == "__main__":
    main_train() 
