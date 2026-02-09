#!/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
from matplotlib import rcParams

def load_features(file_path):
    return np.load(file_path)

def load_energy_data(file_path):
    with open(file_path, "r") as f:
        return np.array([[float(x) * 1000 for x in line.split()[1:6]] for line in f])

def data_loader(base_path, n_snapshot, n_samples, n_features, output_type="donor", num_outputs=2,
                start_index=96000, end_index=100000, step=2, #num_train_samples=250000,
                use_multiprocessing=True):
    """Loads dataset and preprocesses it for training and evaluation with optional multiprocessing."""

    file_indices = range(start_index, end_index, step)
    feature_paths = [f"{base_path}/0{i}/coord_soap_nmax8_lmax6_cut5.npy" for i in file_indices]
    acceptor_paths = [f"{base_path}/0{i}/molecules.lowest.acceptor" for i in file_indices]
    donor_paths = [f"{base_path}/0{i}/molecules.lowest.donor" for i in file_indices]    

    if use_multiprocessing:
        with Pool() as pool:
            features_allox = np.array(pool.map(load_features, feature_paths))
        with Pool() as pool:
            data_out_accep = np.array(pool.map(load_energy_data, acceptor_paths))
            data_out_donor = np.array(pool.map(load_energy_data, donor_paths))
    else:
        features_allox = np.array([load_features(fp) for fp in feature_paths])
        data_out_accep = np.array([load_energy_data(fp) for fp in acceptor_paths])
        data_out_donor = np.array([load_energy_data(fp) for fp in donor_paths])

    features_allo_reshaped = features_allox.reshape(-1, n_features)
    data_out_log_accep = np.log(-data_out_accep.reshape(-1, 5))
    data_out_log_donor = np.log(-data_out_donor.reshape(-1, 5))
    
    D_numpy = features_allo_reshaped[:]

    # Select output type
    if output_type == "donor":
        E_numpy = data_out_log_donor[:, :num_outputs]  # First 1 or 2 columns
    elif output_type == "acceptor":
        E_numpy = data_out_log_accep[:, :num_outputs]  # First 1 or 2 columns
    elif output_type == "both":
        E_numpy = np.concatenate([
            data_out_log_donor[:, :num_outputs],  # First 1 or 2 columns
            data_out_log_accep[:, :num_outputs]   # First 1 or 2 columns
        ], axis=1)
    else:
        raise ValueError("Invalid output_type! Choose 'donor', 'acceptor', or 'both'.")

    # Standardize input data
    scaler = StandardScaler().fit(D_numpy)
    D_whole = scaler.transform(features_allo_reshaped)
    D_numpy = scaler.transform(D_numpy)
    print(D_numpy.shape, E_numpy.shape)
    
    return torch.Tensor(E_numpy), torch.Tensor(D_numpy)


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

def plot_energy_histogram(E_numpy_true, E_numpy_pred, num_bins=100, range_xax=(-30, 0), file_name="data_hist_true_pred.pdf"):
    """
    Plots a histogram of energy distributions for donor molecules.
    
    Parameters:
    - E_numpy_true (numpy.ndarray): Reshaped donor energy data, where each column represents different energy levels.
    - num_bins (int): Number of bins for the histogram.
    """

    colors = ["#d62728", "#1f77b4"]  # Red for 1st, Blue for 2nd

    fig, ax = plt.subplots()
    plt.rcParams["font.family"] = "Times New Roman"

    # Calculate weights to normalize histograms
    weights1 = np.ones(len(E_numpy_true[:, 0])) / len(E_numpy_true[:, 0])
    weights2 = np.ones(len(E_numpy_true[:, 1])) / len(E_numpy_true[:, 1])

    weights3 = np.ones(len(E_numpy_pred[:, 0])) / len(E_numpy_pred[:, 0])
    weights4 = np.ones(len(E_numpy_pred[:, 1])) / len(E_numpy_pred[:, 1])
    
    # Plot histograms
    ax.hist(E_numpy_true[:, 0], bins=num_bins, range=range_xax, 
            weights=weights1, alpha=0.8, label="1st, True", color=colors[0], linewidth=1.0, histtype='step', zorder=2)
    ax.hist(E_numpy_true[:, 1], bins=num_bins, range=range_xax, 
            weights=weights2, alpha=0.8, label="2nd, True", color=colors[1], linewidth=1.0, histtype='step', zorder=2)
    
    ax.hist(E_numpy_pred[:, 0], bins=num_bins, range=range_xax, 
            weights=weights3, alpha=0.4, label="1st, Prediction", color=colors[0], histtype='stepfilled', zorder=3)
    ax.hist(E_numpy_pred[:, 1], bins=num_bins, range=range_xax, 
            weights=weights4, alpha=0.4, label="2nd, Prediction", color=colors[1], histtype='stepfilled', zorder=3)
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

def correlation_plot(E_numpy, D_numpy, model):
    """Evaluates the model on the full dataset and generates a correlation plot."""
    from scipy.stats import gaussian_kde
    plt.rcParams['font.family'] = 'sans-serif'
    #rcParams['font.sans-serif'] = ['Tahoma']
    #plt.rcParams["font.family"] = "Times New Roman"
    
    model.load_state_dict(torch.load("best_model_donor.pt"))
    model.eval()
    
    with torch.no_grad():
        E_whole_pred = model(D_numpy).numpy() 

    print(E_whole_pred.shape)

    E_whole = E_numpy.numpy()

    qq_true, qq_pred = -np.exp(E_whole.flatten()), -np.exp(E_whole_pred.flatten())
 
    print("Minimum predicted energy:", qq_pred.min())
    print("Maximum predicted energy:", qq_pred.max())

    # Density computation
    xy = np.vstack([qq_true, qq_pred])
    print("shape xy", xy.shape)
    z = gaussian_kde(xy)(xy)  # density values

    # Sort by density for better plotting
    idx = z.argsort()
    qq_true, qq_pred, z = qq_true[idx], qq_pred[idx], z[idx]
    
    # Plot correlation
    plt.figure(figsize=(10, 10))
    sc = plt.scatter(qq_true, qq_pred, c=z, cmap='plasma', s=22, edgecolor='none', alpha=0.5, label="Predictions")
    #plt.scatter(qq_true, qq_pred, alpha=0.5, color='#1f77b4', label="Predictions")
    plt.plot(qq_true, qq_true, linestyle="-", color="#d62728", label="y=x")

    #plt.axhline(y=0, color='r', linestyle='-')
    #plt.axvline(x=0, color='r', linestyle='-')
    plt.xlim(-30, 0.5)
    plt.ylim(-30, 0.5)

    plt.xlabel('DFT energy (mHartree)', fontsize=22)
    plt.ylabel('NN energy (mHartree)', fontsize=22)

    cbar = plt.colorbar(sc, label='Density')
    # Change the font size of the label
    cbar.set_label('Density', fontsize=22)

    # Change the font size of the tick labels (the numbers)
    cbar.ax.tick_params(labelsize=18)

    plt.rcParams.update({'font.size': 22})
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc='upper left', frameon=False, fontsize=22)

    # Compute errors
    mae_energy = mean_absolute_error(qq_true, qq_pred)
    mse_energy = mean_squared_error(qq_true, qq_pred)
    rmse_energy = np.sqrt(mse_energy)
    
    print(f"RMSE: {rmse_energy:.3f} mHartree, MAE: {mae_energy:.3f} mHartree")

    plt.savefig("correlation.pdf", dpi=50)
    plt.savefig("correlation_2.png")
    
    return -np.exp(E_whole), -np.exp(E_whole_pred)

def main_eval(base_path):

    n_snapshot = 2000  
    n_samples = 125
    n_features = 952
    torch.manual_seed(123)

    input_size = n_features      # Number of input features
    hidden_sizes = [50]*2  # Sizes of the hidden layers
    output_size = 2      # Number of output features

    model = FFNet(input_size=n_features, hidden_layers=hidden_sizes, output_size=output_size, activation_func=nn.Tanh)

    E_numpy, D_numpy = data_loader(base_path, n_snapshot, n_samples, n_features, num_outputs=output_size, start_index=96000, end_index=100000)

    E_numpy_true, E_whole_pred = correlation_plot(E_numpy, D_numpy, model)

    plot_energy_histogram(E_numpy_true, E_whole_pred, num_bins=100, range_xax=(-30, 0), file_name="data_hist_true_pred.pdf")

if __name__ == "__main__":
    base_path="/home/tahmas41/work/ALMO_nn/Bulk_water_ALMO_karhan/data"
    main_eval(base_path)
