import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from src.utils import time_research_task

class AtomisticDataset(Dataset):
    def __init__(self, features, energies):
        assert len(features) == len(energies), "Mismatch in dataset size"
        self.features = torch.tensor(features, dtype=torch.float32)
        self.energies = torch.tensor(energies, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.energies[idx]

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

@time_research_task
def data_loader(base_path, n_snapshot, n_samples, n_features, mode="train", #scalar=None, 
                output_type="donor", n_outputs = 2,
                start_index=90000, end_index=100000, step=2,
                random_seed=123, valid_size=0.2, num_train_samples = 2000,
                use_multiprocessing=True):
    """
    Unified loader for both Training and Post-Processing.
    
    Args:
        mode (str): "train" returns (D_train, D_valid, E_train, E_valid)
                    "eval" returns (D_tensor, E_tensor)
        scaler: If mode="eval", you MUST pass the scaler fitted during training.

    Load molecular features and energy values, preprocess data, and return train/validation tensors.

    Parameters:
    - n_snapshot (int): Number of snapshots (timesteps).
    - n_samples (int): Number of samples per snapshot.
    - n_features (int): Number of feature dimensions.
    - output_type (str): Choose "donor", "acceptor", or "both".
    - n_outputs (int): Number of outputs to return (1 or 2), taken from first or first two energy columns.
    - base_path (str): Base directory for data files.
    - start_index (int): Start index for file names.
    - end_index (int): End index for file names.
    - step (int): Step size for indexing (default: 2).
    - num_train_samples (int): Number of training samples.
    - valid_size (float): Fraction of data to use for validation.
    - random_seed (int): Random seed for reproducibility.
    - use_multiprocessing (bool): Use multiprocessing for faster file loading.

    Returns:
    - n_features (int): Number of input features.
    - D_train (Tensor): Training feature set.
    - D_valid (Tensor): Validation feature set.
    - E_train (Tensor): Training energy values.
    - E_valid (Tensor): Validation energy values.
    """
    
    assert n_outputs in [1, 2], "n_outputs must be 1 or 2 (selects first or first two energy values)."

    # Prepare file indices
    file_indices = range(start_index, end_index, step)

    # ==========================
    # Load Features
    # ==========================
    features_allox = np.zeros((n_snapshot, n_samples, n_features))
    file_paths = [os.path.join(base_path, f"0{idx}/coord_soap_nmax8_lmax6_cut5.npy") for idx in file_indices]

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
    acceptor_paths = [os.path.join(base_path, f"0{idx}/molecules.lowest.acceptor") for idx in file_indices]
    donor_paths = [os.path.join(base_path, f"0{idx}/molecules.lowest.donor") for idx in file_indices]

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

    #plot_energy_histogram(data_out_donor_reshaped, num_bins=100, range_xax=(-30, 0), file_name="out_data_hist_donor.pdf")
    #plot_energy_histogram(data_out_accep_reshaped, num_bins=100, range_xax=(-30, 0), file_name="out_data_hist_accep.pdf")

    # Convert to log scale
    data_out_log_accep = np.log(-data_out_accep_reshaped)
    data_out_log_donor = np.log(-data_out_donor_reshaped)

    #plot_energy_histogram(data_out_log_donor, num_bins=100, range_xax=(-4, 4), file_name="out_data_hist_donor_log.pdf")
    #plot_energy_histogram(data_out_log_accep, num_bins=100, range_xax=(-4, 4), file_name="out_data_hist_accep_log.pdf")
    # ==========================
    # Select Training Data
    # ==========================
    
    if mode == "train":
        num_samples = num_train_samples
        np.random.seed(random_seed)
        idx = np.random.choice(len(features_allo_reshaped), num_samples, replace=False)
        D_raw = features_allo_reshaped[idx]

        if output_type == "donor":
            E_raw = data_out_log_donor[idx, :n_outputs]
        elif output_type == "acceptor":
            E_raw = data_out_log_accep[idx, :n_outputs]
        elif output_type == "both":
            E_raw = np.concatenate([
                data_out_log_donor[idx, :n_outputs],
                data_out_log_accep[idx, :n_outputs]
            ] , axis=1 )
        else:
            raise ValueError("Invalid output_type! Choose 'donor', 'acceptor', or 'both'.")

        #var_energy_train = E_raw.var()

        # Split
        D_tr_raw, D_va_raw, E_train, E_valid = train_test_split(
            D_raw, E_raw, test_size=valid_size, random_state=random_seed
        )

        print(f"Shapes -> D_train: {D_train.shape}, D_valid: {D_valid.shape}, E_train: {E_train.shape}, E_valid: {E_valid.shape}")

        # FIT SCALER HERE
        scaler = StandardScaler().fit(D_tr_raw)
        D_train = scaler.transform(D_tr_raw)
        D_valid = scaler.transform(D_va_raw)

        return torch.Tensor(D_train), torch.Tensor(D_valid), \
               torch.Tensor(E_train), torch.Tensor(E_valid), scaler

    elif mode == "eval":
        #if scaler is None:
        #    raise ValueError("You must provide the training scaler for evaluation!")

        ## Transform using the SAVED scaler, do NOT .fit()
        #D_eval = scaler.transform(features_allo_reshaped)
        #E_eval = data_out_log_donor[:, :kwargs.get('n_outputs', 2)]
        # Standardize input data
      
        scaler = StandardScaler().fit(features_allo_reshaped)
        D_eval = scaler.transform(features_allo_reshaped)
        if output_type == "donor":
            E_eval = data_out_log_donor[:, :n_outputs]
        elif output_type == "acceptor":
            E_eval = data_out_log_accep[:, :n_outputs]
        elif output_type == "both":
            E_eval = np.concatenate([
                data_out_log_donor[:, :n_outputs],
                data_out_log_accep[:, :n_outputs]
            ] , axis=1 )
        else:
            raise ValueError("Invalid output_type! Choose 'donor', 'acceptor', or 'both'.")

        print(D_eval.shape, E_eval.shape)

        return torch.Tensor(D_eval), torch.Tensor(E_eval)
