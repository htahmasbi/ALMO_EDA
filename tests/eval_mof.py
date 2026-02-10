import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Professional Imports from your new structure
from src.network import FFNet
from src.data_loader import data_loader  # Ensure this matches the modular version
from src.logger import get_logger

# Initialize modular logger
logger = get_logger("SystemEvaluation")

class SystemAnalyzer:
    def __init__(self, model_path, config, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = self._setup_model(model_path)
        logger.info(f"Analyzer initialized with model: {model_path} on {self.device}")

    def _setup_model(self, model_path):
        """Initializes the FFNet and loads pre-trained weights."""
        model = FFNet(
            input_size=self.config['n_features'],
            hidden_layers=self.config['hidden_sizes'],
            output_size=self.config['output_size'],
            activation_func=torch.nn.Tanh  #
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_system(self, base_path, n_snapshots, n_samples):
        """Loads data and runs inference for a specific molecular system."""
        logger.info(f"Loading data from: {base_path}")
        
        # Using the standardized data_loader we built previously
        # Note: We pass start_index=0 and end_index=n_snapshots to match your loop
        try:
            # Reusing the modular data_loader logic
            # Adjust the loader parameters to match the C2W2 system logic
            # For pure prediction, we only need the feature tensor D
            D_whole = self._load_prediction_data(base_path, n_snapshots, n_samples)
            
            with torch.no_grad():
                D_tensor = torch.Tensor(D_whole).to(self.device)
                E_pred_log = self.model(D_tensor).cpu().numpy()
            
            # Reverse log transformation: energy = -exp(log_energy)
            return -np.exp(E_pred_log)
        except Exception as e:
            logger.error(f"Error during system prediction: {e}")
            return None

    def _load_prediction_data(self, base_path, n_snapshots, n_samples):
        """Internal helper to load and standardize features."""
        features_list = []
        for i in range(n_snapshots):
            # Matches your path logic: {system_typ}_coord_{i}_modified_soap_n8l6c5.npy
            # Note: You may need to adjust the filename pattern if it differs per system
            file_name = [f for f in os.listdir(base_path) if f.endswith(f"coord_{i}_modified_soap_n8l6c5.npy")]
            if file_name:
                features_list.append(np.load(os.path.join(base_path, file_name[0])))
        
        features_allox = np.stack(features_list)
        features_reshaped = np.reshape(features_allox, (n_snapshots * n_samples, self.config['n_features']))
        
        # Apply standardization as done in training
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(features_reshaped)
        return scaler.transform(features_reshaped)

    def plot_results(self, E_pred, system_name, num_bins=100, ylim_top=0.25):
        """Generates normalized histograms for the predicted energies."""
        colors = ["#d62728", "#1f77b4"]
        fig, ax = plt.subplots()
        
        weights1 = np.ones(len(E_pred[:, 0])) / len(E_pred[:, 0])
        weights2 = np.ones(len(E_pred[:, 1])) / len(E_pred[:, 1])

        ax.hist(E_pred[:, 0], bins=num_bins, range=(-30, 0), weights=weights1, 
                alpha=0.5, label="1st", color=colors[0], histtype='stepfilled')
        ax.hist(E_pred[:, 1], bins=num_bins, range=(-30, 0), weights=weights2, 
                alpha=0.5, label="2nd", color=colors[1], histtype='stepfilled')
        
        ax.set_xlabel('Energy (mHartree)', fontsize=15)
        ax.set_ylabel('Normalized Frequency', fontsize=15)
        ax.set_ylim(0.00, ylim_top)
        ax.legend(loc='upper left', frameon=False)
        
        plt.tight_layout()
        output_file = f"EDA_{system_name}.pdf"
        plt.savefig(output_file)
        logger.info(f"Plot saved to {output_file}")
        plt.close()

if __name__ == "__main__":
    # Shared Configuration
    config = {
        'n_features': 952,
        'hidden_sizes': [50, 50],
        'output_size': 2,
        'activation': 'Tanh'
    }

    # Systems to evaluate
    systems = ["Zn2W2", "Zn2W5", "Zn2W7", "Cu2W2", "Cu2W5"]
    samples = [72, 183, 258, 72, 174]

    analyzer = SystemAnalyzer(model_path="best_model_donor.pt", config=config)

    for sys_name, n_s in zip(systems, samples):
        path = f"/home/tahmas41/work/ALMO_nn/Bulk_water_ALMO_karhan/data_CPO-27/{sys_name}"
        E_pred = analyzer.predict_system(path, n_snapshots=1000, n_samples=n_s)
        
        if E_pred is not None:
            analyzer.plot_results(E_pred, sys_name)
