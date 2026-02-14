import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde

# Import your classes from your new package structure
from src.network import FFNet
from src.data_loader import data_loader

class ModelEvaluator:
    def __init__(self, config, model_path, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Initializes and loads the weights into the architecture."""
        model = FFNet(
            input_size=self.config['n_features'],
            hidden_layers=self.config['hidden_sizes'],
            output_size=self.config['output_size'],
            activation=self.config['activation']
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def get_predictions(self, D_tensor, E_tensor):
        """Runs inference and converts log-scale back to mHartree."""
        with torch.no_grad():
            D_tensor = D_tensor.to(self.device)
            E_pred_log = self.model(D_tensor).cpu().numpy()
        
        E_true_log = E_tensor.numpy()
        
        # Convert back from log scale: -exp(log(-E))
        # Note: In your loader you used log(-data), so we reverse it here
        y_true = -np.exp(E_true_log)
        y_pred = -np.exp(E_pred_log)
        
        return y_true, y_pred

    def calculate_metrics(self, y_true, y_pred):
        """Computes RMSE and MAE for the delocalization energy."""
        # Flattening to handle multi-output (1st and 2nd energy levels)
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        
        print(f"--- Evaluation Metrics ---")
        print(f"MAE:  {mae:.4f} mHartree")
        print(f"RMSE: {rmse:.4f} mHartree")
        return {"mae": mae, "rmse": rmse}

    def plot_correlation(self, y_true, y_pred, save_path="correlation_plot.png"):
        """Generates a density-colored scatter plot."""
        t, p = y_true.flatten(), y_pred.flatten()
        
        # Calculate point density
        xy = np.vstack([t, p])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        t, p, z = t[idx], p[idx], z[idx]

        plt.figure(figsize=(8, 7))
        plt.scatter(t, p, c=z, s=20, cmap='plasma', alpha=0.5)
        plt.plot([t.min(), t.max()], [t.min(), t.max()], 'r--') # Identity line
        plt.xlabel("DFT Energy (mHartree)")
        plt.ylabel("NN Predicted Energy (mHartree)")
        plt.title("Delocalization Energy Correlation")
        plt.colorbar(label='Density')
        plt.savefig(save_path)
        plt.close()

# Example Usage Entry Point
if __name__ == "__main__":
    # 1. Define evaluation settings (ideally load this from configs/eval_config.yaml)
    eval_config = {
        'n_features': 952,
        'hidden_sizes': [50, 50],
        'output_size': 2,
        'activation': "Tanh"
    }
    
    # 2. Load the test data (Indices not used in training)
    # Using your existing data_loader from src
    E_tensor, D_tensor = data_loader(
        base_path="../data/Bulk_water_ALMO_karhan",
        n_snapshot=2000,
        n_samples=125,
        n_features=952,
        start_index=96000,
        end_index=100000
    )

    # 3. Initialize and Run
    tester = ModelEvaluator(eval_config, model_path="../models/best_model_donor.pt")
    y_true, y_pred = tester.get_predictions(D_tensor, E_tensor)
    tester.calculate_metrics(y_true, y_pred)
    tester.plot_correlation(y_true, y_pred)
