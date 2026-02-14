import torch
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Professional Imports
from src.network import FFNet 
from src.data_loader import data_loader 
from src.logger import get_logger  # New centralized logger

# Initialize logger for this specific module
logger = get_logger("Evaluation-CI test")

def run_test_case(model_path, config):
    logger.info(f"Initializing evaluation for: {model_path}")

    # 1. Load Data using your existing data_loader logic
    try:
        _, _, _, D_test, _, E_test = data_loader(
            n_snapshot=config['n_snapshot'],
            n_samples=config['n_samples'],
            n_features=config['n_features'],
            start_index=config['start_index'],
            end_index=config['end_index'],
            num_train_samples=config['num_test_samples']
        )
        logger.info("Successfully loaded test dataset.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 2. Setup Device and Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FFNet(
        input_size=config['n_features'],
        hidden_layers=config['hidden_sizes'],
        output_size=config['output_size'],
        activation=torch.nn.Tanh # Matches your FFNet implementation
    ).to(device)
    
    # Load the best model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference
    with torch.no_grad():
        X = torch.tensor(D_test, dtype=torch.float32).to(device)
        y_pred_log = model(X).cpu().numpy()

    # 4. Physical Unit Recovery
    # Reversing the log(-data) transformation used in data_loader
    y_true_mh = -np.exp(E_test)
    y_pred_mh = -np.exp(y_pred_log)

    # 5. Metrics & Logging Results
    mae = mean_absolute_error(y_true_mh, y_pred_mh)
    rmse = np.sqrt(mean_squared_error(y_true_mh, y_pred_mh))

    logger.info("Evaluation Results:")
    logger.info(f"MAE:  {mae:.4f} mHartree")
    logger.info(f"RMSE: {rmse:.4f} mHartree")
    
    return mae, rmse

if __name__ == "__main__":

    # Check if we are running on GitHub Actions
    is_ci = os.environ.get('GITHUB_ACTIONS') == 'true'
    
    test_config = {
        'n_snapshot': 5 if is_ci else 2000, # Tiny sample for CI
        'n_samples': 125,
        'n_features': 952,
        'hidden_sizes': [50, 50],
        'output_size': 2,
        'start_index': 90000,
        'end_index': 90010 if is_ci else 94000,
        'num_test_samples': 625 if is_ci else 250000
    }
    
    logger.info(f"Running in CI mode: {is_ci}")
    run_test_case("./models/best_model_donor.pt", test_config)
