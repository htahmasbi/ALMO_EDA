import yaml
import torch
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.network import FFNet 
from src.data_loader import data_loader 
from src.visualization import energy_histogram_plot, correlation_plot
from src.logger import get_logger

# Initialize logger for this specific module
logger = get_logger("Evaluation-CI test")

def main():
    logger.info(f"Initializing evaluation for our model")

    # Load configuration
    with open("configs/inference_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # 1. Load Data using your existing data_loader logic
    try:
        D_test, E_test = data_loader(
                **config['data']
        )

        logger.info("Successfully loaded test dataset.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 2. Setup Device and Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FFNet(
        input_size=config['data']['n_features'],
        hidden_layers=config['model']['hidden_sizes'],
        output_size=config['model']['output_size'],
        activation=config['model']['activation'], 
        dropout_prob=config['model']['dropout']
    ).to(device)
    
    # Load the best model weights
    model.load_state_dict(torch.load(config['model']['model_path'], map_location=device))
    model.eval()

    # 3. Inference
    with torch.no_grad():
        #X = torch.tensor(D_test, dtype=torch.float32).to(device)
        E_pred_log = model(D_test).cpu().numpy()

    # 4. Physical Unit Recovery
    # Reversing the log(-data) transformation used in data_loader
    E_true_mh = -np.exp(E_test)
    E_pred_mh = -np.exp(E_pred_log)

    energy_histogram_plot(E_true_mh, E_pred_mh, file_name="ci_test_histogram.pdf")
    correlation_plot(E_true_mh, E_pred_mh, file_name="ci_correlation.png")

    # 5. Metrics & Logging Results
    mae = mean_absolute_error(E_true_mh, E_pred_mh)
    rmse = np.sqrt(mean_squared_error(E_true_mh, E_pred_mh))

    logger.info("Evaluation Results:")
    logger.info(f"MAE:  {mae:.4f} mHartree")
    logger.info(f"RMSE: {rmse:.4f} mHartree")
    
    return mae, rmse

if __name__ == "__main__":

    # Check if we are running on GitHub Actions
    is_ci = os.environ.get('GITHUB_ACTIONS') == 'true'
    logger.info(f"Running in CI mode: {is_ci}")
    main()
