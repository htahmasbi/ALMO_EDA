import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from almo_eda.visualization import energy_histogram
from almo_eda.network import FFNet
from almo_eda.data_loader import data_loader_mof
from almo_eda.logger import get_logger

# Initialize modular logger
logger = get_logger("System Evaluation MOF")


def main():
    logger.info(f"Initializing evaluation for our model")

    # Load configuration
    with open("configs/mof_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Data using your existing data_loader logic
    try:
        D_test = data_loader_mof(**config["data"])

        logger.info("Successfully loaded test dataset.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Setup Device and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFNet(
        input_size=config["data"]["n_features"],
        hidden_layers=config["model"]["hidden_sizes"],
        output_size=config["model"]["output_size"],
        activation=config["model"]["activation"],
        dropout_prob=config["model"]["dropout"],
    ).to(device)

    # Load the best model weights
    model.load_state_dict(torch.load(config["model"]["model_path"], map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        # X = torch.tensor(D_test, dtype=torch.float32).to(device)
        E_pred_log = model(D_test).cpu().numpy()

    E_pred_mh = -np.exp(E_pred_log)
    energy_histogram(E_pred_mh, file_name="ci_histogram_mof.pdf")


if __name__ == "__main__":
    main()
