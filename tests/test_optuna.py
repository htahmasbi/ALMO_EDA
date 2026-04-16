import yaml
import torch
from src.data_loader import data_loader
from src.optimization import hyperparameter_optuna
from src.logger import get_logger

logger = get_logger("Optimization-CI CPU test")

def main():
    logger.info("Starting Optimization CI Test...")

    # Load configuration
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Data
    try:
        D_tr, D_val, E_tr, E_val = data_loader(**config['data'])
        logger.info("Successfully loaded test dataset for Optuna.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        exit(1)

    device = torch.device('cpu')

    # Run Optimization Engine
    logger.info("Triggering Optuna Search Engine...")
    try:
        study = hyperparameter_optuna(D_tr, D_val, E_tr, E_val, config, device)
        
        logger.info(f"CI Test Passed. Best Trial Value: {study.best_value}")
        logger.info(f"Best Params: {study.best_params}")
    except Exception as e:
        logger.error(f"Optimization pipeline crashed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
