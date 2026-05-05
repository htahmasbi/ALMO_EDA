import yaml
import torch
from torch.utils.data import DataLoader
from almo_eda.data_loader import data_loader, AtomisticDataset
from almo_eda.network import FFNet
from almo_eda.trainer import train_model, CustomLoss
from almo_eda.visualization import loss_plot
from almo_eda.logger import get_logger

logger = get_logger("Training-CI CPU test")

def main():
    logger.info(f"Loading config file...")

    # Load configuration
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Data using the data_loader logic
    try:
        D_tr, D_val, E_tr, E_val = data_loader(
        **config['data']
        )
        logger.info(f"Successfully loaded test dataset.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Setup Device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create Datasets and Loaders
    logger.info(f"Creating train and validation datasets")
    train_loader = DataLoader(AtomisticDataset(D_tr, E_tr), batch_size=config['training']['batch_size'], shuffle=True)
    valid_loader = DataLoader(AtomisticDataset(D_val, E_val), batch_size=config['training']['batch_size'])

    # Initialize Model
    logger.info(f"Initializing model")
    model = FFNet(
        input_size=config['data']['n_features'],
        hidden_layers=config['model']['hidden_sizes'],
        output_size=config['model']['output_size'],
        activation=config['model']['activation'],
        dropout_prob=config['model']['dropout']
    ).to(device)

    # Loss and Optimizer
    criterion = CustomLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(config['training']['lr']), 
        weight_decay=config['training']['weight_decay']
    )

    # Run Training
    logger.info(f"Training started")
    train_losses, valid_losses = train_model(model, 
            optimizer, 
            train_loader, 
            valid_loader, 
            criterion, 
            device, config['training']['epochs'], 
            checkpoint_path=config['training']['checkpoint_path']
            )
    loss_plot(train_losses, valid_losses, file_name="tv_loss.pdf")

if __name__ == "__main__":
    main()
