import yaml
import torch
from src.data_loader import data_loader, AtomisticDataset
from src.network import FFNet
from src.trainer import train_model, CustomLoss
from torch.utils.data import DataLoader

def main():
    # Load configuration
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Data using your existing data_loader logic
    n_feat, var_energy, D_tr, D_val, E_tr, E_val = data_loader(
        **config['data'] # Unpacks dictionary keys as arguments
    )

    # 2. Setup Device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Create Datasets and Loaders
    train_loader = DataLoader(AtomisticDataset(D_tr, E_tr), batch_size=config['training']['batch_size'], shuffle=True)
    valid_loader = DataLoader(AtomisticDataset(D_val, E_val), batch_size=config['training']['batch_size'])

    # 4. Initialize Model
    model = FFNet(
        input_size=n_feat,
        hidden_layers=config['model']['hidden_sizes'],
        output_size=config['data']['num_outputs'],
        activation=config['model']['activation']
    ).to(device)

    # 5. Loss and Optimizer
    criterion = CustomLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(config['training']['lr']), 
        weight_decay=config['training']['weight_decay']
    )

    # 6. Run Training
    train_model(model, optimizer, train_loader, valid_loader, criterion, device, config['training']['epochs'])

if __name__ == "__main__":
    main()
