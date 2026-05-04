import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from almo_eda.network import FFNet
from almo_eda.trainer import train_model, CustomLoss  
from almo_eda.data_loader import AtomisticDataset
from almo_eda.utils import time_research_task
from almo_eda.logger import get_logger

logger = get_logger("Optimization-Engine")

@time_research_task
def hyperparameter_optuna(D_tr, D_val, E_tr, E_val, config, device):
    """
    Main entry point for optimization. 
    Accepts loaded data and config to run the Optuna study.
    """
    opt_cfg = config.get('optimization', {})
    search_space = opt_cfg.get('search_space', {})
    n_trials = opt_cfg.get('n_trials', 10)
    
    # Static data params
    n_features = config['data']['n_features']
    n_outputs = config['data']['n_outputs']

    def objective(trial):
        # Suggest architecture
        num_layers = trial.suggest_categorical('num_layers', search_space['num_layers'])
        h_cfg = search_space['hidden_units']
        hidden_sizes = [
            trial.suggest_int(f'hidden_layer_{i+1}', h_cfg['min'], h_cfg['max'], step=h_cfg['step']) 
            for i in range(num_layers)
        ]

        # Suggest training params
        lr = trial.suggest_categorical('lr', search_space['learning_rates'])
        batch_size = trial.suggest_categorical('batch_size', search_space['batch_sizes'])
        act_name = trial.suggest_categorical('activation', search_space['activations'])

        model = FFNet(
            input_size=n_features,
            hidden_layers=hidden_sizes,
            output_size=n_outputs,
            activation=act_name, 
            dropout_prob=config['model'].get('dropout', 0.0)
        ).to(device)

        # Setup Trial Loaders
        train_loader = DataLoader(AtomisticDataset(D_tr, E_tr), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(AtomisticDataset(D_val, E_val), batch_size=batch_size)

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=config['training'].get('weight_decay', 0.001)
        )
        criterion = CustomLoss()

        # Train for a limited number of epochs per trial
        epochs = opt_cfg.get('epochs_per_trial', 5)
        
        try:
            # We only care about the final validation loss for Optuna
            _, valid_losses = train_model(
                model, optimizer, train_loader, valid_loader, 
                criterion, device, epochs
            )
            return valid_losses[-1] # Return the last epoch's validation loss
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            raise optuna.exceptions.TrialPruned()

    # Run the study
    study = optuna.create_study(
            direction="minimize", 
            sampler=optuna.samplers.TPESampler()
            )
    study.optimize(objective, n_trials=n_trials)
    
    return study
