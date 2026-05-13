# Code Review Improvements - Migration Guide

This document provides guidance for migrating existing code to work with the improvements made in this PR.

## 🔴 Breaking Changes

### 1. Data Loader - Function Signature Change

**Old Code:**
```python
D_train, D_valid, E_train, E_valid = data_loader(
    data_path="data/",
    n_snapshots=100,
    n_samples=500,
    n_features=384,
    mode="train"
)
```

**New Code (Required):**
```python
D_train, D_valid, E_train, E_valid, scaler = data_loader(
    data_path="data/",
    n_snapshots=100,
    n_samples=500,
    n_features=384,
    mode="train"
)
# Save or store the scaler for later use in eval mode
```

### 2. Data Loader - Evaluation Mode

**Old Code (Incorrect - created new scaler):**
```python
D_eval, E_eval = data_loader(..., mode="eval")
```

**New Code (Correct - uses training scaler):**
```python
# First, run training to get the scaler
D_train, D_valid, E_train, E_valid, scaler = data_loader(
    ..., mode="train"
)

# Save scaler if needed
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Later, use the scaler for evaluation
D_eval, E_eval = data_loader(
    ..., mode="eval", scaler=scaler
)

# Or load it from disk
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
D_eval, E_eval = data_loader(..., mode="eval", scaler=scaler)
```

### 3. Train Model - Gradient Clipping

**Old Code:**
```python
train_losses, valid_losses = train_model(
    model, optimizer, train_loader, val_loader,
    criterion, device, num_epochs=100
)
```

**New Code (with optional gradient clipping):**
```python
# Default gradient_clip=1.0 (recommended for stability)
train_losses, valid_losses = train_model(
    model, optimizer, train_loader, val_loader,
    criterion, device, num_epochs=100,
    gradient_clip=1.0  # Can be disabled with gradient_clip=0.0
)
```

### 4. Visualization - Configurable Parameters

**Old Code:**
```python
energy_histogram(y_pred)  # Used hardcoded ylim=(0, 0.25)
correlation_plot(y_true, y_pred)  # Used hardcoded xlim/ylim
loss_plot(train_losses, valid_losses)  # Used hardcoded marker_interval=50
```

**New Code (with customization options):**
```python
# Customize visualization parameters
energy_histogram(
    y_pred,
    file_name="output.pdf",
    num_bins=100,
    range_xax=(-30, 0),
    ylim=(0, 0.30)  # NEW: Customize y-axis limits
)

correlation_plot(
    y_true, y_pred,
    file_name="corr.png",
    xlim=(-30, 0.5),  # NEW: Customize x-axis
    ylim=(-30, 0.5),  # NEW: Customize y-axis
    marker_interval=50  # NEW: Customize marker interval
)

loss_plot(
    train_losses, valid_losses,
    file_name="loss.pdf",
    marker_interval=50  # NEW: Customize marker interval
)
```

---

## 📝 Complete Migration Example

Here's a complete example showing how to update a training script:

### Before (Old Code):
```python
from almo_eda.data_loader import data_loader
from almo_eda.network import FFNet
from almo_eda.trainer import train_model, CustomLoss
from almo_eda.visualization import loss_plot
import torch

# Load data
D_train, D_valid, E_train, E_valid = data_loader(
    data_path="data/",
    n_snapshots=100,
    n_samples=500,
    n_features=384,
    mode="train"
)

# Create model
model = FFNet(
    input_size=384,
    hidden_layers=[128, 64],
    output_size=2,
    activation="ReLU"
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = CustomLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses, valid_losses = train_model(
    model, optimizer, train_loader, val_loader,
    criterion, device, num_epochs=100
)

# Visualize
loss_plot(train_losses, valid_losses)
```

### After (New Code - With All Improvements):
```python
from almo_eda.data_loader import data_loader, AtomisticDataset
from almo_eda.network import FFNet
from almo_eda.trainer import train_model, CustomLoss
from almo_eda.visualization import loss_plot
from almo_eda.logger import get_logger
import torch
from torch.utils.data import DataLoader

logger = get_logger(__name__)

# Load data with scaler
D_train, D_valid, E_train, E_valid, scaler = data_loader(
    data_path="data/",
    n_snapshots=100,
    n_samples=500,
    n_features=384,
    mode="train"
)
logger.info("Data loaded successfully")

# Create data loaders
train_dataset = AtomisticDataset(D_train, E_train)
val_dataset = AtomisticDataset(D_valid, E_valid)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create model with new weight initialization
model = FFNet(
    input_size=384,
    hidden_layers=[128, 64],
    output_size=2,
    activation="ReLU",
    dropout_prob=0.2
).to("cuda" if torch.cuda.is_available() else "cpu")

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = CustomLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train with gradient clipping
train_losses, valid_losses = train_model(
    model, optimizer, train_loader, val_loader,
    criterion, device, num_epochs=100,
    checkpoint_path="models/best_model.pt",
    gradient_clip=1.0  # NEW: Gradient clipping for stability
)

# Visualize with custom parameters
loss_plot(
    train_losses, valid_losses,
    file_name="training_loss.pdf",
    marker_interval=50
)

# Save scaler for later evaluation
import pickle
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

logger.info("Training complete!")
```

---

## 🔧 Configuration File Updates

If using YAML configuration files, update them to match the new structure:

### Before (Old Config):
```yaml
data:
  n_features: 384
  n_outputs: 2

model:
  type: FFNet
  hidden_layers: [128, 64]
  activation: ReLU
  dropout: 0.2

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### After (New Config - Required for Optimization):
```yaml
data:
  n_features: 384
  n_outputs: 2

model:
  type: FFNet
  activation: ReLU
  dropout: 0.2

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.001

optimization:
  n_trials: 20
  epochs_per_trial: 5
  search_space:
    num_layers: [2, 3, 4]
    hidden_units:
      min: 32
      max: 256
      step: 32
    learning_rates: [0.0001, 0.001, 0.01]
    batch_sizes: [32, 64, 128]
    activations: [ReLU, Tanh]
```

---

## ✅ Testing the Migration

After updating your code, verify everything works:

```python
import pytest
import torch
from almo_eda.data_loader import data_loader
from almo_eda.network import FFNet

def test_scaler_return():
    """Verify data_loader returns scaler in train mode."""
    result = data_loader(..., mode="train")
    assert len(result) == 5, "Expected 5-tuple (D_train, D_valid, E_train, E_valid, scaler)"
    D_train, D_valid, E_train, E_valid, scaler = result
    assert scaler is not None

def test_eval_requires_scaler():
    """Verify data_loader requires scaler in eval mode."""
    with pytest.raises(ValueError, match="scaler"):
        data_loader(..., mode="eval", scaler=None)

def test_network_initialization():
    """Verify FFNet weights are properly initialized."""
    model = FFNet(100, [64, 32], 2)
    # Check that weights were initialized (not zeros)
    first_layer_weights = list(model.parameters())[0]
    assert not torch.allclose(first_layer_weights, torch.zeros_like(first_layer_weights))

def test_gradient_clipping():
    """Verify gradient clipping works."""
    model = FFNet(100, [64, 32], 2)
    # This should not raise an error
    result = train_model(
        model, optimizer, train_loader, val_loader,
        criterion, device, num_epochs=1, gradient_clip=1.0
    )
    assert result is not None
```

---

## 📞 Troubleshooting

### Issue: "You must provide the training scaler for evaluation mode!"

**Solution:** You forgot to pass the scaler from training to evaluation:
```python
# Train and get scaler
D_train, D_valid, E_train, E_valid, scaler = data_loader(..., mode="train")

# Use scaler in eval
D_eval, E_eval = data_loader(..., mode="eval", scaler=scaler)
```

### Issue: "Config missing required top-level keys"

**Solution:** Your config is incomplete. Ensure all required sections are present:
```python
required_keys = {"data", "model", "training", "optimization"}
```

### Issue: Model not converging after update

**Cause:** Weight initialization changed. This is usually a good thing, but if you want the old behavior:
```python
# Disable automatic initialization by not calling _init_weights()
# (Not recommended - new initialization is better)
```

**Better:** Adjust hyperparameters (learning rate, batch size) with the new initialization.

---

## ✨ New Features to Leverage

Now that the code is improved, consider using these new features:

1. **Automatic Checkpoint Saving:**
   ```python
   train_losses, valid_losses = train_model(
       ..., checkpoint_path="models/best_model.pt"
   )
   ```

2. **Hyperparameter Optimization:**
   ```python
   from almo_eda.optimization import hyperparameter_optuna
   study = hyperparameter_optuna(D_tr, D_val, E_tr, E_val, config, device)
   best_params = study.best_params
   ```

3. **Better Logging:**
   ```python
   from almo_eda.logger import get_logger
   logger = get_logger("MyModule")
   logger.info("Event occurred")  # Logs to both file and console
   ```

4. **Configurable Visualizations:**
   ```python
   loss_plot(..., marker_interval=100)
   correlation_plot(..., xlim=(-25, 1), ylim=(-25, 1))
   ```
