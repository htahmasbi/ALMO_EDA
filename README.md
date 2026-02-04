# ALMO EDA
This repository contains a PyTorch-based neural network designed to predict electron delocalization energies of water molecules. By leveraging chemical descriptors (SOAP) as inputs, the model bypasses computationally expensive DFT calculations to provide rapid estimates of delocalization energies.

## 📂 Project Structure

```text
.
├── configs/
│   └── base_config.yaml       # Hyperparameters & model settings
├── data/                      # Raw .npy files (git-ignored)
├── models/                    # Saved .pt checkpoints
├── src/                       # Source code
│   ├── __init__.py            # Makes src a Python package
│   ├── data_loader.py         # Data processing & Dataset class
│   ├── network.py             # PyTorch FFNet architecture
│   ├── loss.py                # Physics-informed Loss functions
│   └── trainer.py             # Training & Validation loops
├── main.py                    # Main entry point
├── requirements.txt           # Dependency list
└── README.md                  # Project documentation
```
## Getting Started

1. Install dependencies:
   `pip install -r requirements.txt`

2. Place your data in the `/data` folder.

3. Run the training:
   `python main.py`
