![CPU Test Inference](https://github.com/htahmasbi/ALMO_EDA/actions/workflows/ci-cpu-tests.yml/badge.svg)
![Smoke Test](https://github.com/htahmasbi/ALMO_EDA/actions/workflows/smoke-test-training.yml/badge.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
# ALMO EDA
This repository contains a PyTorch-based neural network designed to predict electron delocalization energies of water molecules. By leveraging chemical descriptors (SOAP) as inputs, the model bypasses computationally expensive DFT calculations to provide rapid estimates of delocalization energies.

## 📂 Project Structure

```text
.
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── LICENSE
├── pyproject.toml
│
├── configs
│   ├── mof_config.yaml
│   ├── inference_config.yaml
│   └── train_config.yaml
├── data
│   └── soap_descriptor.py
├── models
│   └── best_model_donor.pt
├── examples
│   ├── run_mof.py
│   ├── run_inference.py
│   ├── run_optuna.py
│   └── run_training.py
├── tests
│   ├── test_dataset.py
│   ├── test_loss.py 
│   ├── test_network.py
│   └── test_training_smoke.py 
└── almo_eda
    ├── __init__.py
    ├── data_loader.py
    ├── logger.py
    ├── network.py
    ├── utils.py
    ├── visualization.py
    ├── optimization.py
    └── trainer.py
```

## Installation

```bash
git clone https://github.com/htahmasbi/ALMO_EDA.git
python -m pip install -e ".[dev]"
```
or
```bash
python -m pip install -r requirements.txt
``` 
## Data Description

 - Inputs: Descriptors derived from Cartesian ($XYZ$) coordinates of water molecules, capturing local geometric orientations.
 - Target: Electron Delocalization Energy ($E$), calculated via the ALMO-EDA scheme implemented in CP2K.

## Run examples

1. Place your data in the `/data` folder.
2. Modify and run the example scripts:
   `python examples/run_training.py`

## Citation 
If you publish work that uses or mentions this code, please cite the following paper:

```bibtex
@article{Tahmasbi2025,
  title = {Scalable machine learning model for energy decomposition analysis in aqueous systems},
  volume = {163},
  ISSN = {1089-7690},
  url = {http://dx.doi.org/10.1063/5.0303825},
  DOI = {10.1063/5.0303825},
  number = {21},
  journal = {The Journal of Chemical Physics},
  publisher = {AIP Publishing},
  author = {Tahmasbi,  Hossein and Beerbaum,  Michael and Brzoza,  Bartosz and Cangi,  Attila and K\"{u}hne,  Thomas D.},
  year = {2025},
  pages = {214115},
  month = dec 
}
```
