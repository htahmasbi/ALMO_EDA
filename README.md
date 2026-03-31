![Evaluation Tests](https://github.com/htahmasbi/ALMO_EDA/actions/workflows/eval-tests.yml/badge.svg)
![CPU Tests](https://github.com/htahmasbi/ALMO_EDA/actions/workflows/train-tests.yml/badge.svg)
![MOF Tests](https://github.com/htahmasbi/ALMO_EDA/actions/workflows/eval-mof.yml/badge.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
# ALMO EDA
This repository contains a PyTorch-based neural network designed to predict electron delocalization energies of water molecules. By leveraging chemical descriptors (SOAP) as inputs, the model bypasses computationally expensive DFT calculations to provide rapid estimates of delocalization energies.

## 📂 Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs
│   └── base_config.yaml
├── data
│   └── soap_descriptor.py
├── models
│   └── best_model_donor.pt
├── tests
│   ├── eval_mof.py
│   ├── eval_test.py
│   └── eval_train.py
└── src
    ├── __init__.py
    ├── data_loader.py
    ├── logger.py
    ├── network.py
    ├── utils.py
    ├── visualization.py
    └── trainer.py
```

## Getting Started

1. Install dependencies:
   `pip install -r requirements.txt`

2. Place your data in the `/data` folder.

3. Run the training:
   `python tests/eval_train.py`
