![CI CPU Test](https://github.com/htahmasbi/ALMO_EDA/actions/workflows/ci-cpu-tests.yml/badge.svg)
![Smoke Test](https://github.com/htahmasbi/ALMO_EDA/actions/workflows/smoke-test-training.yml/badge.svg)
![MOF Validation](https://github.com/htahmasbi/ALMO_EDA/actions/workflows/test-mof-validation.yml/badge.svg)
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
│   ├── base_config.py
│   ├── eval_config.py
│   └── mof_config.yaml
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
