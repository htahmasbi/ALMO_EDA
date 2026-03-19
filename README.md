# ALMO EDA
This repository contains a PyTorch-based neural network designed to predict electron delocalization energies of water molecules. By leveraging chemical descriptors (SOAP) as inputs, the model bypasses computationally expensive DFT calculations to provide rapid estimates of delocalization energies.

## 📂 Project Structure

```text
.
├── configs
│   └── base_config.yaml
├── data
│   └── soap_descriptor.py
└── tests
│   ├── eval_model.py
│   ├── eval_mof.py
│   ├── eval_test.py
│   ├── eval_train.py
│   ├── post_processing_C2W2.py
│   ├── post_processing.py
│   └── run_train.py
│
├── models
│   └── best_model_donor.pt
├── README.md
├── requirements.txt
├── .gitignore
└── src
    ├── data_loader.py
    ├── __init__.py
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
   `python main.py`
