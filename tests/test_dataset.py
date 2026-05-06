import numpy as np
import pytest
import torch

from almo_eda.data_loader import AtomisticDataset


def test_atomistic_dataset_length_and_shapes():
    features = np.random.randn(10, 952)
    energies = np.random.randn(10, 2)

    dataset = AtomisticDataset(features, energies)

    assert len(dataset) == 10

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (952,)
    assert y.shape == (2,)


def test_atomistic_dataset_rejects_mismatched_lengths():
    features = np.random.randn(10, 952)
    energies = np.random.randn(9, 2)

    with pytest.raises(AssertionError):
        AtomisticDataset(features, energies)
