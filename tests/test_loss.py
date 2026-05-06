import torch

from almo_eda.trainer import CustomLoss


def test_custom_loss_returns_finite_nonnegative_value():
    criterion = CustomLoss()

    output = torch.tensor([[1.0, 2.0], [1.5, 2.5]])
    target = torch.tensor([[1.1, 1.9], [1.4, 2.6]])

    loss = criterion(output, target)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
