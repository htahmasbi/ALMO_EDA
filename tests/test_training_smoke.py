import torch
from torch.utils.data import DataLoader, TensorDataset

from almo_eda.network import FFNet
from almo_eda.trainer import CustomLoss, train_model


def test_train_model_smoke():
    torch.manual_seed(123)

    x = torch.randn(20, 952)
    y = torch.randn(20, 2)

    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=5)
    valid_loader = DataLoader(dataset, batch_size=5)

    model = FFNet(
        input_size=952,
        hidden_layers=[8],
        output_size=2,
        activation="ReLU",
        dropout_prob=0.0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = CustomLoss()
    device = torch.device("cpu")

    train_losses, valid_losses = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        device=device,
        num_epochs=1,
        early_stopping=False,
        checkpoint_path=None,
    )

    assert len(train_losses) == 1
    assert len(valid_losses) == 1
