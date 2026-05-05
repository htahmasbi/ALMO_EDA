import torch

from almo_eda.network import FFNet


def test_ffnet_output_shape():
    model = FFNet(
        input_size=952,
        hidden_layers=[50, 50],
        output_size=2,
        activation="Tanh",
        dropout_prob=0.0,
    )

    x = torch.randn(4, 952)
    y = model(x)

    assert y.shape == (4, 2)
