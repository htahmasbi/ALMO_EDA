import torch.nn as nn

class FFNet(nn.Module):
    """
    A configurable feed-forward neural network with multiple hidden layers,
    custom activation functions, Dropout, BatchNorm, and weight initialization.
    """
    def __init__(self, input_size, hidden_layers, output_size, activation="ReLU", dropout_prob=0.0):
        """
        Parameters:
        - input_size (int): Number of input features.
        - hidden_layers (list of int): List specifying the size of each hidden layer.
        - output_size (int): Number of output features.
        - activation_func (nn.Module): Activation function class (e.g., nn.ReLU).
        - dropout_prob (float): Dropout probability (default 0.2).
        """
        super().__init__()
        
        # Map config string to Torch class
        activations = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
        act_func = activations.get(activation, nn.ReLU)

        layers = []
        in_features = input_size
        for h_size in hidden_layers:
            layers.append(nn.Linear(in_features, h_size))
            layers.append(act_func())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            in_features = h_size

        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.
        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, input_size).
        
        Returns:
        - Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.model(x)
