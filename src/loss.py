import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        mse = torch.mean((output - target) ** 2)
        # Compute variation of energy (assuming it's std of target)
        variation_of_energy = torch.var(target) + 1e-8 
        return mse / variation_of_energy
