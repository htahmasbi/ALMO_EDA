import torch
import torch.nn as nn
import torch.optim as optim
from almo_eda.utils import time_research_task
from almo_eda.logger import get_logger

logger = get_logger("Training", log_file="training.log")


@time_research_task
def train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    criterion,
    device,
    num_epochs,
    early_stopping=True,
    patience=20,
    checkpoint_path=None,
):
    best_valid_loss = float("inf")
    i_worse = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation logic...
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                valid_loss += criterion(model(features), targets).item()

        valid_loss /= len(val_loader)
        valid_losses.append(valid_loss)
        # Update learning rate based on validation loss
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
            # torch.save(model.state_dict(), "models/best_model_test.pt")

            # Reset counter if improvement
            i_worse = 0

        # Only increment if early stopping is enabled and loss didn't improve
        elif early_stopping:
            i_worse += 1
            if i_worse >= patience:
                break

        # print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
    return train_losses, valid_losses


class CustomLoss(nn.Module):
    """
    Normalized MSE loss that accounts for the variance of target values.

    Loss = MSE / Var(target)

    This helps the model adapt to varying scales of energy values.
    """
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        mse = torch.mean((output - target) ** 2)

        # Compute variation of energy (assuming it's std of target)
        variation_of_energy = torch.var(target, correction=0) + 1e-8  # Avoid division by zero

        return mse / variation_of_energy
