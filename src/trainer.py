import torch
import torch.optim as optim

def train_model(model, optimizer, train_loader, val_loader, criterion, device, num_epochs, early_stopping=True, patience=20):
    best_valid_loss = float('inf')
    i_worse = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

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
        # Update learning rate based on validation loss
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "models/best_model.pt")
            # Reset counter if improvement 
            i_worse = 0
        # Only increment if early stopping is enabled and loss didn't improve
        elif early_stopping:
            i_worse += 1
            if i_worse >= patience:
                break
        
        #print(f"Epoch {epoch+1}: Val Loss {valid_loss:.4f}")
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
