import model
#import data_processing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim



# Training function
def train_model(model, train_loader, optimizer, criterion, device, epochs=10):
    """
    Train the model using the training dataset.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training dataset.
        optimizer: Optimizer for updating the model weights.
        criterion: Loss function to compute the training loss.
        device: Device to train on ('cuda' or 'cpu').
        epochs: Number of training epochs.
    """
    model.to(device)
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


# Validation function
def validate_model(model, val_loader, criterion, device):
    """
    Validate the model using the validation dataset.

    Args:
        model: The model to validate.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function to compute the validation loss.
        device: Device to validate on ('cuda' or 'cpu').
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == batch_Y).sum().item()
            total += batch_Y.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy