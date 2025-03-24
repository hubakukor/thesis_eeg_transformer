import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



# Training function
def train_model(model, X_train, Y_train, epochs=10, optimizer=None, lr=0.0001):
    """
    Train the model using the training dataset.

    Args:
        model: The model to train.
        X_train, Y_train: Training dataset and its labels.
        epochs: Number of training epochs.
    """

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.train()  # Set model to training mode

    # criterion = nn.CrossEntropyLoss()

    #use weighted loss
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=Y_train.numpy()
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare data
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), Y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

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

    return model


# Validation function
def validate_model(model, X_test, Y_test):
    """
    Validate the model using the validation dataset.

    Args:
        model: The model to validate.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function to compute the validation loss.
        device: Device to validate on ('cuda' or 'cpu').
    """
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()  # Set model to evaluation mode

    criterion = nn.CrossEntropyLoss()

    # Convert and load test data
    val_dataset = TensorDataset(torch.from_numpy(X_test).float(), Y_test)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_Y.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = sum([p == t for p, t in zip(all_preds, all_targets)]) / len(all_preds)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
    pred_counts = Counter(all_preds)
    true_counts = Counter(all_targets)

    print(f"\n True class distribution: {true_counts}")
    print(f" Predicted class distribution: {pred_counts}")

    return avg_loss, accuracy, balanced_acc