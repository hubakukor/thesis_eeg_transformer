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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE

def train_model(model, train_dataset, val_dataset,
                         epochs=50, optimizer=None, lr=0.0005, patience=5,
                         batch_size_train=8, batch_size_val=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)


    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(batch_Y).sum().item()
            total += batch_Y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        # ---- VALIDATE ----
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for val_X, val_Y in val_loader:
                val_X, val_Y = val_X.to(device), val_Y.to(device)
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_Y).item()

                _, predicted = torch.max(val_outputs, dim=1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(val_Y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, "
            f"Balanced Accuracy: {val_bal_acc:.4f}"
        )

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def evaluate_model(
    model,
    X,
    Y,
    crop_len=500,
    n_crops=5,
    return_preds_targets=False,
    plot_confusion_matrix=False,
):
    """
    Multi-crop evaluation for fixed-length Transformer input.

    Args:
        model: trained model
        X: np.ndarray, shape (N, C, T_full)
        Y: torch.Tensor or np.ndarray, shape (N,)
        crop_len: length of each crop in samples (default: 500)
        n_crops: number of crops per trial (default: 5, evenly spaced)
        return_preds_targets: return (preds, targets) if True
        plot_confusion_matrix: plot confusion matrix if True

    Returns:
        avg_loss, accuracy, balanced_acc
        (optionally) all_preds, all_targets
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # Ensure Y is a 1D numpy array
    if torch.is_tensor(Y):
        y_np = Y.cpu().numpy()
    else:
        y_np = np.asarray(Y)

    N, C, T_full = X.shape
    if crop_len > T_full:
        raise ValueError(f"crop_len={crop_len} > T_full={T_full}")

    # Define crop start indices: n_crops evenly spaced positions
    max_start = T_full - crop_len
    if n_crops == 1:
        # center crop
        crop_starts = [max_start // 2]
    else:
        crop_starts = np.linspace(0, max_start, num=n_crops).astype(int).tolist()

    print(f"Using crop starts (samples): {crop_starts}")

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(N):
            x_trial = X[i]          # (C, T_full)
            label = int(y_np[i])

            # Build crops for this trial: (n_crops, C, crop_len)
            crops = []
            for s in crop_starts:
                e = s + crop_len
                crops.append(x_trial[:, s:e])
            crops = np.stack(crops, axis=0)          # (K, C, crop_len)
            crops_t = torch.from_numpy(crops).float().to(device)

            # Forward pass on all crops at once
            logits = model(crops_t)                  # (K, n_classes)

            # Average logits across crops
            avg_logits = logits.mean(dim=0, keepdim=True)   # (1, n_classes)

            # Compute loss on averaged logits
            label_t = torch.tensor([label], dtype=torch.long, device=device)
            loss = criterion(avg_logits, label_t)
            total_loss += loss.item()

            # Prediction from averaged logits
            _, pred = torch.max(avg_logits, dim=1)
            pred = int(pred.item())

            all_preds.append(pred)
            all_targets.append(label)

    avg_loss = total_loss / N
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)

    print(f"Multi-crop Eval -> Loss: {avg_loss:.4f}, "
          f"Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}")

    true_counts = Counter(all_targets)
    pred_counts = Counter(all_preds)
    print(f"\n True class distribution: {true_counts}")
    print(f" Predicted class distribution: {pred_counts}")

    if plot_confusion_matrix:
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix (Multi-crop)")
        plt.show()

    if return_preds_targets:
        return avg_loss, accuracy, balanced_acc, all_preds, all_targets
    else:
        return avg_loss, accuracy, balanced_acc


##### Old training functions #####

#create augmented data
def augment_with_noise(X, Y, ratio=0.5, noise_level=0.01):
    """
    Augments data X, Y by adding Gaussian noise to a fraction (ratio) of the data.

    Parameters:
        X (numpy array): Training data, shape (N, C, T) or (N, features)
        Y (numpy array): Labels, shape (N,)
        ratio (float): How many extra samples to create (e.g., 0.5 = 50% more)
        noise_level (float): Standard deviation of Gaussian noise relative to data range

    Returns:
        Augmented X, Y
    """
    n_samples = int(X.shape[0] * ratio)

    # Randomly select samples to copy
    idx = np.random.choice(X.shape[0], n_samples, replace=True)
    X_selected = X[idx]
    Y_selected = Y[idx]

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level * np.std(X_selected), X_selected.shape)
    X_noisy = X_selected + noise

    # Concatenate original and augmented data
    X_augmented = np.concatenate([X, X_noisy], axis=0)
    Y_augmented = np.concatenate([Y, Y_selected], axis=0)

    return X_augmented, Y_augmented



# Training function
def train_model_old(model, X_train, Y_train, X_val, Y_val, epochs=50, optimizer=None,
                lr=0.0005, patience=5, noise_augmentation=0):
    """
    Train the model using the training dataset.

    Args:
        model: The model to train.
        X_train, Y_train: Training dataset and its labels.
        X_val, Y_val: Validation dataset and its labels.
        epochs: Maximum number of training epochs.
        optimizer: Optimizer to use for training. If not specified, Adam optimizer is used.
        lr: Learning rate for the optimizer.
        patience: Number of epochs without improvement before early stopping.
        noise_augmentation: Ratio of the synthetic samples to create. 0 means no augmentation.

    Returns:
        model: Trained model.
    """

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.train()  # Set model to training mode

    criterion = nn.CrossEntropyLoss()

    # #use weighted loss
    # class_weights = compute_class_weight(
    #     class_weight='balanced',
    #     classes=np.array([0, 1]),
    #     y=Y_train.numpy()
    # )
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    #
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Augment the training data with noise
    if not noise_augmentation == 0:
        print(f"Creating syntetic samples with noise, ratio of noise samples: {noise_augmentation}")
        X_train, Y_train = augment_with_noise(X_train, Y_train, ratio=noise_augmentation)


    # Prepare data
    if noise_augmentation == 0:
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), Y_train)
    else:
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())  #bc of different format when creating synthetic samples

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), Y_val)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

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

            #track training accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(batch_Y).sum().item()
            total += batch_Y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for val_X, val_Y in val_loader:
                val_X, val_Y = val_X.to(device), val_Y.to(device)
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_Y).item()

                _, predicted = torch.max(val_outputs, dim=1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(val_Y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = np.mean(np.array(val_preds) == np.array(val_targets))
        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Balanced Accuracy: {val_bal_acc:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state) # load best model
    return model


# Validation function
def validate_model_old(model, X_test, Y_test, return_preds_targets=False, plot_confusion_matrix=False):
    """
    Validate the model using the validation dataset.

    Args:
        model: The model to validate.
        X_test, Y_test: Validation dataset and its labels.
        return_preds_targets: If True, returns the predictions and targets for the confusion matrix.
        plot_confusion_matrix: If True, plots the confusion matrix.

    Returns:
        avg_loss, accuracy, balanced_acc: Average loss, accuracy, and balanced accuracy.
        all_preds, all_targets (optional): Predictions and targets for the confusion matrix.
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

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
    pred_counts = Counter(all_preds)
    true_counts = Counter(all_targets)

    #label distribution
    print(f"\n True class distribution: {true_counts}")
    print(f" Predicted class distribution: {pred_counts}")

    #plot confusion matrix if requested
    if plot_confusion_matrix:
        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

    if return_preds_targets:
        return avg_loss, accuracy, balanced_acc, all_preds, all_targets
    else:
        return avg_loss, accuracy, balanced_acc