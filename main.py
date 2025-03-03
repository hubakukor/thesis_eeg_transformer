import data_processing
import train_validate
import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim




# Train and save the model
#train_model(model, train_loader, optimizer, criterion, device, epochs=10)
# torch.save(model.state_dict(), "eeg_transformer_first_dataset.pth")

# #load model
# model.load_state_dict(torch.load("eeg_transformer_first_dataset.pth"))
# model.to(device)  # Don't forget to send it to the correct device after loading

# Validate the model
#validate_model(model, val_loader, criterion, device)