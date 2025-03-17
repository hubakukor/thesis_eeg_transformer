#import data_processing
#import train_validate
#import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
from load_cybathlon_files import load_fif_data




# Train and save the model
#train_model(model, train_loader, optimizer, criterion, device, epochs=10)
# torch.save(model.state_dict(), "eeg_transformer_first_dataset.pth")

# #load model
# model.load_state_dict(torch.load("eeg_transformer_first_dataset.pth"))
# model.to(device)  # Don't forget to send it to the correct device after loading

# Validate the model
#validate_model(model, val_loader, criterion, device)

#load data from par2024 global c
data_dir = r"D:\suli\thesis\par2024_two_cmd_c_global\par2024_two_cmd_c_global"

X_train, X_test, Y_train, Y_test = load_fif_data(data_dir)