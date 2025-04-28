import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
import numpy as np
from loading import load_fif_data
from train_validate import train_model, validate_model
from model import EEGTransformerModel
from collections import Counter

#load data from files

#data directories
data_dir_c = r"D:\suli\thesis\par2024_two_cmd_c_global\par2024_two_cmd_c_global"
data_dir_b = r"D:\suli\thesis\par2024_two_cmd_b_global\par2024_two_cmd_b_global"
data_dir_inv = r"D:\suli\thesis\par2024_inv\par2024_inv"

#event libraries:

# #global c
# event_id_c = {
#     'Stimulus/13': 0,  # both legs
#     'Stimulus/15': 1  # Subtract
# }

#global b
event_id_b = {
    'Stimulus/12': 0,  # Right hand
    'Stimulus/15': 1  # Subtract
}
#
#inv
# event_id_inv = {
#     'Stimulus/12': 0,  # Right hand
#     'Stimulus/13': 1,  # both legs
#     'Stimulus/15': 2  # Subtract
# }
#
# #match the labels for the classes for training on the combined datasets
# event_id_b = {k: v for k, v in event_id_inv.items() if k in ['Stimulus/12', 'Stimulus/15']}
# event_id_c = {k: v for k, v in event_id_inv.items() if k in ['Stimulus/13', 'Stimulus/15']}


# #load data into train and test sets
# X_train_c, X_val_c, X_test_c, Y_train_c, Y_val_c, Y_test_c = load_fif_data(data_dir_c, event_id_c)

X_train_b, X_val_b, X_test_b, Y_train_b, Y_val_b, Y_test_b = load_fif_data(data_dir_b, event_id_b)


# X_train_inv, X_val_inv, X_test_inv, Y_train_inv, Y_val_inv, Y_test_inv = load_fif_data(data_dir_inv, event_id_b)

# Info about the number of events loaded
'''
#print info about the loaded data
print("Dataset C:", X_train_c.shape[0] + X_test_c.shape[0], "epochs")
print("Dataset B:", X_train_b.shape[0] + X_test_b.shape[0], "epochs")
print("Dataset Inv:", X_train_inv.shape[0] + X_test_inv.shape[0], "epochs")

print("C labels:", torch.unique(Y_train_C, return_counts=True))
print("B labels:", torch.unique(Y_train_B, return_counts=True))
print("Inv labels:", torch.unique(Y_train_Inv, return_counts=True))

print("Shape C:", X_train_c.shape, X_test_c.shape)
print("Shape B:", X_train_b.shape, X_test_b.shape)
print("Shape Inv:", X_train_inv.shape, X_test_inv.shape)
'''

# Train the model on global b
# Define model
model_b = EEGTransformerModel(embedding_type='none')
train_model(model_b, X_train_b, Y_train_b, X_val_b, Y_val_b, lr=0.0005, noise_augmentation=0)
#torch.save(model_b.state_dict(), "model_trained_on_global_b.pth")

#validate
print("Test model on global b dataset")
validate_model(model_b, X_test_b, Y_test_b)


# #Train on c
# #Define model
# model_c = EEGTransformerModel(embedding_type='sinusoidal')
# train_model(model_c, X_train_c, Y_train_c, epochs=30, lr=0.0005)
# torch.save(model_c.state_dict(), "model_trained_on_global_c.pth")
# print("Trained on dataset c with no embedding")
# print("Validate model c on model c")
# validate_model(model_c, X_test_c, Y_test_c)

# # #Train on inv
# # #Define model
# model_inv = EEGTransformerModel(num_classes = len(event_id_inv), embedding_type='sinusoidal')
# train_model(model_inv, X_train_inv, Y_train_inv, epochs=20, lr=0.0005)
# #torch.save(model_inv.state_dict(), "model_trained_on_global_inv.pth")
#
# print("Validate model inv on dataset inv")
# validate_model(model_inv, X_test_inv, Y_test_inv)

# pretrain on inv

# model_inv = EEGTransformerModel(embedding_type='sinusoidal')
# train_model(model_inv, X_train_inv, Y_train_inv, epochs=20, lr=0.0005)
# # torch.save(model_inv.state_dict(), "model_pretrained_on_inv.pth")
#
#
# # Freeze everything
# for param in model_inv.parameters():
#     param.requires_grad = False
#
# # Unfreeze and reset classifier
# for param in model_inv.fc.parameters():
#     param.requires_grad = True
# model_inv.fc.reset_parameters()
#
# # unfreeze and reset projection layer
# for param in model_inv.proj.parameters():
#     param.requires_grad = True
# model_inv.proj.reset_parameters()
#
# # Unfreeze the last transformer layer
# for param in model_inv.transformer.layers[-1].parameters():
#     param.requires_grad = True
#
# # Create optimizer only for unfrozen params, lower learning rate
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_inv.parameters()), lr=0.0001)
#
# # Train
# train_model(model_inv, X_train_b, Y_train_b, epochs=10, optimizer=optimizer)
# # torch.save(model_inv.state_dict(), "model_transfer_inv_to_b.pth")
#
# print("Validate model inv on global b")
# validate_model(model_inv, X_test_b, Y_test_b)
# print("Validate model inv on global c")
# validate_model(model_inv, X_test_c, Y_test_c)


# # Pretrain on all 3, then transfer on global c
#
# X_train_all = np.concatenate([X_train_c, X_train_b, X_train_inv], axis=0)
# Y_train_all = torch.cat((Y_train_c, Y_train_b, Y_train_inv), dim=0)
#
# print("Pretraining class counts:", torch.unique(Y_train_all, return_counts=True))
#
# model_all = EEGTransformerModel(num_classes=3)
# train_model(model_all, X_train_all, Y_train_all, epochs=40, lr=0.0005)
# torch.save(model_all.state_dict(), "model_pretrained_on_all.pth")
#
# print("Finished pretraining on all datasets, starting transfer learning on c")
# print("Transfer tuning on global c class counts:", torch.unique(Y_train_c, return_counts=True))
#
# model_transfer = EEGTransformerModel(num_classes=2)
# state_dict = torch.load("model_pretrained_on_all.pth")
#
# # Remove fc layer weights since dimensions changed
# del state_dict["fc.weight"]
# del state_dict["fc.bias"]
# model_transfer.load_state_dict(state_dict, strict=False)
#
# #freeze everything but fc and projection layer
# for param in model_transfer.parameters():
#     param.requires_grad = False
# for param in model_transfer.proj.parameters():
#     param.requires_grad = True
# for param in model_transfer.fc.parameters():
#     param.requires_grad = True
# # Unfreeze the last transformer layer
# for param in model_transfer.transformer.layers[-1].parameters():
#     param.requires_grad = True
#
# #tune on global c
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_transfer.parameters()), lr=0.0001)
# train_model(model_transfer, X_train_c, Y_train_c, epochs=20, optimizer=optimizer)
#
# validate_model(model_transfer, X_test_c, Y_test_c)
