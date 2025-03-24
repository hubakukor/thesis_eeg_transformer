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

#events in global c
event_id = {
    'Stimulus/13': 0,  # both legs
    'Stimulus/15': 1  # Subtract
}
data_dir_c = r"D:\suli\thesis\par2024_two_cmd_c_global\par2024_two_cmd_c_global"
X_train_c, X_test_c, Y_train_c, Y_test_c = load_fif_data(data_dir_c, event_id)

#events in global b
# event_id = {
#     'Stimulus/12': 0,  # Right hand
#     'Stimulus/15': 1  # Subtract
# }
#
# data_dir_b = r"D:\suli\thesis\par2024_two_cmd_b_global\par2024_two_cmd_b_global"
# X_train_b, X_test_b, Y_train_b, Y_test_b = load_fif_data(data_dir_b, event_id)

# event_id = {
#     'Stimulus/12': 0,  # Right hand
#     'Stimulus/13': 1,  # both legs
#     'Stimulus/15': 2  # Subtract
# }
#
# data_dir_inv = r"D:\suli\thesis\par2024_inv\par2024_inv"
# X_train_inv, X_test_inv, Y_train_inv, Y_test_inv = load_fif_data(data_dir_inv, event_id)

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
# model_b = EEGTransformerModel()
# train_model(model_b, X_train_b, Y_train_b, epochs=10)
#torch.save(model_b.state_dict(), "model_trained_on_global_b.pth")

# #validate
# print("Validate model b on global b")
# validate_model(model_b, X_test_b, Y_test_b)

# #purposely overfit
# idx_0 = (Y_train_c == 0).nonzero(as_tuple=True)[0][:5]
# idx_1 = (Y_train_c == 1).nonzero(as_tuple=True)[0][:5]
# idx = torch.cat((idx_0, idx_1))
#
# X_small = X_train_c[idx]
# Y_small = Y_train_c[idx]
#
# # Train with high LR and more epochs
# model = EEGTransformerModel(input_channels=63, seq_len=1501, num_classes=2)
# train_model(model, X_small, Y_small, epochs=50, lr=0.01)
# validate_model(model, X_small, Y_small)


#Train on c
#Define model
model_c = EEGTransformerModel()
train_model(model_c, X_train_c, Y_train_c, epochs=20, lr=0.01)
#torch.save(model_b.state_dict(), "model_trained_on_global_b.pth")

print("Validate model c on model c")
validate_model(model_c, X_test_c, Y_test_c)

#pretrain on inv

# model_inv = EEGTransformerModel()
# train_model(model_inv, X_train_inv, Y_train_inv, epochs=20)
# torch.save(model_inv.state_dict(), "model_pretrained_on_inv.pth")
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
# # Optionally: unfreeze and reset projection layer too
# for param in model_inv.proj.parameters():
#     param.requires_grad = True
# model_inv.proj.reset_parameters()
#
# # Create optimizer only for unfrozen params, lower learning rate
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_inv.parameters()), lr=0.005)
#
# # Train using the same function
# train_model(model_inv, X_train_c, Y_train_c, epochs=10, optimizer=optimizer)
#
# # print("Validate model inv on global b")
# # validate_model(model_inv, X_test_b, Y_test_b)
# print("Validate model inv on global c")
# validate_model(model_inv, X_test_c, Y_test_c)


# Pretrain on all 3, then transfer on global b
'''
X_train_all = np.concatenate([X_train_c, X_train_b, X_train_inv], axis=0)
Y_train_all = torch.cat((Y_train_c, Y_train_b, Y_train_inv), dim=0)

model_all = EEGTransformerModel()
train_model(model_all, X_train_all, Y_train_all, epochs=10)
torch.save(model_all.state_dict(), "model_pretrained_on_all.pth")


# Freeze all layers
for param in model_all.parameters():
    param.requires_grad = False
for param in model_all.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_all.parameters()), lr=0.0005)
model = train_model(model_all, X_train_b, Y_train_b, epochs=5, optimizer=optimizer)

validate_model(model, X_test_b, Y_test_b)
'''