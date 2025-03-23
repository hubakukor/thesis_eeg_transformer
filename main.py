import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
from load_cybathlon_files import load_fif_data
from train_validate import train_model, validate_model
from model import EEGTransformerModel



# Train and save the model
#train_model(model, train_loader, optimizer, criterion, device, epochs=10)
# torch.save(model.state_dict(), "eeg_transformer_first_dataset.pth")

# #load model
# model.load_state_dict(torch.load("eeg_transformer_first_dataset.pth"))
# model.to(device)  # Don't forget to send it to the correct device after loading

# Validate the model
#validate_model(model, val_loader, criterion, device)

#load data from files
data_dir_c = r"D:\suli\thesis\par2024_two_cmd_c_global\par2024_two_cmd_c_global"
X_train_c, X_test_c, Y_train_c, Y_test_c = load_fif_data(data_dir_c)

data_dir_b = r"D:\suli\thesis\par2024_two_cmd_b_global\par2024_two_cmd_b_global"
X_train_b, X_test_b, Y_train_b, Y_test_b = load_fif_data(data_dir_b)

data_dir_inv = r"D:\suli\thesis\par2024_inv\par2024_inv"
X_train_inv, X_test_inv, Y_train_inv, Y_test_inv = load_fif_data(data_dir_inv)

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
'''
# Define model
model_b = EEGTransformerModel()
train_model(model_b, X_train_b, Y_train_b, epochs=20)
#torch.save(model_b.state_dict(), "model_trained_on_global_b.pth")


#validate
print("Validate model b on global b")
validate_model(model_b, X_test_b, Y_test_b)
print("Validate model b on global c")
validate_model(model_b, X_test_c, Y_test_c)
'''

# Pretrain the model on inv, transfer on global b
'''
model_inv = EEGTransformerModel()
train_model(model_inv, X_train_inv, Y_train_Inv, epochs=10)
torch.save(model_inv.state_dict(), "model_pretrained_on_inv.pth")


#Unfreeze  classifier
for param in model_inv.parameters():    # Freeze all layers
    param.requires_grad = False
for param in model_inv.fc.parameters():   #Unfreeze  classifier
    param.requires_grad = True

# Create optimizer only for unfrozen params, lower learning rate
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_inv.parameters()), lr=0.0005)

# Train using the same function
model = train_model(model_inv, X_train_b, Y_train_b, epochs=5, optimizer=optimizer)

print("Validate model inv on global b")
validate_model(model_inv, X_test_b, Y_test_b)
print("Validate model inv on global c")
validate_model(model_inv, X_test_c, Y_test_c)
'''

# Pretrain on all 3, then transfer on global b
X_train_all = np.concatenate((X_train_c, X_train_b, X_train_inv), axis=0)
Y_train_all = torch.cat((Y_train_c, Y_train_b, Y_train_inv), dim=0)

