import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
import numpy as np
from loading import load_fif_data, load_for_complete_cross_validation
from train_validate import train_model, validate_model
from model import EEGTransformerModel, ShallowConvNet, MultiscaleConvolution
from collections import Counter
import pandas as pd
from datetime import datetime
from braindecode.models import EEGConformer
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#load data from files

#data directories
data_dir_c = r"D:\suli\thesis\par2024_two_cmd_c_global\par2024_two_cmd_c_global"
data_dir_b = r"D:\suli\thesis\par2024_two_cmd_b_global\par2024_two_cmd_b_global"
data_dir_inv = r"D:\suli\thesis\par2024_inv\par2024_inv"

#event libraries:

#global c
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

# X_train_b, X_val_b, X_test_b, Y_train_b, Y_val_b, Y_test_b = load_fif_data(data_dir_b, event_id_b, test_set=True)


# X_train_inv, X_val_inv, X_test_inv, Y_train_inv, Y_val_inv, Y_test_inv = load_fif_data(data_dir_inv, event_id_b, test_set=True)
# X_train_inv, X_val_inv, Y_train_inv, Y_val_inv = load_fif_data(data_dir_inv, event_id_b, test_set=False)


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
# model_b = MultiscaleConvolution()


# train_model(model_b, X_train_b, Y_train_b, X_val_b, Y_val_b)
#torch.save(model_b.state_dict(), "model_trained_on_global_b.pth")

#validate





#Train on c
#Define model
# model_c = EEGTransformerModel(embedding_type='sinusoidal')
# train_model(model_c, X_train_c, Y_train_c, X_val_c, Y_val_c, epochs=50, lr=0.0005, noise_augmentation=0.0)
# torch.save(model_c.state_dict(), "model_trained_on_global_c.pth")
# # print("Trained on dataset c with no embedding")
# print("Validate model c")
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

# model_convnet = ShallowConvNet()
# model = EEGTransformerModel(embedding_type='sinusoidal')
# train_model(model, X_train_inv, Y_train_inv, X_val_inv, Y_val_inv, epochs=50, lr=0.0005)
# print("Trained on inv")
# torch.save(model.state_dict(), "model_pretrained_on_inv_for_transfer.pth")

# model_conformer = EEGConformer(
#     n_chans=63,          # number of EEG channels
#     n_classes=2,          # number of output classes
#     input_window_samples=1501,  # number of time samples in each window
#     sfreq=500,
#     final_fc_length="auto",  # length of the final fully connected layer
# )
# train_model(model_conformer, X_train_inv, Y_train_inv, X_val_inv, Y_val_inv, epochs=50, lr=0.0005)
# torch.save(model_conformer.state_dict(), "conformer_pretrained_on_inv_for_transfer.pth")

# print("Validate model before transfer learning")
# validate_model(model_inv, X_test_c, Y_test_c)
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
# train_model(model_inv, X_train_c, Y_train_c, X_val_c, Y_val_c, epochs=10, optimizer=optimizer)
# # torch.save(model_inv.state_dict(), "model_transfer_inv_to_b.pth")
#
# print("Validate model after transfer learning")
# validate_model(model_inv, X_test_c, Y_test_c)
# print("Validate model inv on global c")
# validate_model(model_inv, X_test_c, Y_test_c)


# Using cross validation, fine tune the pretrained_on_inv model on the global b dataset

folder_names = ["E055", "E056", "E057", "E058", "E059", "E060", "E061", "E062", "E063", "E064"] #global b
# # folder_names = ["E055", "E056", "E057", "E058", "E059", "E060", "E061", "E062", "E063", "E064", "E065", "E066", "E067", "E068"] #global c
#
results = []
#
# Save predictions for confusion matrix
all_preds_all_folds = []
all_targets_all_folds = []
#
for target_folder in folder_names:
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_for_complete_cross_validation(data_dir_b, event_id_b, target_folder, augment_train_ratio=0.25)
    print(f"\n==== Training on all folders except '{target_folder}' ====")
#
    model = EEGTransformerModel()
#
#     # model = EEGConformer(
#     #     n_chans=63,  # number of EEG channels
#     #     n_classes=2,  # number of output classes
#     #     input_window_samples=1501,  # number of time samples in each window
#     #     sfreq=500,
#     #     final_fc_length="auto",  # length of the final fully connected layer
#     # )
#
#     # Load the weights of the pretrained model
#     # state_dict = torch.load("model_pretrained_on_inv_for_transfer.pth")
#     # model.load_state_dict(state_dict)
#     #
#
#     # #set the trainable layers in the conformer model
#     # for param in model.parameters():
#     #     param.requires_grad = False #freeze everything
#     #
#     # for name, param in model.named_parameters():
#     #     if (
#     #         "fc" in name or # last fully connected layer
#     #         "projection" in name or # last projection layer
#     #         "encoder.5" in name  # last encoder block
#     #     ):
#     #         param.requires_grad = True
#
# # #set the trainable layers in the eeg_transformer model
# #     # Freeze everything
# #     for param in model.parameters():
# #         param.requires_grad = False
# #
# #     # Unfreeze and reset classifier
# #     for param in model.fc.parameters():
# #         param.requires_grad = True
# #     # model.fc.reset_parameters()
# #
# #     # unfreeze and reset projection layer
# #     for param in model.proj.parameters():
# #         param.requires_grad = True
# #     # model.proj.reset_parameters()
# #
# #     # Unfreeze the last transformer layer
# #     for param in model.transformer.layers[-1].parameters():
# #         param.requires_grad = True
#
#     # #Unfreeze layers in the shallow convnet model
#     # model.classifier.reset_parameters()
#     # for param in model.classifier.parameters():
#     #     param.requires_grad = True
#     #
#     # #Unfreeze the spatial convolution layer
#     # for param in model.conv_spat.parameters():
#     #     param.requires_grad = True
#
#     # Create optimizer only for unfrozen params, lower learning rate
#     # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
#
    # Train
    train_model(model, X_train, Y_train, X_val, Y_val, epochs=50)

    print("Test model")
    val_loss, val_acc, val_bal_acc, preds, targets = validate_model(model, X_test, Y_test, return_preds_targets=True)

    # Accumulate all predictions
    all_preds_all_folds.extend(preds)
    all_targets_all_folds.extend(targets)
#
    #Log results
    results.append({
        "test_folder": target_folder,
        "accuracy": val_acc,
        "balanced_accuracy": val_bal_acc,
        "loss": val_loss
    })

#save the results into a csv file
df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"transf_multiconv_augmented_train_{timestamp}.xlsx"
filepath = os.path.join("results_xlsx", filename)
df.to_excel(filepath, index=False)
#
# # Plot confusion matrix for all folds
# cm = confusion_matrix(all_targets_all_folds, all_preds_all_folds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix (Custom Model, All Folds)")
# plt.show()
