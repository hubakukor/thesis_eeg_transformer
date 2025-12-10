import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import torch.optim as optim
import numpy as np
from loading import load_fif_data, load_for_complete_cross_validation, load_bci_dataset, load_physionet_eeg, make_default_split, match_common_channels
from train_validate import train_model, evaluate_model
from datasets import EEGTrialsDataset, EEGCroppedDataset, EEGFixedCenterCropDataset
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
data_dir_bci = r"datasets/BCICIV_2a_gdf"
data_dir_physionet = r"datasets/eeg-motor-movementimagery-dataset-1.0.0/files"

#event libraries:

#global c
# event_id_c = {
#     'Stimulus/13': 0,  # both legs
#     'Stimulus/15': 1  # Subtract
# }

#global b
# event_id_b = {
#     'Stimulus/12': 0,  # Right hand
#     'Stimulus/15': 1  # Subtract
# }
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


# #save the results into a csv file
# df = pd.DataFrame(results)
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# filename = f"transf_multiconv_zscorenorm_{timestamp}.xlsx"
# filepath = os.path.join("results_xlsx", filename)
# df.to_excel(filepath, index=False)
#

BCI_CANONICAL_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz"
]

def make_eeg_transformer(input_channels, seq_len, num_classes):
    return EEGTransformerModel(
        input_channels=input_channels,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        embedding_type='sinusoidal',
        conv_block_type='multi',
    )


def main():

    # ----------------------------------------------------
    # 0) Infer seq_len from Physio checkpoint (for fair comparison)
    # ----------------------------------------------------
    print("=== Loading Physio pretrained weights to infer seq_len (for baseline) ===")
    state_dict = torch.load("physio_pretrained_fulltrials.pt", map_location="cpu")
    pretrained_seq_len = state_dict["positional_embedding.pe"].shape[0]
    print(f"Pretrained seq_len (Physio) = {pretrained_seq_len}")

    # ----------------------------------------------------
    # 1) Load BCI data (4 s epochs)
    # ----------------------------------------------------
    print("\n=== Loading BCI dataset (full 4 s trials) ===")
    X_tr, X_val, X_te, y_tr, y_val, y_te, bci_ch_names = load_bci_dataset(
        data_dir=data_dir_bci,
        tmin=0.0,
        tmax=4.0,
    )

    print("BCI shapes BEFORE time-length alignment:")
    print("  Train:", X_tr.shape)
    print("  Val:  ", X_val.shape)
    print("  Test: ", X_te.shape)
    print("  Channels:", len(bci_ch_names), bci_ch_names)

    input_channels = X_tr.shape[1]
    bci_seq_len = X_tr.shape[2]
    print(f"BCI seq_len: {bci_seq_len}, Physio seq_len: {pretrained_seq_len}")

    # ----------------------------------------------------
    # 2) Align BCI time length to pretrained_seq_len
    # ----------------------------------------------------
    def fix_time_len(X, target_len):
        N, C, T = X.shape
        if T == target_len:
            return X
        elif T > target_len:
            # Trim at the end
            return X[:, :, :target_len]
        else:
            # Pad with zeros at the end (unlikely here)
            pad_width = target_len - T
            pad = np.zeros((N, C, pad_width), dtype=X.dtype)
            return np.concatenate([X, pad], axis=2)

    X_tr = fix_time_len(X_tr, pretrained_seq_len)
    X_val = fix_time_len(X_val, pretrained_seq_len)
    X_te = fix_time_len(X_te, pretrained_seq_len)

    print("\nBCI shapes AFTER time-length alignment:")
    print("  Train:", X_tr.shape)
    print("  Val:  ", X_val.shape)
    print("  Test: ", X_te.shape)

    # ----------------------------------------------------
    # 3) Filter to overlapping 3 classes: 0,1,2 (drop tongue=3)
    # ----------------------------------------------------
    print("\n=== Filtering BCI to 3 classes (left/right/feet) ===")

    def filter_3classes(X, y):
        mask = (y != 3).cpu().numpy().astype(bool)  # keep 0,1,2
        X_f = X[mask]
        y_f = y[mask]
        return X_f, y_f

    X_tr_3, y_tr_3 = filter_3classes(X_tr, y_tr)
    X_val_3, y_val_3 = filter_3classes(X_val, y_val)
    X_te_3, y_te_3 = filter_3classes(X_te, y_te)

    print("BCI shapes AFTER class filtering:")
    print("  Train:", X_tr_3.shape, "labels:", y_tr_3.shape)
    print("  Val:  ", X_val_3.shape, "labels:", y_val_3.shape)
    print("  Test: ", X_te_3.shape, "labels:", y_te_3.shape)

    # Label codes 0/1/2 already match Physio (left/right/feet).

    # ----------------------------------------------------
    # 4) Build full-trial TensorDatasets (no crops)
    # ----------------------------------------------------
    print("\n=== Building BCI TensorDatasets (full 4 s trials, 3 classes) ===")
    train_X_tensor = torch.from_numpy(X_tr_3).float()
    val_X_tensor   = torch.from_numpy(X_val_3).float()

    train_dataset = TensorDataset(train_X_tensor, y_tr_3)
    val_dataset   = TensorDataset(val_X_tensor,   y_val_3)

    # ----------------------------------------------------
    # 5) Init BCI model FROM SCRATCH (no pretraining)
    # ----------------------------------------------------
    print("\n=== Initializing 3-class BCI model (no pretraining) ===")
    N_CLASSES = 3

    model_bci = make_eeg_transformer(
        input_channels=input_channels,
        seq_len=pretrained_seq_len,  # same as transfer model
        num_classes=N_CLASSES,
    )

    # IMPORTANT: do NOT load Physio weights here.
    # This is your baseline.

    # ----------------------------------------------------
    # 6) Train on BCI-only (full fine-tuning)
    # ----------------------------------------------------
    print("\n=== Training BCI 3-class baseline (full trials) ===")
    model_bci = train_model(
        model=model_bci,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50,
        lr=5e-4,
        patience=5,
        batch_size_train=8,
        batch_size_val=8,
    )

    torch.save(model_bci.state_dict(), "bci_baseline_3class_fulltrials.pt")
    print("\nSaved BCI-only baseline model to bci_baseline_3class_fulltrials.pt")

    # ----------------------------------------------------
    # 7) Final evaluation on BCI 3-class test set
    # ----------------------------------------------------
    print("\n=== Final evaluation on BCI 3-class baseline (full trials) ===")
    test_loss, test_acc, test_bal_acc = evaluate_model(
        model=model_bci,
        X=X_te_3,
        Y=y_te_3,
        crop_len=pretrained_seq_len,  # full trial
        n_crops=1,
        plot_confusion_matrix=True,
    )

    print(f"BCI 3-class baseline (no pretraining) - "
          f"Test Loss: {test_loss:.4f}, "
          f"Acc: {test_acc:.4f}, "
          f"BalAcc: {test_bal_acc:.4f}")


if __name__ == "__main__":
    main()