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
from model import EEGTransformerModel, ShallowConvNet, MultiscaleConvolution, TFViT2
from tf_features import pick_channels_by_name, eeg_to_logspec_stft, evaluate_model_tf
from collections import Counter
import pandas as pd
from datetime import datetime
from braindecode.models import EEGConformer
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score

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

def make_eeg_transformer(input_channels, seq_len, num_classes=3):
    return EEGTransformerModel(
        input_channels=input_channels,
        seq_len=seq_len,          # full trial length in samples
        num_classes=num_classes,  # 3 classes: left/right/feet
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        embedding_type="sinusoidal",
        conv_block_type="multi",
    )


# def main():
#     # -------------------------
#     # Config
#     # -------------------------
#     data_dir_bci = r"datasets/BCICIV_2a_gdf"
#     tmin, tmax = 0.0, 4.0
#     n_classes = 4
#
#     # STFT params
#     sfreq = 250
#     nperseg = 1001
#     noverlap = 0
#     fmin, fmax = 8.0, 30.0
#     eps = 1e-8
#
#     # Training params
#     epochs = 50
#     patience = 7
#     batch_size = 16
#     lr = 3e-4
#     weight_decay = 0.01
#
#     # ViT params
#     patch_f = 3
#     patch_t = 4
#     d_model = 192
#     nhead = 6
#     num_layers = 4
#     dropout = 0.1
#
#     # -------------------------
#     # 1) Load BCI (your loader already does label mapping)
#     # -------------------------
#     print("=== Loading BCI ===")
#     X_tr, X_val, X_te, y_tr, y_val, y_te, ch_names = load_bci_dataset(
#         data_dir=data_dir_bci,
#         tmin=tmin,
#         tmax=tmax,
#     )
#
#     print("Time-domain shapes:")
#     print("  Train:", X_tr.shape, "y:", y_tr.shape)
#     print("  Val:  ", X_val.shape, "y:", y_val.shape)
#     print("  Test: ", X_te.shape, "y:", y_te.shape)
#     print("  Channels:", len(ch_names))
#
#     # -------------------------
#     # 2) Compute TF (log-power STFT)
#     # -------------------------
#     print("\n=== Computing TF (log-power STFT) ===")
#     Xtr_tf = eeg_to_logspec_stft(X_tr, sfreq, None, nperseg, noverlap, fmin, fmax, eps)
#     Xva_tf = eeg_to_logspec_stft(X_val, sfreq, None, nperseg, noverlap, fmin, fmax, eps)
#     Xte_tf = eeg_to_logspec_stft(X_te, sfreq, None, nperseg, noverlap, fmin, fmax, eps)
#
#     print("TF shapes:")
#     print("  Train TF:", Xtr_tf.shape)
#     print("  Val TF:  ", Xva_tf.shape)
#     print("  Test TF: ", Xte_tf.shape)
#
#     # -------------------------
#     # 3) TF normalization (train stats only)
#     # -------------------------
#     print("\n=== TF normalization (train stats only) ===")
#     mu = Xtr_tf.mean(axis=(0, 3), keepdims=True)               # (1, C, F, 1)
#     sigma = Xtr_tf.std(axis=(0, 3), keepdims=True) + 1e-6      # (1, C, F, 1)
#
#     Xtr_tf = (Xtr_tf - mu) / sigma
#     Xva_tf = (Xva_tf - mu) / sigma
#     Xte_tf = (Xte_tf - mu) / sigma
#
#     print("Train TF mean/std (after):", float(Xtr_tf.mean()), float(Xtr_tf.std()))
#     print("Val   TF mean/std (after):", float(Xva_tf.mean()), float(Xva_tf.std()))
#     print("Test  TF mean/std (after):", float(Xte_tf.mean()), float(Xte_tf.std()))
#
#     # -------------------------
#     # 4) Build datasets
#     # -------------------------
#     y_tr = y_tr if torch.is_tensor(y_tr) else torch.from_numpy(y_tr).long()
#     y_val = y_val if torch.is_tensor(y_val) else torch.from_numpy(y_val).long()
#     y_te = y_te if torch.is_tensor(y_te) else torch.from_numpy(y_te).long()
#
#     train_ds = TensorDataset(torch.from_numpy(Xtr_tf).float(), y_tr)
#     val_ds   = TensorDataset(torch.from_numpy(Xva_tf).float(), y_val)
#
#     # -------------------------
#     # 5) Init model
#     # -------------------------
#     print("\n=== Initializing TFViT2 ===")
#     Cin, Fbins, Tbins = Xtr_tf.shape[1], Xtr_tf.shape[2], Xtr_tf.shape[3]
#     print(f"Input TF grid: Cin={Cin}, F={Fbins}, TT={Tbins}")
#
#     model = TFViT2(
#         in_chans=Cin,
#         num_classes=n_classes,
#         d_model=d_model,
#         nhead=nhead,
#         num_layers=num_layers,
#         dim_feedforward=4 * d_model,
#         dropout=dropout,
#         patch_f=patch_f,
#         patch_t=patch_t,
#         base_grid=(6, 8),  # doesn't have to match; it's interpolated
#     )
#
#     # -------------------------
#     # 6) Train
#     # -------------------------
#     print("\n=== Training ===")
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     model = train_model(
#         model=model,
#         train_dataset=train_ds,
#         val_dataset=val_ds,
#         epochs=epochs,
#         lr=lr,  # ignored if your train_model uses the passed optimizer lr, but harmless
#         patience=patience,
#         batch_size_train=batch_size,
#         batch_size_val=batch_size,
#         optimizer=optimizer,
#     )
#
#     # Save
#     os.makedirs("checkpoints", exist_ok=True)
#     ckpt_path = os.path.join("checkpoints", "bci_tfvitt2_6_35hz.pt")
#     torch.save(model.state_dict(), ckpt_path)
#     print(f"\nSaved model to: {ckpt_path}")
#
#     # -------------------------
#     # 7) Evaluate on test
#     # -------------------------
#     print("\n=== Test evaluation (TF) ===")
#     test_loss, test_acc, test_bal = evaluate_model_tf(model, Xte_tf, y_te, batch_size=64)
#     print("\n=== Results ===")
#     print(f"Test Loss:  {test_loss:.4f}")
#     print(f"Test Acc:   {test_acc:.4f}")
#     print(f"Test BalAcc:{test_bal:.4f}")
#
#
# if __name__ == "__main__":
#     main()

def eeg_to_mu_beta_logpower_fft(X, sfreq=250, eps=1e-8):
    """
    Whole-trial FFT -> log bandpower for mu and beta.
    X: [N, C, T]
    Returns: feats [N, C, 2] where last dim is [mu, beta]
    """
    N, C, T = X.shape
    freqs = np.fft.rfftfreq(T, d=1.0/sfreq)

    mu_band   = (freqs >= 8.0)  & (freqs < 13.0)
    beta_band = (freqs >= 13.0) & (freqs <= 30.0)

    feats = np.zeros((N, C, 2), dtype=np.float32)

    for i in range(N):
        # rFFT over time for all channels at once -> [C, F]
        fft = np.fft.rfft(X[i], axis=-1)
        power = (np.abs(fft) ** 2)  # [C, F]

        mu_power = power[:, mu_band].mean(axis=1)
        beta_power = power[:, beta_band].mean(axis=1)

        feats[i, :, 0] = np.log(mu_power + eps)
        feats[i, :, 1] = np.log(beta_power + eps)

    return feats

def normalize_train_stats(Xtr, Xva, Xte):
    """
    Z-score using TRAIN stats only, per feature.
    """
    mu = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-6
    return (Xtr - mu) / std, (Xva - mu) / std, (Xte - mu) / std

def main():
    # Load your BCI data with your existing loader
    X_tr, X_val, X_te, y_tr, y_val, y_te, ch_names = load_bci_dataset(
        data_dir="datasets/BCICIV_2a_gdf",
        tmin=0.0,
        tmax=4.0,
    )

    # 1) Whole-trial mu/beta features
    Xtr_mb = eeg_to_mu_beta_logpower_fft(X_tr, sfreq=250)
    Xva_mb = eeg_to_mu_beta_logpower_fft(X_val, sfreq=250)
    Xte_mb = eeg_to_mu_beta_logpower_fft(X_te, sfreq=250)

    print("Mu/Beta feature shapes:", Xtr_mb.shape, Xva_mb.shape, Xte_mb.shape)  # (N, 22, 2)

    # 2) Normalize (train stats only)
    Xtr_mb, Xva_mb, Xte_mb = normalize_train_stats(Xtr_mb, Xva_mb, Xte_mb)

    # 3) Flatten to (N, 22*2)
    Xtr_flat = Xtr_mb.reshape(len(Xtr_mb), -1)
    Xva_flat = Xva_mb.reshape(len(Xva_mb), -1)
    Xte_flat = Xte_mb.reshape(len(Xte_mb), -1)

    # 4) Train simple classifier
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")
    clf.fit(Xtr_flat, y_tr)

    # 5) Test
    y_pred = clf.predict(Xte_flat)
    acc = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)

    print("\n=== Whole-trial Mu/Beta baseline ===")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Balanced Acc:  {bal_acc:.4f}")

if __name__ == "__main__":
    main()